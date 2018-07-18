#!/usr/bin/env python3
"""
This module converts a  Gaussian input
files to a series of inputs for CFOUR program.
In the other way, this module produces inputs readable
by the ANCO program which resemble Gaussian output
"""

# Some tunable globals
default_cfour_out_prefix = './cfour_jobs'
default_template_prefix = './templates'
default_template_filename = 'zmat_ccsd_template.txt'
default_cfour_basis_file_path = './templates/GENBAS'
encoding = 'utf-8'

from jinja2 import Environment, FileSystemLoader
from numpy import arange
from subprocess import Popen, DEVNULL, STDOUT
import re
import mmap
import numpy as np
import os
import shutil

def get_section_bounds(data, section_separator, entry_number=0):
    """
    Scans data for section separators, counts them.
    For a given entry_number returns start and end positions
    of the section.

    Parameters
    ----------
    data : bytearray
        data to scan
    section_separator : bytearray
        section separator
    entry_number : int
        number of the needed entry. If not found, an error is raised
    Returns
    -------
    start_at : int
         the beginning of the requested section
    end_at : int
         the end of the requested section
    num_sections : int
         total number of sections
    """

    # Count sections and find the beginning and the end
    # of the needed section
    start_at = 0

    # Find the 1st location of separator, if any
    num_sections = 1
    next_at = data.find(section_separator)
    if next_at > 0:
        num_sections += 1
        next_at += len(section_separator)

    # Scan through the array and count separators
    while next_at > 0:
        if num_sections - 1 == entry_number:
            start_at = next_at
        next_at = data.find(section_separator, next_at)
        if next_at > 0:
            num_sections += 1
            next_at += len(section_separator)

    # now find the end of the section
    end_at = data.find(section_separator, start_at)
    if end_at < 0:  # the last entry
        end_at = len(data)

    return start_at, end_at, num_sections


def search_named_re_entries(data, pattern, start_at=0, end_at=None):
    """Searches for the first occurence of pattern in data,
    collects named matches into a dict

    Parameters
    ----------
    data : bytearray
          data to search in
    pattern : bytearray
          pattern (byte string)
    start_at : int, optional
          position to start the search
    end_at : int, optional
          position to end the search
    Returns
    -------
    result : dict
         dictionary containing all named matched fields
    start_at : int
         end of the match, new start position
    """
    if end_at is None:
        end_at = len(data)

    test_pattern = re.compile(
            pattern
        )
    test_match = test_pattern.search(data, start_at, end_at)
    if test_match is None:
        raise ValueError(
            test_pattern.pattern.decode(encoding) +
            ' not found at {}'.format(start_at))

    res = {key:val.decode(encoding)
           for key,val in test_match.groupdict().items() if val is not None}
    next_start_at = test_match.end(0)

    return res, next_start_at


def extract_from_gau_input(filename, entry_number=0):
    """This function extracts all necessary information
    from Gaussian input files

    Parameters
    ----------
     filename : str
             filename of the Gaussian job to extract from
     entry_number : int
             number of entry if there are multiple --Link1-- lines, default 0
    Returns
    -------
    res : dict
        dictionary containning all extracted information
    """
    res = {}

    with open(filename, 'r+') as fp:
        data = mmap.mmap(fp.fileno(), 0)

        # find the bounds of the needed section
        section_separator = '--Link1--'.encode(encoding)
        start_at, end_at, num_sections = get_section_bounds(
        data, section_separator, entry_number)
            
        # extract the chk name, which we will use as an identifier
        # for testing patterns use https://pythex.org/     :)))!!!
        chk_name_pattern = b'(%Chk=(?P<chk_name>(\S+)_Q(?P<diff_order>\d+)_(?P<filenumber>\d+)))'
        
        r, start_at = search_named_re_entries(
            data, chk_name_pattern, start_at, end_at)
        res.update(r)

        # skip commands and get molecule name, charge and multiplicity
        gau_preamble_pattern = b'(?P<command>.*)( *\n){2}((?P<molecule>\S+)\s+)(.*)( *\n){2}((?P<charge>\d),\s*(?P<multiplicity>\d))'
        
        r, start_at = search_named_re_entries(
            data, gau_preamble_pattern, start_at, end_at)
        res.update(r)
        
        # extracts element symbol, cartesian coordinates and isotope mass
        # from a single Gaussian xyz entry
        xyz_pattern = b'(?:\s*)(?P<element>\w+)(\((ISO=|Iso=)(?P<isotope>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?).*\))?(?:\s+)(?P<x>[-+]?(\d+(\.\d*)?|\.\d+)(\ [eE][-+]?\d+)?)(?:\s+)(?P<y>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?:\s+)(?P<z>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'

        # parse the first line
        r, start_at = search_named_re_entries(
            data, xyz_pattern, start_at, end_at)

        # we will store xyz information in the pyscf format
        xyz = [[r['element'], (r['x'], r['y'], r['z'])]]
        isotopes = [(r['element'], r.get('isotope', None))]

        # parse the rest
        while True:
            try:
                r, start_at = search_named_re_entries(
                    data, xyz_pattern, start_at, end_at
                )
                xyz.append(
                    [r['element'],
                     (r['x'], r['y'], r['z'])]
                    )
                isotopes.append((r['element'], r.get('isotope', None)))
            except ValueError:
                break # We have read the last line in the section

    res.update({'xyz' : xyz})
    res.update({'isotopes' : isotopes})
    return res


def render_zmat_file(datadict, filename,
                     template_filename=default_template_filename,
                     template_prefix=default_template_prefix):
    """
    Renders a ZMAT file, which contains input for the
    CFOUR program, using data from datadict

    Parameters
    ----------
    datadict : dict
             data to fill into the template file
    filename : str
             name of the file to output
    template_filename : str, default default_template_filename
             name of the template to fill
    template_prefix : str, default default_template_prefix
             path to the template folder
    """
    env = Environment(
        loader=FileSystemLoader(template_prefix),
    )
    template = env.get_template(template_filename)

    # remove isotopes if not all were provided
    # TODO: populate non-filled isotope entries with standard values
    drop_isotopes = any(entry[1] is None for entry in datadict['isotopes'])
    if drop_isotopes:
        datadict.pop('isotopes')
    
    with open(filename, 'w') as fp:
        fp.write(template.render(
            res=datadict))

def prepare_cfour_inputs_from_gau(
        gau_filename, cfour_out_prefix=default_cfour_out_prefix,
        cfour_dir_basename='job',
        cfour_basis_file_path=default_cfour_basis_file_path):
    """
    This function takes a Gaussian input, and for
    each section found creates a set of directories with
    CFOUR input files in the given path
    """

    # create prefix dir if it does not exist
    if not os.path.exists(cfour_out_prefix):
        os.makedirs(cfour_out_prefix)

    # find the number of sections
    with open(gau_filename, 'r+') as fp:
        data = mmap.mmap(fp.fileno(), 0)
        section_separator = '--Link1--'.encode(encoding)
        _, _, num_sections = get_section_bounds(
            data, section_separator)

    # for each section create a file
    for entry_number in range(num_sections):
        # parse entry
        res = extract_from_gau_input(
            gau_filename, entry_number)

        # make directory and copy basis file
        dir_name = '/'.join([
            cfour_out_prefix,
            cfour_dir_basename + "_q{}_{}".format(
                res['diff_order'], res['filenumber'])
        ])
        os.makedirs(dir_name)

        # copy basis file
        shutil.copyfile(
            cfour_basis_file_path,
            "/".join((dir_name, 'GENBAS'))
        )
        
        # render ZMAT file
        filename = '/'.join((
            dir_name, 'ZMAT'))
        render_zmat_file(res, filename)


def main():
    """
    This function can run 3 operations:
    - prepare CFOUR input from Gaussian input
    - run CFOUR on each input (supports resuming)
    - collect results from CFOUR output to files
      readable by ANCO (we mimic Gaussian outputs)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Converts between Gaussian and CFOUR inputs')
    subparsers = parser.add_subparsers(
        dest='command', metavar='command',
        help='action to perform')
    subparsers.required = True

    # parse commands to prepare
    parser_prepare = subparsers.add_parser(
        'prepare', help='prepare cfour job')

    parser_prepare.add_argument(
        'gau_filename', type=str,
        help='Gaussian input file'
    )
    parser_prepare.add_argument(
        '--job_dir_prefix', dest='cfour_out_prefix', type=str,
        default=default_cfour_out_prefix,
        help='output directory for cfour job directories'
    )
    parser_prepare.add_argument(
        '--job_dir_basename', dest='cfour_dir_basename',
        type=str, default='job',
        help='basename for cfour job directories'
    )
    parser_prepare.add_argument(
        '--basis_file', dest='cfour_basis_file_path',
        type=str, default=default_cfour_basis_file_path,
        help='path to cfour basis file'
    )

    # parse commands to run
    parser_run = subparsers.add_parser(
        'run', help='run all cfour jobs in a dir')

    parser_run.add_argument(
        '--job_dir_prefix', dest='cfour_out_prefix', type=str,
        default=default_cfour_out_prefix,
        help='directory containing cfour job directories'
    )

    parser_run.add_argument(
        '--keep_going', dest='keep_going',
        action='store_const', const=True, default=False,
        help='finish partially completed job dir hierarchy'
    )

    # parse commands to collect
    parser_collect = subparsers.add_parser(
        'collect', help='collect cfour jobs output')

    parser_collect.add_argument(
        '--job_dir_prefix', dest='cfour_out_prefix', type=str,
        default=default_cfour_out_prefix,
        help='directory containing cfour job directories'
    )

    parser_collect.add_argument(
        '--clean_cfour', dest='clean_cfour',
        action='store_const', const=True, default=False,
        help='delete cfour output after collection'        
    )

    args = parser.parse_args()
    #print(args)

    if args.command == 'prepare':
        prepare_cfour_inputs_from_gau(
            gau_filename=args.gau_filename,
            cfour_out_prefix=args.cfour_out_prefix,
            cfour_dir_basename=args.cfour_dir_basename,
            cfour_basis_file_path=args.cfour_basis_file_path
        )
        


if __name__ == '__main__':
    main()
