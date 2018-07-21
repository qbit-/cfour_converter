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
default_cfour_template_filename = 'zmat_ccsd_template.txt'
default_cfour_basis_file_path = './templates/GENBAS'
default_fch_template_filename = 'fch_template.txt'
default_fch_out_prefix = 'fch_outs'
encoding = 'utf-8'

from jinja2 import Environment, FileSystemLoader
from numpy import arange
from subprocess import Popen, DEVNULL, STDOUT
import re
import mmap
import numpy as np
import os
import shutil

#----------------------prepare-----------------------------

def count_pattern_matches(data, pattern, start_at=0, end_at=None):
    """Counts the number pattern occurs in data"""

    if end_at is None:
        end_at = len(data)

    n_found = 0
    next_at = data.find(pattern, start_at, end_at)
    if next_at > 0:
        n_found += 1
        next_at += len(pattern)

    # Scan through the array and count separators
    while next_at > 0:
        next_at = data.find(pattern, next_at, end_at)
        if next_at > 0:
            n_found += 1
            next_at += len(pattern)

    return n_found


def get_nth_block_bounds_by_pattern(data, start_pattern, end_pattern, entry_number=0):
    """
    Returns bounds of the block by start and end patterns.
    Searches for the first entry_number occurences of the start
    pattern. No checks are done, may be confused.
    """
    start_expr = re.compile(start_pattern)
    end_expr = re.compile(end_pattern)

    start_at = 0
    for skip in range(entry_number):
        start_at = data.find(start_pattern, start_at)
        if start_at > 0:
            start_at += len(start_pattern)

    end_match = end_expr.search(data, start_at)
    end_at = end_match.end(0) if end_match is not None else -1

    return start_at, end_at


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


def search_multiline_named_entries(data, pattern,
                                   start_at=0, end_at=None):
    """Extracts named entries of a pattern by
    scanning sequentially for matches

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
    res : list
        returns a list of dictionaries containing the result
    """
    res = []

    # parse the first line
    r, start_at = search_named_re_entries(
        data, pattern, start_at, end_at)
    res.append(r)

    # parse the rest
    while True:
            try:
                r, start_at = search_named_re_entries(
                    data, pattern, start_at, end_at
                )
                res.append(r)
            except ValueError:
                break # We have read the last line in the section
    return res


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
        section_separator = b'--Link1--'
        start_at, end_at = get_nth_block_bounds_by_pattern(
            data, section_separator, section_separator,
            entry_number
        )
        if end_at == -1:  # properly handle the last block
            end_at = len(data)

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

        # search input
        table = search_multiline_named_entries(
            data, xyz_pattern, start_at, end_at)

        # we will store xyz information in the pyscf format
        # repack
        xyz = []
        isotopes = []
        for line in table:
            xyz.append([line['element'], (line['x'], line['y'], line['z'])])
            isotopes.append((line['element'], line.get('isotope', None)))

        res.update({'xyz' : xyz})
        res.update({'isotopes' : isotopes})

    return res


def render_zmat_file(datadict, filename,
                     template_filename=default_cfour_template_filename,
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
    template_filename : str, default default_cfour_template_filename
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
        cfour_basis_file_path=default_cfour_basis_file_path,
        verbose=False):
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
        section_separator = b'--Link1--'
        num_sections = count_pattern_matches(
            data, section_separator) + 1

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

    if verbose:
        print('Created {} input directories in {}'.format(
            num_sections, cfour_out_prefix))

#-------------------- run -----------------------------

def check_clean(job_dir):
    """
    Checks if the cfour job_dir is clean
    """
    keep_files = ['ZMAT', 'GENBAS']

    for name in os.listdir(job_dir):
        if name not in keep_files:
            return False
    return True
    
def check_complete(job_dir, output_name='OUTPUT'):
    """
    Simple check if cfour job is complete

    Parameters
    ----------
    job_dir : str
           path to the job directory containing the ZMAT file
    out_filename : str, default OUTPUT
           filename of the output
    Returns
    -------
    status : bool
           Completion status
    """
    # Check the last line in the output
    complete_pattern = re.compile(
        b'(?:--executable xjoda finished with status\s*)(?P<exitcode>\d)'
    )
    try:
        with open('/'.join((job_dir, output_name)), 'r+') as fp:
            complete_match = complete_pattern.search(
                fp.readlines()[-1].encode(encoding)
            )
    except:
        return False

    if complete_match is not None:
        if int(complete_match.group('exitcode')) != 0:
            return False
    else:
        return False
    
    # Additionally, check if job produced the properties files
    required_files = ['FCM', 'FCMFINAL']
    return all(os.path.isfile('/'.join((job_dir, filename)))
               for filename in required_files)


def cleanup_incomplete_job_dir(job_dir):
    """
    Removes all temporary files from a job directory
    """
    keep_files = ['ZMAT', 'GENBAS', 'OUTPUT']

    for name in os.listdir(job_dir):
        if name not in keep_files:
            full_name = '/'.join((job_dir, name))
            if os.path.isfile(full_name):
                os.remove(full_name)

            elif os.path.isdir(full_name):
                shutil.rmtree(full_name)


def run_job(job_dir, out_filename='OUTPUT'):
    """
    Run cfour in a specified directory, collect results

    Parameters
    ----------
    job_dir : str
           path to the job directory containing the ZMAT file
    out_filename : str, default OUTPUT
           filename of the output
    """
    with open("/".join((job_dir, out_filename)), 'w') as fp:
        p = subprocess.Popen(['xcfour'], cwd=job_dir,
                             stderr=subprocess.STDOUT,
                             stdout=fp)
    p.wait()


def run_all(
        cfour_out_prefix=default_cfour_out_prefix,
        keep_going=False, verbose=False):
    """
    Runs all jobs in a directory with possible restart

    Parameters
    ----------
    cfour_out_prefix : str
              directory containing cfour job directories
    keep_going : bool, default False
              try to continue job execution
    verbose : bool, default False
              print progress
    """
    if verbose:
        print('Running jobs in dir: {}'.format(cfour_out_prefix))

    # Find out how many jobs we potentially have
    n_jobs = 0
    for name in os.listdir(cfour_out_prefix):
        full_name = '/'.join((cfour_out_prefix, name))
        if os.path.isdir(full_name):
            n_jobs += 1

    # Walk the dir tree and run jobs
    job_num = 0
    for name in os.listdir(cfour_out_prefix):
        full_name = '/'.join((cfour_out_prefix, name))
        if os.path.isdir(full_name):
            if verbose:
                print('[{}/{}] {}..'.format(
                    job_num+1, n_jobs, full_name), end="")
            is_clean = check_clean(full_name)
            if is_clean:
                if verbose:
                    print('clean, run..')
                run_job(full_name)
            else:
                is_complete = check_complete(full_name)
                if is_complete:
                    if verbose:
                        print('complete, skip')
                    job_num += 1
                    continue
                else:
                    if verbose:
                        print(
                            'incomplete', end="")
                    if keep_going:
                        cleanup_incomplete_job_dir(
                            full_name
                        )
                        if verbose:
                            print('..force run..')
                        run_job(full_name)
                    else:
                        if verbose:
                            print('')
                            break
            job_num += 1

    if verbose:
        if job_num == n_jobs:
            print('All done!')
        else:
            print('Check for errors')


#---------------------convert----------------------------

def extract_from_cfour_out(job_dir, output_name='OUTPUT'):
    """
    Extracts information from the CFOUR output files
    """

    res = {}
    with open('/'.join((job_dir, output_name)), 'r+') as fp:
        data = mmap.mmap(fp.fileno(), 0)

        # Extract energy from output
        energy_pattern = b'(?:CCSD\(T\) energy\s+)(?P<energy>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'

        r, _ = search_named_re_entries(
            data, energy_pattern, 0, len(data)
        )
        res.update(r)

        # Now extract coordinates
        # find block bounds
        start_at, end_at = get_nth_block_bounds_by_pattern(
            data,
            b'(Coordinates \(in bohr\)\s*)',
            b'(-+\s+)(\s*Interatomic distance matrix)'
        )

        # find entries
        xyz_pattern = b'(?:\s*)(?P<element>\w+)(?:\s+)(?:\d+)(?:\s+)(?P<x>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?:\s+)(?P<y>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?:\s+)(?P<z>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?:\s+)'
        table = search_multiline_named_entries(
            data, xyz_pattern, start_at, end_at)

        # repack
        xyz = []
        for row in table:
            xyz.append([row['element'], (row['x'], row['y'], row['z'])])
        res.update({'xyz' : xyz})

        # save the number of atoms
        natoms = len(xyz)
        res.update({'natoms' : natoms})
        
    # Now extract Cartesian gradient
    with open('/'.join((job_dir, 'GRD'))) as fp:
        lines = fp.readlines()
        grd = []
        for row, line in zip(xyz, lines[-natoms:]):
            mass, x, y, z = line.rstrip('\n').split()
            grd.append([row[0], (x, y, z)])

        res.update({'grd' : grd})
        
    # Now extract hessian
    try:
        with open('/'.join((job_dir, 'FCMFINAL'))) as fp:
            lines = fp.readlines()
            raw = []
            for line in lines[1:]:
                raw.append(line.rstrip('\n').split())
            hess = np.array(raw).reshape([3*natoms, 3*natoms])
            # order along columns: Atom1.x Atom1.y Atom1.z Atom2.x Atom2.y...
            # order along rows : Atom1.x Atom1.y Atom1.z Atom2.x Atom2.y ...
            # extract the lower triangle and store it
            idx = np.tril_indices_from(hess)
            res.update({'hess' : hess[idx]})

    except FileNotFoundError:
        if verbose:
            print('No hessian found')
    
    # Now extract dipole moment
    try:
        with open('/'.join((job_dir, 'DIPOL'))) as fp:
            line = fp.readline()
            dipole = np.array(line.rstrip('\n').split())

            res.update({'dipole' : dipole})
    except FileNotFoundError:
        if verbose:
            print('No dipole moment found')
    
    # Now extract dipole gradient
    try:
        with open('/'.join((job_dir, 'DIPDER'))) as fp:
            lines = fp.readlines()
            Ex, Ey, Ez = [], [], []
            # Read 3 blocks, corresponding to all X coordinates, all
            # Y coordinates and all Z coordinates
            for block in range(3):
                # the blocks are of length natoms interspersed with
                # single lines containing garbage
                text_slice = lines[
                    block*natoms+1+block:(block+1)*natoms+1+block]
                for line in text_slice:
                    raw = line.rstrip('\n').split()
                    Ex.append(raw[1])
                    Ey.append(raw[2])
                    Ez.append(raw[3])

            # Repack to order along columns: Ex Ey Ez
            # order along rows: Atom1.x Atom1.y Atom1.z Atom2.x Atom2.y ....
            dipder = np.zeros([3*natoms, 0])
            for E in [Ex, Ey, Ez]:
                accum = []
                for elem in zip(
                        E[:natoms],
                        E[natoms:2*natoms],
                        E[2*natoms:3*natoms]):
                    for el in elem:
                        accum.append(el)
                dipder = np.column_stack(
                    (dipder, accum))

            res.update({'dipder' : dipder})
    except FileNotFoundError:
        if verbose:
            print('Dipole derivatives not found')
            
    # Now extract polarizabilities
    try:
        with open('/'.join((job_dir, 'POLAR'))) as fp:
            lines = fp.readlines()
            polar = np.zeros([0, 3])
            for line in lines:
                polar = np.row_stack(
                    (polar,
                     line.rstrip('\n').split())
                    )

            res.update({'polar' : polar})
    except FileNotFoundError:
        if verbose:
            print('No polarizabilities found')

    return res


def render_fch_file(datadict, filename,
                    template_filename=default_fch_template_filename,
                    template_prefix=default_template_prefix):
    """
    Renders an FCH file, which resembles output from Gaussian
    and can be fad to ANCO program, using data from datadict

    Parameters
    ----------
    datadict : dict
             data to fill into the template file
    filename : str
             name of the file to output
    template_filename : str, default default_fch_template_filename
             name of the template to fill
    template_prefix : str, default default_template_prefix
             path to the template folder
    """
    env = Environment(
        loader=FileSystemLoader(template_prefix),
    )
    template = env.get_template(template_filename)

    with open(filename, 'w') as fp:
        fp.write(template.render(
            res=datadict))

def convert_cfour_to_fch(
        cfour_out_prefix=default_cfour_out_prefix,
        fch_out_prefix=default_fch_out_prefix,
        clean_cfour=False, keep_going=False, verbose=False):
    """
    Converts a directory structure containing CFOUR job directories
    to a series of Gaussian fch-like files
    """
    if verbose:
        print('Converting CFOUR outputs in dir: {}'.format(cfour_out_prefix))

    # Ensure the output directory exists
    if not os.path.isdir(fch_out_prefix):
        os.mkdir(fch_out_prefix)
        if verbose:
            print('Creating: {}'.format(fch_out_prefix))

    # Find out how many job outputs we potentially have
    n_jobs = 0
    for name in os.listdir(cfour_out_prefix):
        full_name = '/'.join((cfour_out_prefix, name))
        if os.path.isdir(full_name):
            n_jobs += 1

    # Walk the dir tree and extract jobs one at a time
    job_num = 0
    for name in os.listdir(cfour_out_prefix):
        full_cfour_dir_name = '/'.join((cfour_out_prefix, name))
        if os.path.isdir(full_cfour_dir_name):
            if verbose:
                print('[{}/{}] {}..'.format(
                    job_num+1, n_jobs, full_cfour_dir_name), end="")
            is_complete = check_complete(full_cfour_dir_name)
            if is_complete:
                if verbose:
                    print('complete, converting'.format(
                        full_cfour_dir_name), end='')
                full_fch_name = '/'.join((
                    fch_out_prefix, name + '.fch'))
                if os.path.exists(full_fch_name):
                    if not keep_going:
                        if verbose:
                            print('..{} exists, aborting'.format(
                                full_fch_name
                            ))
                        break    
                res = extract_from_cfour_out(full_cfour_dir_name)
                render_fch_file(res, full_fch_name)
                if verbose:
                    print('')
            else:
                if verbose:
                    print(
                        'incomplete'.format(full_cfour_dir_name), end="")
                if keep_going:
                    if verbose:
                        print('..skip')
                    pass
                else:
                    if verbose:
                        print('')
                        break
            job_num += 1

    all_done = (job_num == n_jobs)

    if clean_cfour:
        if all_done:
            if verbose:
                print('Removing {}..'.format(cfour_out_prefix))
            #shutil.rmtree(cfour_out_prefix)
        else:
            if verbose:
                print(
                    'Not removing {} as errors were detected'.format(
                        cfour_out_prefix
                    ))
    if verbose:
        if all_done:
            print('All done!')
        else:
            print('Check for errors')


#---------------------wrapper----------------------------

def main():
    """
    This function can run 3 operations:
    - prepare CFOUR input from Gaussian input
    - run CFOUR on each input (supports resuming)
    - convert results from CFOUR output to files
      readable by ANCO (we mimic Gaussian outputs)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Converts between Gaussian and CFOUR inputs')

    parser.add_argument(
        '--verbose', dest='verbose',
        action='store_const', const=True, default=False,
        help='display additional information'
    )

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

    # parse commands to convert
    parser_convert = subparsers.add_parser(
        'convert', help='convert cfour jobs output')

    parser_convert.add_argument(
        '--job_dir_prefix', dest='cfour_out_prefix', type=str,
        default=default_cfour_out_prefix,
        help='directory containing cfour job directories'
    )

    parser_convert.add_argument(
        '--fch_out_prefix', dest='fch_out_prefix',
        default=default_fch_out_prefix,
        help='directory to output fch files'        
    )

    parser_convert.add_argument(
        '--clean_cfour', dest='clean_cfour',
        action='store_const', const=True, default=False,
        help='delete cfour output after conversion'        
    )

    parser_convert.add_argument(
        '--keep_going', dest='keep_going',
        action='store_const', const=True, default=False,
        help='continue even if errors were detected'        
    )

    args = parser.parse_args()
    # strip a stray trailing / in the always present
    # path parameter
    args.cfour_out_prefix = args.cfour_out_prefix.rstrip('/')

    if args.command == 'prepare':
        prepare_cfour_inputs_from_gau(
            gau_filename=args.gau_filename,
            cfour_out_prefix=args.cfour_out_prefix,
            cfour_dir_basename=args.cfour_dir_basename,
            cfour_basis_file_path=args.cfour_basis_file_path,
            verbose=args.verbose
        )

    elif args.command == 'run':
        run_all(
            cfour_out_prefix=args.cfour_out_prefix,
            keep_going=args.keep_going,
            verbose=args.verbose
        )

    elif args.command == 'convert':
        convert_cfour_to_fch(
            cfour_out_prefix=args.cfour_out_prefix,
            fch_out_prefix=args.fch_out_prefix,
            clean_cfour=args.clean_cfour,
            keep_going=args.keep_going,
            verbose=args.verbose
        )

if __name__ == '__main__':
    main()
