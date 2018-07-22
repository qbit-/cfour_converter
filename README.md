# Cfour converter
* 2018, qbit271@protonmail.com

This is a simple script to facilitate the interoperation
between CFOUR and Gaussian software. If your program can read Gaussian
__.fch__ files, you can use this script instead of creating a new
interface for CFOUR.

The script can run 3 operations:
* prepare CFOUR input(s) from Gaussian input. Multiple sections
separated by the __--Link1--__ keyword are supported. Each CFOUR
input is placed into its own directory. The jobs are named
according to the __%Chk=__ lines found in the input file.
* run CFOUR on each input (supports resuming)
* convert results from CFOUR output to __.fch__-like files (complete
compatibility is not guaranteed, but should be easy to correct)

Extraction of information is done with the help of regular expressions,
and output is produced using Jinja2 templates

Dependencies:
* Numpy
* Jinja2

## How to use this script
You can use this script as a normal python program:

```bash
python cfour_converter.py [commands]
```
or directly by making it executable

```bash
chmod +x cfour_converter.py
./cfour_converter.py [commands]
```
### preparing CFOUR inputs from Gaussian inputs
To prepare CFOUR inputs call the script with the `prepare` argument:

```bash
./cfour_converter.py prepare gaussian_input.gjf
```
full list of options can be found by using `-h` argument

### runnung CFOUR
Call the script with the `run` subcommand. If the calculation is terminated before completion, it is possible to continue calling the script with the `--keep_going` argument. The command can be configured with the `--command` keyword:

```bash
./cfour_converter.py run --keep_going --cfour_out_prefix prefix/to/cfour/input/directories --command xcfour
```

### Converting CFOUR output to Gaussian __.fch__ files
Call the script with the `convert` subcommand. A series of __.fch__ files will be produced, one per CFOUR sub-directory. The original CFOUR directory can be removed with the `--clean_cfour` argument.

```bash
./cfour_converter.py convert --fch_out_prefix output/fch/files/here
```
