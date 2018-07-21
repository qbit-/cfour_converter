# cfour_converter
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