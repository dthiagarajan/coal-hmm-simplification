''' Generates images for the posteriors given the s, u, v trials (assumed to be obtained after training).

Example usage:
    python generate_images.py -file chrM.maf -s 0.1 -u 0.2 -v 0.3 -verbose *

This will output images to train/chrM_posteriors.png and train/chrM_training_ll.png
    - posteriors has posterior over each alignment column for the 4 hidden states
    - training_ll has log-likelihood over the iterations, just to make sure everything works properly
        * by default, 100 iterations run to ensure s, u, v don't change too much
    - verbose output to see progress of Baum-Welch (can put any character to have verbosity)
'''

from felsenstein import *
from hmm import *
from utils import *

import numpy as np
import sys

options = getopts(sys.argv)
s = 0.1
if ("s" in options):
    s = float(options["s"])
u = 0.1
if ("u" in options):
    u = float(options["u"])
v = 0.1
if ("v" in options):
    v = float(options["v"])
verbose = False
if ("verbose" in options):
    verbose = True
file = "chrM.maf"
if ("file" in options):
    file = options["file"]

target = file.split(".")[0]

sequence = parse_maf(file)
print("Number of alignment columns: %d" % len(sequence))

generate_images(sequence, s, u, v, target, iterations=100, stop_diff=0.00001, verbose=verbose)
