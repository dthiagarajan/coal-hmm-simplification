''' Trains/validates s, u, v on a given multiple-alignment file. 

Example usage:
    python validate.py -num_iter 1 -num_trials 1 -stop_diff 0.00001 -verbose * -file chrM.maf

This will print a posterior P(z | x) over all i in [1,n], where n is the number of alignment columns.
    - 1 iteration for Baum-Welch
    - 1 validation trial
    - stop difference for Baum-Welch is 0.00001
    - verbose output to see progress of Baum-Welch (can put any character to have verbosity)
    - analyzing the multiple-alignment in chrM.maf
'''

from felsenstein import *
from hmm import *
from utils import *

import numpy as np
import sys

options = getopts(sys.argv)
num_iter = 100
if ("num_iter" in options):
    num_iter = int(options["num_iter"])
num_trials = 10
if ("num_trials" in options):
    num_trials = int(options["num_trials"])
stop_diff = 0.00001
if ("stop_diff" in options):
    stop_diff = float(options["stop_diff"])
verbose = False
if ("verbose" in options):
    verbose = True
file = "chrM.maf"
if ("file" in options):
    file = options["file"]


sequence = parse_maf(file)
print("Number of alignment columns: %d" % len(sequence))

attr = validate(sequence, num_trials=num_trials, num_iter=num_iter, verbose=verbose)
posteriors = [a[0] for a in attr]
s_values = [a[1] for a in attr]
u_values = [a[2] for a in attr]
v_values = [a[3] for a in attr]

print("Final validation results:")
print("P(z | x): %.4f +/- %.4f" % (np.mean(posteriors), np.sqrt(np.var(posteriors))))
print("s: %.4f +/- %.4f" % (np.mean(s_values), np.sqrt(np.var(s_values))))
print("u: %.4f +/- %.4f" % (np.mean(u_values), np.sqrt(np.var(u_values))))
print("v: %.4f +/- %.4f" % (np.mean(v_values), np.sqrt(np.var(v_values))))
