'''File containing utility functions and classes.'''
import numpy as np
import progressbar
import re

''' Defines a container for information needed for alignment columns.'''
class AlignmentColumn():
    ''' Initializes an alignment column with default parameters

    Parameters:
        sequences: map from sources (e.g. human) to sequences specific to the alignment column
        seqlen: length of any sequence in this alignment column
        score: score for the alignment column
    '''
    def __init__(self):
        self.sequences = {}
        self.seqlen = 0
        self.score = 0.0


''' Collects command-line options and organizes into a dictionary.

Taken from: https://gist.github.com/dideler/2395703
'''
def getopts(argv):
    opts = {}
    while argv:
        if argv[0][0] == '-':
            opts[argv[0][1:]] = argv[1]
        argv = argv[1:]
    return opts



''' Parses a .maf file and returns a list of AlignmentColumn objects.

Arguments:
    filename (str): .maf file to be parsed
Returns:
    alignment_columns: list of AlignmentColumn objects

'''
def parse_maf(filename):
    with open(filename, "r") as f:
        alignment_columns = []
        next_alignment_column = AlignmentColumn()
        undetermined_flag = False
        lines = f.readlines()
        bar = progressbar.ProgressBar()
        for line in bar(lines):
            line = re.sub('\s+', ' ', line).strip()
            if len(line) == 0:
                if (not undetermined_flag and len(next_alignment_column.sequences) == 4):
                    alignment_columns.append(next_alignment_column)
                next_alignment_column = AlignmentColumn()
                undetermined_flag = False
            else:
                id = line[0]
                ignored = 'ieq#'
                if (id in ignored): continue
                elif (id == 'a'):
                    score = float(line.split(" ")[1][6:])
                    next_alignment_column.score = score
                else:
                    srcs = ["hg38", "panTro4", "gorGor3", "ponAbe2"]
                    src_names = ["human", "chimp", "gorilla", "orangutan"]
                    src_map = {k: v for (k, v) in zip(srcs, src_names)} 
                    [_, src, start, alignment_size, strand_id,
                        source_size, seq] = line.split(" ")
                    if ('N' in seq): undetermined_flag = True
                    for poss_src in srcs:
                        if src.startswith(poss_src):
                            next_alignment_column.sequences[src_map[poss_src]] = seq.upper()
                            next_alignment_column.seqlen = int(alignment_size)
                            break
    return alignment_columns
