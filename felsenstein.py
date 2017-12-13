''' Functions for computing the probability of a topology using Felsenstein's algorithm.'''
import numpy as np


class Node():
    ''' Initializes a node with given parameters.

    Arguments:
        name: name of node (only relevant for leaves)
        left: left child (Node)
        right: right child (Node)
        branch_length: length of branch that leads to this node (float)
        branch_id: id of branch that leads to this node (int)
        probs: probability of observed bases beneath this node
                [list of 4 probs for 'atcg'] (initialized by default)
    '''

    def __init__(self, name, left, right, branch_length, branch_id):
        self.name = name
        self.left = left
        self.right = right
        self.branch_length = branch_length
        self.branch_id = branch_id
        self.probs = [None for _ in range(4)]

''' Evaluates P(b|a, t) under the Jukes-Cantor model

Arguments:
    b: base (string)
    a: base (string)
    t: branch length (time) (float)
    u: mutation rate (float, defaults to 1)
Returns:
    prob: float representing the probability P(b|a, t)
'''

def jcm(b, a, t, u=1.0):
    x_ut = np.exp(-4 * u * t / 3.0)
    if b == a:
        return 0.25 * (1 + (3 * x_ut))
    else:
        return 0.25 * (1 - x_ut)


''' Constructs the ordering of the post-order traversal of ```index```
    topology from the PSet.
Arguments:
    index: which topology to use
Returns:
    list of Nodes corresponding to post-order traversal of the topology
    branch_probs: list of 4x4 matrices, where row -> col is the a -> b in P(b |a, t_i)

'''


def initialize_topology(index):
    # Taken from Table 3
    a, b, c, a_, b_, c_ = (4.56, 2.90, 28.53, 6.11, 2.03, 27.85)
    x = (c - a - b) / 2
    x_ = (c_ - a_ - b_) / 2
    a1 = [a, a, a + b, c - x, b, x]
    a2 = [a_, a_, a_ + b_, c_ - x_, b_, x_]
    branch_lengths = np.array([a1, a2, a2, a2], dtype=float)
    
    if (index == 0 or index == 1): names = ['human', 'chimp', 'gorilla', 'orangutan']
    elif (index == 2): names = ['human', 'gorilla', 'chimp', 'orangutan']
    else: names = ['chimp', 'gorilla', 'human', 'orangutan']
    bases = 'atcg'
    branches = [0, 1, 2, 3]
    leaves = [Node(s, None, None, bl, i) for (s, i, bl) in
              zip(names, branches, branch_lengths[index, :])]
    ordering = None
    branch_probs = [np.zeros((4, 4), dtype=float) for _ in range(6)]
    
    hcp = Node(None, leaves[0], leaves[1], branch_lengths[index, 4], 4)
    hc_gp = Node(None, hcp, leaves[2], branch_lengths[index, 5], 5)
    root = Node('root', hc_gp, leaves[3], None, None)
    ordering = [leaves[0], leaves[1], hcp, leaves[2], hc_gp, leaves[3], root]

    # if (index == 0):
    #     hcp = Node(None, leaves[0], leaves[1], branch_lengths[index, 4], 4)
    #     hc_gp = Node(None, hcp, leaves[2], branch_lengths[index, 5], 5)
    #     root = Node('root', hc_gp, leaves[3], None, None)
    #     ordering = [leaves[0], leaves[1], hcp, leaves[2], hc_gp, leaves[3], root]
    # elif (index == 1):
    #     hcp = Node(None, leaves[0], leaves[1], branch_lengths[index, 4], 4)
    #     hc_op = Node(None, hcp, leaves[2], branch_lengths[index, 5], 5)
    #     root = Node('root', hc_op, leaves[3], None, None)
    #     ordering = [leaves[0], leaves[1], hcp, leaves[2], hc_op, leaves[3], root]
    # elif (index == 2):
    #     hgp = Node(None, leaves[0], leaves[2], branch_lengths[index, 4], 4)
    #     c_hgp = Node(None, leaves[1], hgp, branch_lengths[index, 5], 5)
    #     root = Node('root', c_hgp, leaves[3], None, None)
    #     ordering = [leaves[1], leaves[0], leaves[2], hgp, c_hgp, leaves[3], root]
    # else: 
    #     cgp = Node(None, leaves[1], leaves[2], branch_lengths[index, 4], 4)
    #     h_cgp = Node(None, leaves[0], cgp, branch_lengths[index, 5], 5)
    #     root = Node('root', h_cgp, leaves[3], None, None)
    #     ordering = [leaves[0], leaves[1], leaves[2], cgp, h_cgp, leaves[3], root]

    for i, M in enumerate(branch_probs):
        for r, b_1 in enumerate(bases):
            for c, b_2 in enumerate(bases):
                M[r][c] = jcm(b_1, b_2, branch_lengths[index, i])
    return ordering, branch_probs


''' Computes the likelihood of the data given the topology specified by ordering

Arguments:
    data: sequence data (dict: name of sequence owner -> sequence)
    seqlen: length of sequences
    ordering: postorder traversal of our topology
    bp: branch probabilities for the given branches
Returns:
    total_log_prob: log likelihood of the topology given the sequence data
'''
def likelihood(data, seqlen, ordering, bp):
    bases = 'ATCG'
    root_probs = []
    for i in range(seqlen):
        for node in ordering:
            # Base case
            if (node.left == None and node.right == None):
                node.probs = [1.0 if b == data[node.name]
                             [i] else 0.0 for b in bases]
                continue
            for i_a, a in enumerate(bases):
                t_i = node.left.branch_length
                t_i_id = node.left.branch_id
                t_j = node.right.branch_length
                t_j_id = node.right.branch_id
                term_1 = 0.0
                term_2 = 0.0
                for i_b, b in enumerate(bases):
                    if (t_i == 0):
                        term_1 += node.left.probs[i_b] if a == b else 0
                    else:
                        term_1 += node.left.probs[i_b] * bp[t_i_id][i_a][i_b]
                    if (t_j == 0):
                        term_2 += node.right.probs[i_b] if a == b else 0
                    else:
                        term_2 += node.right.probs[i_b] * bp[t_j_id][i_a][i_b]
                node.probs[i_a] = (term_1 * term_2)
        root_probs.append(np.sum(ordering[-1].probs) / 4.0)
    total_log_prob = np.sum(np.array([np.log(p) for p in root_probs]))
    return total_log_prob

def felsenstein(data, seqlen, index):
    ordering, probs = initialize_topology(index)
    return likelihood(data, seqlen, ordering, probs)
