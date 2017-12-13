'''Utility functions for the HMM implementation of the Coal-HMM.'''
from felsenstein import *
from sklearn.preprocessing import normalize
from utils import *

import matplotlib.pyplot as plt
import numpy as np

'''Sums over a vector of log probabilities in a numerically stable manner.
Arguments:
    a: log probabilities (float vector)
Returns: 
    summed log probabilities
'''
def sum_vec_log_probs(a):
    output = a[0]
    for i in range(1, a.size):
        if (output > a[i]):
            output = output + np.log1p(np.exp(a[i] - output))
        else:
            output = a[i] + np.log1p(np.exp(output - a[i]))
    return output


''' Converts a dictionary of dictionaries (K x K) to a square matrix.
If the set of keys is K, then the dictionary should have all entries in K as a
key, and each dictionary within the overarching dictionary should have an entry
for each element of K, e.g. if K = {a, b}, the format of the given dictionary
should be:
    {a: {a: ..., b: ...}, b: {a: ..., b: ...}}

Arguments:
    d: dictionary in specified format
Returns:
    ordering: ordering of the keys
    indexing: mapping of index to hidden state
    mat: matrix of probabilities
'''

def convert_trans_to_matrix(probs):
    mat = np.array([[0 for _ in range(len(probs))] for _ in range(len(probs))],
                   dtype=float)
    ordering = {}
    indexing = {}
    for i, k in enumerate(probs):
        ordering[k] = i
        indexing[i] = k
    for a in probs:
        for b in probs[a]:
            mat[ordering[a]][ordering[b]] = probs[a][b]
    return ordering, indexing, mat

''' Generates the transition probabilities table given the parameters.

Arguments:
    s, u, v: parameters as described in the parameters
Returns:
    transition_probabilities (dict of log-probabilities)
'''
def get_transition_probabilities(s, u, v):
    transition_probabilities = {
        'HC1': {'HC1': np.log(1 - (3*s)), 'HC2': np.log(s), 'HG': np.log(s), 'CG': np.log(s)},
        'HC2': {'HC1': np.log(u), 'HC2': np.log(1 - (u + (2*v))), 'HG': np.log(v), 'CG': np.log(v)},
        'HG': {'HC1': np.log(u), 'HC2': np.log(v), 'HG': np.log(1 - (u + (2*v))), 'CG': np.log(v)},
        'CG': {'HC1': np.log(u), 'HC2': np.log(v), 'HG': np.log(v), 'CG': np.log(1 - (u + (2*v)))}
    }
    return transition_probabilities


''' Generates the initial probabilities table given the parameters.

Arguments:
    s, u, v: parameters as described in the parameters
Returns:
    ip (dict of log-probabilities)
'''
def get_initial_probabilities(s, u, v):
    psi = 1.0 / (1 + ((3 * s) / u))
    ip = {'HC1': np.log(psi), 'HC2': np.log((1.0 - psi) / 3),
            'HG': np.log((1.0 - psi) / 3), 'CG': np.log((1.0 - psi) / 3)}
    return ip


''' Calculates the emission matrix for all possible 1-length alignment columns
    without gaps using Felsenstein's algorithm and the Jukes-Cantor model.

Returns:
    k_ordering: ordering of the hidden states
    l_ordering: ordering of the 1-length alignment columns
    mat: matrix of log-probabilities (4 by 4 by 4 by 4 by 4; hidden state -> etc.)
'''
def compute_emiss_matrix():
    hidden_states = ['HC1', 'HC2', 'HG', 'CG']
    k_ordering = {h: i for i, h in enumerate(hidden_states)}
    l_ordering = {b: i for i, b in enumerate('ATCG')}
    row_len = 0
    mat = np.zeros((4, 4, 4, 4, 4), dtype=float)
    for i, h in enumerate(hidden_states):
        index = 0
        for i_1, a in enumerate('ATCG'):
            for i_2, b in enumerate('ATCG'):
                for i_3, c in enumerate('ATCG'):
                    for i_4, d in enumerate('ATCG'):
                        input = {'human': a, 'chimp': b,
                                 'gorilla': c, 'orangutan': d}
                        mat[i, i_1, i_2, i_3, i_4] = felsenstein(input, 1, i)
    return k_ordering, l_ordering, mat

''' Computes the probabilities for each alignment column using Felsenstein's algorithm.

Arguments:
    acs: list of alignment columns
    emiss_matrix: 4x4x4x4x4 matrix with probabilities of each 1-length
                  sequence alignment for each hidden state
    h_ordering: ordering of hidden states for emiss_matrix
    e_ordering: ordering of 1-length sequence alignment for emiss_matrix
Returns:
    emiss_probs: 4xn matrix with log-probabilities for each alignment column,
                 where n is the number of alignment columns. The sum of the
                 probabilities in each column (in the normal space) will be 1.
'''
def compute_emiss_probs(acs, emiss_matrix, h_ordering, e_ordering):
    emiss_probs = np.zeros((4, len(acs)), dtype = float)
    for col, ac in enumerate(acs):
        for row in range(4):
            probs = []
            for i in range(ac.seqlen):
                if (ac.sequences['human'][i] in '-N' or
                    ac.sequences['chimp'][i] in '-N' or
                    ac.sequences['gorilla'][i] in '-N' or
                    ac.sequences['orangutan'][i] in '-N'): continue

                h_i = e_ordering[ac.sequences['human'][i]]
                c_i = e_ordering[ac.sequences['chimp'][i]]
                g_i = e_ordering[ac.sequences['gorilla'][i]]
                o_i = e_ordering[ac.sequences['orangutan'][i]]
                probs.append(emiss_matrix[row, h_i, c_i, g_i, o_i])
            ''' Note that this sum might be volatile due to underflow. '''
            emiss_probs[row, col] = np.sum(probs)
    D = np.zeros((emiss_probs.shape[1], 1))
    for i in range(emiss_probs.shape[1]):
        D[i] = sum_vec_log_probs(emiss_probs[:, i])
    for i in range(emiss_probs.shape[1]):
        emiss_probs[:, i] = np.subtract(emiss_probs[:, i], D[i])
    return emiss_probs


''' Outputs the forward and backward probabilities of a given observation.
Arguments:
    obs: the alignment columns (AlignmentColumn list)
    trans_probs: the transition log-probabilities (dictionary of dictionaries)
    emiss_probs: the emission log-probabilities for each alignment column (matrix)
    init_probs: the log-probabilities of each hidden state initally (dictionary)
Returns:
    list of tuples (F, t_f, B, t_b, R), where
        F: matrix of forward probabilities
        t_f: P(obs) as per forward algorithm
        B: matrix of backward probabilities
        t_b: P(obs) as per backward algorithm
        R: matrix of posterior probabilities
'''
def forward_backward(obs, trans_probs, emiss_probs, init_probs):
    hs_to_index, index_to_hs, tpm = convert_trans_to_matrix(trans_probs)
    output = []
    N = len(obs)
    K = len(trans_probs)
    F = np.array([[0 for _ in range(N)] for _ in range(K)], dtype=float)
    B = np.array([[0 for _ in range(N)] for _ in range(K)], dtype=float)

    # Initialization
    for h in hs_to_index:
        i = hs_to_index[h]
        F[i, 0] = init_probs[h] + emiss_probs[i, 0]
    B[:, -1] = 0

    # Iteration
    for i in range(1, N):
        f_i = i
        b_i = N - i - 1
        for h in hs_to_index:
            k = hs_to_index[h]

            # Forward calculation
            f_a = (F[:, f_i - 1])
            f_b = (tpm[:, k])
            f_check = sum_vec_log_probs(np.add(f_a, f_b))
            ep_f = emiss_probs[k, f_i]
            F[k, f_i] = ep_f + f_check

            # Backward calculation
            b_a = (B[:, b_i + 1])
            b_b = (tpm[k, :].T)
            ep_b = (emiss_probs[:, b_i + 1])
            B[k, b_i] = sum_vec_log_probs(np.add(np.add(b_a, b_b), ep_b))

    # Total probabilities calculation
    t_f = sum_vec_log_probs(F[:, -1])
    b_a = B[:, 0]
    b_b = np.array([init_probs[k] for k in init_probs]).T
    ep_b = (emiss_probs[:, 0])
    t_b = sum_vec_log_probs(np.add(np.add(b_a, b_b), ep_b))

    R = np.add(F, B)
    D = np.zeros(R.shape[1])
    for i in range(R.shape[1]):
        D[i] = sum_vec_log_probs(R[:, i])
    for i in range(R.shape[1]):
        R[:, i] = np.exp(np.subtract(R[:, i], D[i]))
    return (F, t_f, B, t_b, R)


''' Performs 1 EM step.

Arguments:
    fb_output: the output of the forward-backward implementation above
    obs: the sequence in question
    tp: transition probabilities in the log space
    ep: emission probabilities in the log space of the observations
    ip: initalization probabilities in the log space
Returns:
    s, u, v: parameters of the transition probabilities
    tp: updated transition probabilities, in the log space
    ip: updated initial probabilities, in the log space
'''
def em(fb_output, obs, tp, ep, ip):
    hs_to_index, index_to_hs, tpm = convert_trans_to_matrix(tp)
    (F, t_f, B, t_b, R) = fb_output
    # E step
    A = np.zeros((4, 4))
    E = np.zeros((4, 256))
    for k in ['HC1', 'HC2', 'HG', 'CG']:
        for l in ['HC1', 'HC2', 'HG', 'CG']:
            i_k = hs_to_index[k]
            i_l = hs_to_index[l]
            a_kl = tpm[i_k, i_l]
            f = F[i_k, :-1]
            b = B[i_l, 1:]
            e = np.array(ep[i_l, 1:])
            A[i_k, i_l] = sum_vec_log_probs(f + b + e + a_kl - t_f)
        '''Not doing EM for emission probabilities.'''

    # M step
    for i in range(A.shape[0]):
        A[i, :] = (np.subtract(A[i, :], sum_vec_log_probs(A[i, :])))
    '''Not doing EM for emission probabilities.'''
    def update(tp, ep, ip, A, E):
        ''' Update s, u, v here. '''
        A_e = np.exp(A)
        s = np.sum(A_e[0, 1:]) / (3.0 * np.sum(A_e[0, :]))
        u = np.sum(A_e[1:, 0]) / (np.sum(A_e[1:, :]))
        val = ((A_e[1, 1] + A_e[2, 2] + A_e[3, 3]) / (np.sum(A_e[1:, :])))
        v = (1 - val - u) / 2.0

        return (s, u, v), get_transition_probabilities(s, u, v), get_initial_probabilities(s, u, v)
    return update(tp, ep, ip, A, E)


''' Returns the sequence of most-likely hidden states
    given the posterior probabilities matrix, calclulated
    using the forward-backward algorithm.

Arguments:
    R: matrix of posterior probabilities (see forward-backward for impl.)
        (matrix)
Returns:
    sequence: list of hidden states (list of strings)
    log_prob: log-probability of sequence (double)

'''
def decode_posterior(R):
    indexing = ['HC1', 'HC2', 'HG', 'CG']
    states = np.argmax(R, axis=0)
    sequence = [indexing[s] for s in states]
    log_prob = np.sum(np.log([R[s_i, i] for i, s_i in enumerate(states)]))
    return sequence, log_prob

''' Trains the parameters ```s, u, v``` until a valid stopping condition is reached.

Arguments:
    sequence: list of AlignmentColumn objects
    s, u, v: parameters of the transition probabilities (floats)
    iterations: number of iterations to run (defaults to 100). If
                set to None, will run until training update results in
                a log-likelihood difference less than stop_diff. (None or int)
    stop_diff: stopping condition for training, i.e. the minimum
               log-likelihood increase allowed between iterations
               before stopping (float)
    verbose: if True, print information about each training iteration
    target: name of target region being analyzed
    output_img: whether or not to write images to file, or not at all
Returns:
    log_prob: likelihood of the observations alone after training
    log_posterior_prob: log-likelihood of all MAP states for each alignment column, combined 
    posterior_decoding: MAP estimate of genealogy for each alignment column,
                        based only on forward-backward probability calculations
                        (see paper for more details) (list of strings)
    trans_params: values of (s, u, v) over the training iterations (list of 3-tuples)
    s, u, v: parameters of the transition probabilities

This will also output two images to the train/ directory:
    training_ll.png: the log-likelihood at each iteration while training
    posteriors.png: the posteriors for each hidden state after training,
                    for each alignment column
'''
def train(sequence, s, u, v, iterations=100, stop_diff=0.0001, verbose=False, target=None, output_img = True):
    if (verbose):
        print("Setting up probability matrices.")
    tp = get_transition_probabilities(s, u, v)
    ip = get_initial_probabilities(s, u, v)
    h_ordering, e_ordering, m = compute_emiss_matrix()
    ep = compute_emiss_probs(sequence, m, h_ordering, e_ordering)
    psi = 1.0 / (1 + ((3 * s) / u))

    (F, curr_log_prob, B, t_b, R) = forward_backward(sequence, tp, ep, ip)
    log_likelihoods = [curr_log_prob]
    iter = 0
    no_iter_flag = False
    if (iterations == None):
        no_iter_flag = True
        iterations = iter + 1
    curr_diff = 1
    curr_diff_diff = 0
    trans_params = [(s, u, v)]
    while (curr_diff > stop_diff and iter < iterations):
        if verbose:
            print("Iteration %d log-likelihood: %.6f" % (iter, curr_log_prob))
        (s, u, v), tp, ip = em((F, curr_log_prob, B, t_b, R), sequence, tp, ep, ip)
        trans_params.append((s, u, v))
        (F, log_prob, B, t_b, R) = forward_backward(sequence, tp, ep, ip)
        curr_diff_diff = (log_prob - curr_log_prob) - curr_diff
        curr_diff = log_prob - curr_log_prob
        assert(curr_diff > 0)
        if verbose:
            print("Iteration %d likelihood increase:  %.10f" % (iter, curr_diff))
            print("Iteration %d likelihood-diff increase:  %.10f" % (iter, curr_diff_diff))
            print("-----------------")
        curr_log_prob = log_prob
        log_likelihoods.append(curr_log_prob)
        iter += 1
        if (no_iter_flag): iterations += 1
    if verbose:
        print("Final (iteration %d) log-likelihood: %.6f" % (iter, curr_log_prob))
    s = np.exp(tp['HC1']['HC2'])
    u = np.exp(tp['HC2']['HC1'])
    v = np.exp(tp['HG']['HC2'])
    
    if output_img:
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Marginalized Log-likelihood")
        ax.set_title(str(target) + " Log-likelihood While Training")
        plt.plot(range(iter+1), log_likelihoods, 'r-')
        plt.savefig("train/" + str(target) + "_training_ll.png")
        plt.clf()

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True, figsize=(20,20))
        ax4.set_xlabel("Alignment Column")
        ax1.set_title(str(target) + " Hidden State Posteriors")
        ax1.plot(range(R.shape[1]), R[0, :])
        ax1.set_ylabel("HC1 posterior")
        ax2.plot(range(R.shape[1]), R[1, :])
        ax2.set_ylabel("HC2 posterior")
        ax3.plot(range(R.shape[1]), R[2, :], color='r')
        ax3.set_ylabel("HG posterior")
        ax4.plot(range(R.shape[1]), R[3, :], color='b')
        ax4.set_ylabel("CG posterior")
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.savefig("train/" + str(target) + "_posteriors.png")

    posterior_decoding, posterior_prob = decode_posterior(R)
    return curr_log_prob, posterior_prob, posterior_decoding, trans_params, s, u, v

def validate(sequence, num_trials = 20, num_iter = 1000, verbose=False):
    attr = []
    for iter in range(num_trials):
        s = np.random.rand(1, 1)[0][0] / 3
        u = np.random.rand(1, 1)[0][0] / 3
        v = np.random.rand(1, 1)[0][0] / 3
        print("Validating on trial %d/%d with (s = %.3f, u = %.3f, v = %.3f)." % (iter, num_trials, s, u, v))
        p_x, posterior_x, posterior_decoding, _, s, u, v, = train(sequence, s, u, v, iterations=num_iter, stop_diff=0.00001, output_img=False, verbose=verbose)
        attr.append((posterior_x, s, u, v))
    return attr


def generate_images(sequence, s, u, v, target, iterations=100, stop_diff=0.00001, verbose=True):
    p_x, posterior_x, posterior_decoding, _, s, u, v, = train(sequence, s, u, v, iterations=iterations, stop_diff=0.00001, target=target, verbose=verbose)
