# Coal-HMM simplification - CS 4775 (Fall 2017) Final Project

This is the repo for the final project for CS 4775 (Fall 2017).

To run the code, do the following.

1. Download the relevant data by running download.py
    * ```python download.py```
2. Get mean/variance values for the posteriors, s, u, and v by running validate.py
    * indicate which file to analyze
    * set the number of iterations per validation trial
    * set the number of validation trials
    * set the stop-difference for Baum-Welch for all trials
    * see file for more details (like example usage)
3. Generate the graph of the posterior over alignment columns for the final s, u, and v values from validate.py by running generate_images.py
    * indicate which file to analyze
    * set s, u, and v as separate flags
    * see file for more details (like example usage)

Please email me at dt372@cornell.edu for any questions.
