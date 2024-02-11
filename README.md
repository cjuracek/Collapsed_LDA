# STA_663_Final
   ## This Repository shows the implementation of Latent Dirichlet Allocation with collapsed gibbs sampling in python by Cole Juracek and Pierre Gardan.
    
## Project Installation

### Via Poetry

[Poetry](https://python-poetry.org/) is the recommended method to install this project. Simply run:

```poetry install```

To install the project's requirements into a new virtual environment.

### Via pip

Project installation can also be done with pip. Run:

```pip install -r requirements.txt```

It is recommended to install the project requirements into a project-specific virtual environment.

### Additional Steps

In addition to the project dependencies, additional steps need to be taken to download the language model used for parsing and
a list of stopwords. Run:

```make install```

To handle these commands

# Repository Layout

TODO

# Examples

Examples on 3 datasets can be found within `collapsed_lda/examples/`:
- [20 newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) (via Scikit)
- [Reuters-21578](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html)
- [NIPS dataset](https://archive.ics.uci.edu/ml/datasets/NIPS+Conference+Papers+1987-2015)

Data used for these examples can be found in the top-level `data/` directory where appropriate

# Comparisons
In comparisons, we compare our algorithm against existing implementations using different methods such as sklearn (variational bayesian inference) and gensim (PLDA).

# Testing

TODO

### - The detailed code can be found in document src 
The sampler file containing the actual function for the gibbs sampler. The utility file contains functions to prepare the data into usable tokens and titles while inference is made of only one function to print top words