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
a list of stopwords. Run

```make install```

To handle these commands


### - The detailed code can be found in document src 
The sampler file containing the actual function for the gibbs sampler. The utility file contains functions to prepare the data into usable tokens and titles while inference is made of only one function to print top words. The test file tests that some of our functions return the desired output
   
### - Examples on two datasets are implemented in Examples
The 20NewsGroup dataset from scikit-learn, a dataset of articles from Reuter and the NIPS dataset extracted from   https://archive.ics.uci.edu/ml/datasets/NIPS+Conference+Papers+1987-2015are used
  
### - The Data folder contains multiple data files like the one used in Examples

### - In comparisons, we compare our algorithm with existing ones using different methods such as sklearn (variational bayesian inference) and gensim (PLDA).
