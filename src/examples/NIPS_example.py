import zipfile
from time import time

import numpy as np
import pandas as pd

from src.lda.inference import get_top_n_words
from src.lda.sampler import LatentDirichletAllocation

if __name__ == "__main__":
    zf = zipfile.ZipFile("../Data/NIPS_1987-2015.csv.zip")
    nips_df = pd.read_csv(zf.open("NIPS_1987-2015.csv"))
    nips_df_red = nips_df.iloc[:, 1:].sample(frac=0.1, axis="columns")

    # Data already has stop words removed
    words = nips_df.iloc[:, 0]
    words = words.astype("str")

    titles_to_tokens = {}
    for j in range(nips_df_red.shape[1]):
        bag_of_words = []
        idx = np.nonzero(nips_df_red.iloc[:, j])
        for i in idx[0]:
            bag_of_words += [words[i]] * nips_df_red.iloc[i, j]
        titles_to_tokens[nips_df_red.columns[j]] = bag_of_words

    K = 10
    t0 = time()
    z, phi, theta = LatentDirichletAllocation(titles_to_tokens, K, 2 / K, niter=10)
    print("done in %0.3fs." % (time() - t0))

    print(get_top_n_words(phi, 5, list(words)))
