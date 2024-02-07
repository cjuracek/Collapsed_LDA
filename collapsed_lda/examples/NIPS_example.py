import zipfile
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import pandas as pd

from collapsed_lda.lda import LatentDirichletAllocation


@click.command()
@click.option("--n-iter", default=10, type=int)
@click.option("--k", default=10, type=int)
@click.option("--data-dir", type=Path, default="./data/nips/")
def main(n_iter, k, data_dir):
    data_path = data_dir / "NIPS_1987-2015.csv"
    if not data_path.exists():
        print(f"Data path {data_path} does not exist. Extracting...", end=" ")
        zip_path = data_path.with_suffix(".csv.zip")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path.parent)
        print("Done")

    # nips_df = pd.read_csv(zf.open("NIPS_1987-2015.csv"))
    print("Reading data frame...", end=" ")
    nips_df = pd.read_csv(data_path)
    print(f"Done. Read {len(nips_df)} rows")
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

    t0 = perf_counter()
    lda = LatentDirichletAllocation(
        doc_to_tokens=titles_to_tokens, K=k, alpha=2 / k, beta=0.01
    )
    print(f"Running LDA for {n_iter} iterations...")
    lda.fit(n_iter=n_iter)
    t1 = perf_counter()
    print(f"Done in {t1 - t0:.3f}s")
    print(lda.get_top_n_words(n=5))


if __name__ == "__main__":
    main()
