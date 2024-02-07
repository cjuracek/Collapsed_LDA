from time import perf_counter

import click

from collapsed_lda.lda import LatentDirichletAllocation
from collapsed_lda.utility import *


@click.command()
@click.option(
    "--data-path",
    default="data/reuters21578/reut2-000.sgm",
    help="Path to reuters data file (.sgm)",
)
def main(data_path):
    with open(data_path) as f:
        data = f.read()

    titles_to_articles = parse_sgm_file(data)
    titles_to_tokens = {
        title: tokenize_doc(doc) for title, doc in titles_to_articles.items()
    }

    # Remove articles whose content is 'blah blah blah'
    extra_words = ["reuter", "said", "also", "would"]
    titles_to_tokens = {
        title: remove_stop_words(tokens, extra_words=extra_words)
        for title, tokens in titles_to_tokens.items()
        if "blah" not in tokens
    }
    titles_to_tokens_stem = {
        title: stem_tokens(tokens) for title, tokens in titles_to_tokens.items()
    }

    t0 = perf_counter()
    lda = LatentDirichletAllocation(
        doc_to_tokens=titles_to_tokens_stem, K=5, alpha=2 / 5, beta=0.01
    )
    lda.fit(n_iter=10)
    t1 = perf_counter()
    print(f"Done in {t1 - t0:.3f} seconds")
    print(lda.get_top_n_words(n=5))


if __name__ == "__main__":
    main()
