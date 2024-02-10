"""Implement gensim LDA approximation in parallel on the Reuters dataset. To be run from the repo root."""
from pprint import pprint
from time import perf_counter

import click

from collapsed_lda.utility.gensim import gensim_lda, get_topic_distribution
from collapsed_lda.utility.utility import *


@click.command()
@click.option("--k", type=int, default=5)
@click.option("--n-top-words", type=int, default=10)
def main(k, n_top_words):
    # Get the data ready similarly to our implemented example
    with open("data/reuters21578/reut2-000.sgm") as f:
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

    doc_titles = list(titles_to_tokens_stem)
    doc_tokens = list(titles_to_tokens_stem.values())

    # Run the algorithm
    t0 = perf_counter()
    lda_model, corpus = gensim_lda(k, doc_tokens)
    doc_n = 0
    get_topic_distribution(lda_model, corpus, doc_n)
    t1 = perf_counter()
    print(f"Done in {t1 - t0:.3f}s")

    # Print the keywords in each topic
    pprint(lda_model.print_topics())

    print(f"\nTopic distribution for Doc {doc_titles[doc_n]}:")
    get_topic_distribution(lda_model, corpus, doc_n)


if __name__ == "__main__":
    main()
