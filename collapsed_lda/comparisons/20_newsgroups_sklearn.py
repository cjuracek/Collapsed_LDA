"""Implement SKLEARN LatentDirichletAllocation with variational Bayes on the 20NewsGroup dataset

Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
to filter out useless terms early on. The posts are stripped of headers,
footers, and quoted replies. Common English words, words occurring in
only one document, and words in >= 95% of the documents are removed.
"""
from time import time

import click
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from collapsed_lda.utility.utility import *


@click.command()
@click.option("--n-iter", default=10, type=int)
@click.option("--n-topics", default=20, type=int)
def main(n_iter, n_topics):
    print(f"Fetching 20 newsgroups data...", end=" ")
    dataset = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=("headers", "footers", "quotes")
    )
    print("Done")

    data = dataset["data"]
    title_docs = {i: datum for i, datum in enumerate(data)}

    # First get the data ready similarly to our implemented example
    titles_to_tokens = {title: tokenize_doc(doc) for title, doc in title_docs.items()}

    # Remove articles whose content is 'blah blah blah'
    extra_words = [
        "maxaxaxaxaxaxaxaxaxaxaxaxaxaxax",
        "said",
        "also",
        "would",
        "get",
        "say",
        "go",
        "do",
        "one",
    ]
    titles_to_tokens = {
        title: remove_stop_words(tokens, extra_words=extra_words)
        for title, tokens in titles_to_tokens.items()
        if "blah" not in tokens
    }
    titles_to_tokens_stem = {
        title: stem_tokens(tokens) for title, tokens in titles_to_tokens.items()
    }

    # Transforming the data to a list of texts according to the required format for the count vectorizer
    data_skl = list(titles_to_tokens_stem.values())
    data_skl = [" ".join(doc) for doc in data_skl]

    t0 = time()
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer()
    tf = tf_vectorizer.fit_transform(data_skl)

    # Run the algorithm
    print(f"Fitting LDA model with n_iter={n_iter}...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=n_iter,
        learning_method="online",
        random_state=0,
        verbose=1,
    )
    lda.fit(tf)
    t1 = time()
    print(f"Done in {t1 - t0}s")

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    n_top_words = 5
    print_top_words(lda, tf_feature_names, n_top_words)

    # Get topics for a sample document
    docs_skl = list(titles_to_tokens_stem)
    doc_n = 0
    print(f"Topics in Doc {docs_skl[doc_n]}:")
    topics_spec_doc(lda, tf, n_topics, doc_n)


if __name__ == "__main__":
    main()
