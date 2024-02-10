"""Implement SKLEARN LatentDirichletAllocation with varitional Bayes on the Reuter dataset
Ref: Olivier Grisel
     Lars Buitinck
     Chyi-Kwei Yau
"""
from time import perf_counter

import click
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from collapsed_lda.comparisons.print_sklearn import *
from collapsed_lda.utility import *


@click.command()
@click.option("--n-topics", default=5, type=int)
@click.option("--n-top-words", default=10, type=int)
@click.option("--n-iter", default=10, type=int)
def main(n_topics, n_top_words, n_iter):
    # Get the data ready similarly to our implemented example
    with open("data/reuters21578/reut2-000.sgm") as f:
        data = f.read()

    title_docs = parse_sgm_file(data)
    titles_to_tokens = {title: tokenize_doc(doc) for title, doc in title_docs.items()}
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

    # Transform the data to a list of texts according to the required format for the CountVectorizer
    data_skl = list(titles_to_tokens_stem.values())
    data_skl = [" ".join(doc) for doc in data_skl]

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    count_vectorizer = CountVectorizer()
    doc_word_matrix = count_vectorizer.fit_transform(data_skl)

    # Run the algorithm
    t0 = perf_counter()
    print(f"Fitting LDA model with n_iter={n_iter}...")
    lda_skl = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        random_state=0,
        verbose=1,
        max_iter=n_iter,
    )
    lda_skl.fit(doc_word_matrix)
    t1 = perf_counter()
    print(f"Done in {t1 - t0:.3f}s\n")

    print("Top words by topic:")
    tf_feature_names = count_vectorizer.get_feature_names_out()
    print_top_words(lda_skl, tf_feature_names, n_top_words)

    # Get topic distributions for a sample doc
    doc_titles = list(titles_to_tokens_stem)
    doc_n = 0
    print(f'Topic distribution for Doc "{doc_titles[doc_n]}":')
    topics_spec_doc(lda_skl, doc_word_matrix, n_topics, doc_n)


if __name__ == "__main__":
    main()
