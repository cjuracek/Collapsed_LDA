"""Implement gensim LDA approximation in parallel on the 20NewsGroup dataset"""

from pprint import pprint
from time import perf_counter

from sklearn.datasets import fetch_20newsgroups

from collapsed_lda.comparisons.func_gensim import *
from collapsed_lda.utility import *

if __name__ == "__main__":
    K = 20
    print("Fetching 20 newsgroups dataset...", end=" ")
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

    docs_gen = list(titles_to_tokens_stem)  # list of document titles
    data_gen = list(titles_to_tokens_stem.values())  # list of words for each docs

    # Run the algorithm
    t0 = perf_counter()
    print("Running LDA...")
    lda_model, corpus = gensim_lda(K, data_gen)
    get_topic_distribution(lda_model, corpus, doc_n=0)

    doc_lda = lda_model[corpus]
    t1 = perf_counter()
    print(f"Done in {t1 - t0:.3f}s")

    # Print the key words in each topic
    pprint(lda_model.print_topics())
