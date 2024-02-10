from time import perf_counter

import click
import spacy
from sklearn.datasets import fetch_20newsgroups
from spacy.lang.en.stop_words import STOP_WORDS

from collapsed_lda.lda import LatentDirichletAllocation
from collapsed_lda.utility.utility import *


@click.command()
@click.option("--fast-dev-run", is_flag=True, default=False)
@click.option("--k", default=20, type=int)
@click.option("--alpha", default=2 / 20, type=float)
@click.option("--n-iter", type=int, default=10)
def main(fast_dev_run, k, alpha, n_iter):
    print("Loading data from 20 newsgroups dataset... ", end="")
    dataset = fetch_20newsgroups(
        shuffle=True, random_state=1, remove=("headers", "footers", "quotes")
    )
    if fast_dev_run:
        dataset["data"] = dataset["data"][:100]

    print(f"Done. Loaded {len(dataset.data)} items")

    # Remove empty / white space documents
    non_empty_data = [
        article for article in dataset["data"] if article and not article.isspace()
    ]

    # Process the articles with spaCy (tokenization only needed)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat", "ner"])
    print("Running spaCy processing")
    id_to_tokens = {i: nlp(article) for i, article in tqdm(enumerate(non_empty_data))}
    print("Done processing")

    # Remove the stop words and lemmatize
    STOP_WORDS.update(
        ["think", "know", "people", "like", "thing", "good", "use", "come"]
    )
    id_to_tokens = {
        i: preprocess_spacy_doc(article, STOP_WORDS)
        for i, article in id_to_tokens.items()
    }

    unique_words = set().union(*id_to_tokens.values())
    vocabulary = list(unique_words)

    # Remove rare and overly common words from corpus
    filtered_tokens = filter_extremes(id_to_tokens.values(), vocabulary, more_than=10)
    id_to_filtered = {i: tokens for i, tokens in enumerate(filtered_tokens)}

    # Run LDA
    print("RUNNING LDA")
    start_time = perf_counter()
    lda = LatentDirichletAllocation(doc_to_tokens=id_to_filtered, K=k, alpha=alpha)
    lda.fit(n_iter=n_iter)
    end_time = perf_counter()
    print(f"Done in {(end_time - start_time):.2f}")
    print(lda.get_top_n_words(5))


if __name__ == "__main__":
    main()
