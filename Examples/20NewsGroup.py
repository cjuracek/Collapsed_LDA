# Implement LDA with collapsed gibbs sampling on the 20NewsGroup dataset
from src.utility import *
from src.sampler import LatentDirichletAllocation
from src.utility import get_unique_words
from src.inference import *
from sklearn.datasets import fetch_20newsgroups
from time import perf_counter
from tqdm import tqdm

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

if __name__ == '__main__':

    # With version of sklearn below .22
    dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    # Remove space-only documents
    non_empty_data = [article for article in dataset['data'][:100] if article and not article.isspace()]

    # Process the articles with spaCy (tokenization only needed)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat', 'ner'])
    print('Running spaCy processing')
    id_to_tokens = {i: nlp(article) for i, article in tqdm(enumerate(non_empty_data))}
    print('Done processing')

    # Remove the stop words and lemmatize
    STOP_WORDS.update(['think', 'know', 'people', 'like', 'thing', 'good', 'use', 'come'])
    id_to_tokens = {i: preprocess_spacy_doc(article, STOP_WORDS) for i, article in id_to_tokens.items()}

    # TODO
    #   Filter id_to_tokens
    #   Save processed spaCy docs
    #   Save results of LDA to file
    #   Consider making an LDA class

    unique_words = set().union(*id_to_tokens.values())
    vocabulary = list(unique_words)

    # Remove rare and overly common words from corpus
    filtered_tokens = filter_extremes(id_to_tokens.values(), vocabulary, more_than=10)
    id_to_filtered = {i: tokens for i, tokens in enumerate(filtered_tokens)}
    unique_words = set().union(*id_to_filtered.values())
    vocabulary = list(unique_words)

    # Run LDA
    print('RUNNING LDA')
    start_time = perf_counter()
    topic, phi, theta = LatentDirichletAllocation(id_to_filtered, K=20, alpha=2 / 20, niter=10)
    end_time = perf_counter()
    print(f'Done in {(end_time - start_time):.2f}')
    print(get_top_n_words(phi, 5, vocabulary))
