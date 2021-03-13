# Implement LDA with collapsed gibbs sampling on the 20NewsGroup dataset
from src.utility import *
from src.sampler import LatentDirichletAllocation, get_unique_words
from src.inference import *
from sklearn.datasets import fetch_20newsgroups
from time import perf_counter

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

if __name__ == '__main__':

    # With version of sklearn below .22
    dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

    # Process the articles with spaCy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat', 'ner'])
    print('Running spaCy processing')
    id_to_tokens = {i: nlp(article) for i, article in enumerate(dataset['data'])}
    print('Done processing')

    # Remove the stop words and lemmatize
    id_to_tokens = {i: preprocess_spacy_doc(article, STOP_WORDS) for i, article in id_to_tokens.items()}

    # Remove articles whose content is 'blah blah blah'
    # extra_words = ['maxaxaxaxaxaxaxaxaxaxaxaxaxaxax', 'said', 'also', 'would', 'get', 'say', 'go', 'do', 'one']
    # id_to_tokens = {title: remove_stop_words(tokens, extra_words=extra_words)
    #                 for title, tokens in id_to_tokens.items() if 'blah' not in tokens}

    vocabulary = get_unique_words(id_to_tokens.values())

    # Run LDA
    print('RUNNING LDA')
    start_time = perf_counter()
    topic, phi, theta = LatentDirichletAllocation(id_to_tokens, K=20, alpha=2/20, niter=10)
    end_time = perf_counter()
    print(f'Done in {(end_time - start_time):.2f}')
    print(get_top_n_words(phi, 5, vocabulary))
