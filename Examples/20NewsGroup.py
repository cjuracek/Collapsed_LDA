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
    id_to_tokens = {i: tokenize_doc(doc) for i, doc in enumerate(dataset['data'])}

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
    doc_list = {i: nlp(article) for i, article in enumerate(dataset['data'][:5])}
    doc_list = {i: preprocess_spacy_doc(article, STOP_WORDS) for i, article in doc_list.items()}

    # Remove articles whose content is 'blah blah blah'
    extra_words = ['maxaxaxaxaxaxaxaxaxaxaxaxaxaxax', 'said', 'also', 'would', 'get', 'say', 'go', 'do', 'one']
    id_to_tokens = {title: remove_stop_words(tokens, extra_words=extra_words)
                    for title, tokens in id_to_tokens.items() if 'blah' not in tokens}

    # Stem tokens
    titles_to_tokens_stem = {title: stem_tokens(tokens) for title, tokens in id_to_tokens.items()}

    vocabulary = get_unique_words(titles_to_tokens_stem.values())

    # Run LDA
    start_time = perf_counter()
    topic, phi, theta = LatentDirichletAllocation(titles_to_tokens_stem, K=20, alpha=2/20, niter=10)
    end_time = perf_counter()
    print(f'Done in {(end_time - start_time):.2f}')
    print(get_top_n_words(phi, 5, vocabulary))
