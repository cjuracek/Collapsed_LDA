from src.utility import get_unique_words

import numpy as np

class LatentDirichletAllocation:

    def __init__(self, K, iden_to_tokens):
        self.vocabulary = get_unique_words(iden_to_tokens.values())
        self.theta_matrix = np.zeros((K, len(iden_to_tokens)))
        self.phi_matrix = np.zeros((K, len(self.vocabulary)))


    def get_top_n_words(self, n, return_probs=False):
        topic_top_words = {}

        for k in range(self.phi_matrix.shape[0]):
            # Find the top probability indices, then take the first n of them
            top_n_idx = np.argsort(self.phi_matrix[k, :])[::-1][:n]
            top_n_words = [self.vocabulary[i] for i in top_n_idx]

            if return_probs:
                top_n_probs = self.phi_matrix[k, top_n_idx]
                top_n_probs = np.around(top_n_probs, 4)
                topic_top_words[k + 1] = [(word, prob) for word, prob in zip(top_n_words, top_n_probs)]
            else:
                topic_top_words[k + 1] = top_n_words

        return topic_top_words