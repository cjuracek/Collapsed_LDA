import numpy as np


def get_top_n_words(phi_matrix, n, unique_words, return_probs=False):
    topic_top_words = {}

    for k in range(phi_matrix.shape[0]):
        # Find the top probability indices, then take the first n of them
        top_n_idx = np.argsort(phi_matrix[k, :])[::-1][:n]
        top_n_words = [unique_words[i] for i in top_n_idx]

        if return_probs:
            top_n_probs = phi_matrix[k, top_n_idx]
            top_n_probs = np.around(top_n_probs, 4)
            topic_top_words[k + 1] = [
                (word, prob) for word, prob in zip(top_n_words, top_n_probs)
            ]
        else:
            topic_top_words[k + 1] = top_n_words

    return topic_top_words
