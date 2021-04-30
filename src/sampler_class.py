from collections import Counter

from src.utility import get_unique_words

import numpy as np

class LatentDirichletAllocation:

    def __init__(self, K, iden_to_tokens):
        self.K = K
        self.iden_to_tokens = iden_to_tokens
        self.vocabulary = get_unique_words(iden_to_tokens.values())
        self.theta_matrix = np.zeros((K, len(iden_to_tokens)))
        self.phi_matrix = np.zeros((K, len(self.vocabulary)))

    def fit(self):
        document_word_topics_MC, document_topic_counts, word_topic_counts, total_topic_counts = self._initialize_topics()

    def _initialize_topics(self):
        """
        Randomly initialize topic / word count information needed for sampling

        :return: 4 dictionaries of counts (see comments below)
        """

        # Contains the ordered list of topics for each document (Dict of lists)
        document_word_topics_MC = {}

        # Counts of each topic per document (Dict of dicts)
        document_topic_counts = {title: Counter() for title in self.iden_to_tokens.keys()}

        # Counts number of times a given word is assigned to each topic (dict of dicts)
        word_topic_counts = {word: Counter() for word in self.vocabulary}

        # Counts of each topic across all documents
        total_topic_counts = Counter()

        for doc, words in self.iden_to_tokens.items():

            # Start with randomly assigned topics - update appropriate counts
            topics = np.random.randint(low=1, high=self.K + 1, size=len(words))
            document_word_topics_MC[doc] = [[topic] for topic in topics]
            document_topic_counts[doc].update(topics)
            total_topic_counts.update(topics)

            # Update the topic counts per word
            for unique_word in set(words):
                unique_word_topics = [topic for idx, topic in enumerate(topics) if words[idx] == unique_word]
                word_topic_counts[unique_word].update(unique_word_topics)

        return document_word_topics_MC, document_topic_counts, word_topic_counts, total_topic_counts

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