from collections import Counter
from random import choices
from statistics import mode

import numpy as np
from tqdm import trange

from collapsed_lda.utility import get_unique_words


class LatentDirichletAllocation:
    def __init__(self, doc_to_tokens, K, alpha, beta=0.01, verbose=True):
        self.iden_to_tokens = doc_to_tokens
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.vocabulary = get_unique_words(doc_to_tokens.values())
        self.W = len(self.vocabulary)
        self.theta_matrix = np.zeros((K, len(doc_to_tokens)))
        self.phi_matrix = np.zeros((K, self.W))
        self.verbose = verbose

    def fit(self, n_iter):
        """Perform collapsed Gibbs sampling to discover latent topics in corpus

        :param n_iter: Number of iterations to run the Gibbs sampler for
        """

        (
            document_word_topics_MC,
            document_topic_counts,
            word_topic_counts,
            total_topic_counts,
        ) = self._initialize_topics()

        if self.verbose:
            print(f"Running LDA for {n_iter} iterations...")

        for j in trange(n_iter):  # One iteration of Gibbs sampler
            print(f"Running iteration {j + 1} out of {n_iter}")
            for doc, words in self.iden_to_tokens.items():
                for word_idx, word in enumerate(words):
                    densities = np.zeros(self.K)
                    curr_topic = document_word_topics_MC[doc][word_idx][
                        -1
                    ]  # Get most recent topic of MC chain

                    # Calculate probability that a given latent topic z_ij belongs to topic k for each k
                    for k in range(self.K):
                        # Relevant counts needed for computation - see paragraph before Eq. 1
                        N_kj = document_topic_counts[doc][k]
                        N_wk = word_topic_counts[word][k]
                        N_k = total_topic_counts[k]

                        # New draw is conditioned on everything BUT this observation
                        if curr_topic == k:
                            N_kj -= 1
                            N_wk -= 1
                            N_k -= 1

                        # Eq. 1
                        a_kj = N_kj + self.alpha
                        b_wk = (N_wk + self.beta) / (N_k + self.W * self.beta)
                        densities[k] = a_kj * b_wk

                    # Draw a new topic and append to MC - normalization not needed
                    new_topic = choices(range(self.K), weights=densities)[0]
                    document_word_topics_MC[doc][word_idx].append(new_topic)

                    # No need to update counts if topic is the same
                    if new_topic == curr_topic:
                        continue

                    # Update counts
                    document_topic_counts[doc][curr_topic] -= 1
                    document_topic_counts[doc][new_topic] += 1

                    word_topic_counts[word][curr_topic] -= 1
                    word_topic_counts[word][new_topic] += 1

                    total_topic_counts[curr_topic] -= 1
                    total_topic_counts[new_topic] += 1

        # Determine topic for word from the chain
        self._compute_MC_topic_approx(document_word_topics_MC)

        # Estimate other model parameters we are interested in
        self._compute_phi_estimates(word_topic_counts, total_topic_counts)
        self._compute_theta_estimates(document_topic_counts)

    def _compute_phi_estimates(self, word_topic_counts, total_topic_counts):
        """
        Compute estimate of the phi matrix, containing word distributions per topic

        :param word_topic_counts: Dictionary that maps words to their respective counts per topic
        :param total_topic_counts: Dictionary that maps each topic to the number of times it appears in corpus
        """

        for w, word in enumerate(self.vocabulary):
            for k in range(self.K):
                N_wk = word_topic_counts[word][k]
                N_k = total_topic_counts[k]

                self.phi_matrix[k, w] = (N_wk + self.beta) / (N_k + self.W * self.beta)

    def _compute_theta_estimates(self, document_topic_counts):
        """
        Compute a matrix containing the mixture components of each document

        :param document_topic_counts: A dictionary mapping titles to topic counts in that document
        """
        for j, (doc, topics) in enumerate(document_topic_counts.items()):
            for topic in topics:
                N_kj = document_topic_counts[doc][topic]
                N_j = sum(document_topic_counts[doc].values())
                self.theta_matrix[topic, j] = (N_kj + self.alpha) / (
                    N_j + self.K * self.alpha
                )

    def _initialize_topics(self):
        """
        Randomly initialize topic / word count information needed for sampling

        :return: 4 dictionaries of counts (see comments below)
        """
        if self.verbose:
            print("Initializing topics...")

        # Contains the ordered list of topics for each document (Dict of lists)
        document_word_topics_MC = {}

        # Counts of each topic per document (Dict of dicts)
        document_topic_counts = {
            title: Counter() for title in self.iden_to_tokens.keys()
        }

        # Counts number of times a given word is assigned to each topic (dict of dicts)
        word_topic_counts = {word: Counter() for word in self.vocabulary}

        # Counts of each topic across all documents
        total_topic_counts = Counter()

        for doc, words in self.iden_to_tokens.items():
            # Start with randomly assigned topics - update appropriate counts
            topics = np.random.randint(low=0, high=self.K, size=len(words))
            document_word_topics_MC[doc] = [[topic] for topic in topics]
            document_topic_counts[doc].update(topics)
            total_topic_counts.update(topics)

            # Update the topic counts per word
            for unique_word in set(words):
                unique_word_topics = [
                    topic
                    for idx, topic in enumerate(topics)
                    if words[idx] == unique_word
                ]
                word_topic_counts[unique_word].update(unique_word_topics)

        return (
            document_word_topics_MC,
            document_topic_counts,
            word_topic_counts,
            total_topic_counts,
        )

    def _compute_MC_topic_approx(self, document_word_topics_MC):
        """
        Given a Markov chain of word topics, compute a Monte Carlo approximation by picking mode of topics

        :param document_word_topics_MC: Dictionary that maps identifiers (titles) to a Markov chain of their topics
        :return: Dictionary that maps identifiers (titles) to the Monte Carlo approx of their topics (mode)
        """

        document_word_topics = {title: [] for title in document_word_topics_MC.keys()}

        # Iterate through all chains in all documents
        for doc, word_chains in document_word_topics_MC.items():
            # Iterate through chains within individual documents
            for word_chain in word_chains:
                most_frequent_topic = mode(word_chain)
                document_word_topics[doc].append(most_frequent_topic)

        self.document_word_topics = document_word_topics

    def get_top_n_words(self, n, return_probs=False):
        """
        Calculate the top n words with highest posterior probability for every topic

        :param n: Top number of words to find
        :param return_probs: Should we return probabilities with these words?
        :return: A dictionary mapping topics to the respective top words
        """
        topic_top_words = {}

        for k in range(self.phi_matrix.shape[0]):
            # Find the top probability indices, then take the first n of them
            top_n_idx = np.argsort(self.phi_matrix[k, :])[::-1][:n]
            top_n_words = [self.vocabulary[i] for i in top_n_idx]

            if return_probs:
                top_n_probs = self.phi_matrix[k, top_n_idx]
                top_n_probs = np.around(top_n_probs, 4)
                topic_top_words[k] = [
                    (word, prob) for word, prob in zip(top_n_words, top_n_probs)
                ]
            else:
                topic_top_words[k] = top_n_words

        return topic_top_words
