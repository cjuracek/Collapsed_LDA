import numpy as np
from random import choices
from scipy import stats


def LatentDirichletAllocation(iden_to_tokens, K, alpha, niter, beta=0.01):
    """ Perform collapsed Gibbs sampling to discover latent topics in corpus

    :param iden_to_tokens: A dictionary that maps unique identifiers (titles) to their contents
    :param K: Number of topics for LDA to discover
    :param alpha: Determines sparsity of topic distributions per document
    :param beta: Determines sparsity of word distributions per topic
    :param niter: Number of iterations to run the Gibbs sampler for
    :return: Topics per document (z), phi matrix, and theta matrix
    """

    document_word_topics_MC, document_topic_counts, word_topic_counts, total_topic_counts = initialize_topics(iden_to_tokens, K)
    unique_words = get_unique_words(iden_to_tokens.values())
    W = len(unique_words)

    for i in range(niter):  # One iteration of Gibbs sampler
        print(f'Running iteration {i} out of {niter}')
        for doc, words in iden_to_tokens.items():
            for i, word in enumerate(words):
                densities = np.zeros(K)
                curr_topic = document_word_topics_MC[doc][i][-1]
                for k in range(1, K + 1):
                    N_kj = document_topic_counts[doc][k]
                    N_wk = word_topic_counts[word][k]
                    N_k = total_topic_counts[k]

                    # New draw is conditioned on everything BUT this observation
                    if curr_topic == k:
                        N_kj -= 1
                        N_wk -= 1
                        N_k -= 1

                    # Eq. 1
                    a_kj = N_kj + alpha
                    b_wk = (N_wk + beta) / (N_k + W * beta)

                    densities[k - 1] = a_kj * b_wk

                # Draw a new topic
                densities /= np.sum(densities)  # Normalize
                new_topic = choices(range(1, K + 1), densities)[0]

                document_word_topics_MC[doc][i].append(new_topic)
                if new_topic == curr_topic:
                    continue

                # Update counts
                #document_word_topics[doc][i] = new_topic

                document_topic_counts[doc][curr_topic] -= 1
                document_topic_counts[doc][new_topic] += 1

                word_topic_counts[word][curr_topic] -= 1
                word_topic_counts[word][new_topic] += 1

                total_topic_counts[curr_topic] -= 1
                total_topic_counts[new_topic] += 1

    document_word_topics = compute_MC_topic_approx(document_word_topics_MC)
    phi_matrix = compute_phi_estimates(word_topic_counts, total_topic_counts, K, unique_words, beta)
    theta_matrix = compute_theta_estimates(document_topic_counts, K, alpha)

    return document_word_topics, phi_matrix, theta_matrix


def initialize_topics(iden_to_tokens, K):
    """
    Randomly assign a topic to each word in the corpus

    :param iden_to_tokens: A dictionary that maps unique identifiers (titles) to their contents
    :param K: Number of choices of topics
    :return: 4 dictionaries of counts (see comments below)
    """
    
    # Contains the ordered list of topics for each document (Dict of lists)
    document_word_topics_MC = {title: [] for title in iden_to_tokens.keys()}

    # Counts of each topic per document (Dict of dicts)
    document_topic_counts = {title: dict.fromkeys(range(1, K + 1), 0) for title in iden_to_tokens.keys()}

    unique_words = get_unique_words(iden_to_tokens.values())
    # Counts of each topic per word (dict of dicts)
    word_topic_counts = {word: dict.fromkeys(range(1, K + 1), 0) for word in unique_words}

    # Counts of each topic across all documents
    total_topic_counts = dict.fromkeys(range(1, K + 1), 0)

    for doc, words in iden_to_tokens.items():
        for i, word in enumerate(words):
            topic = np.random.randint(1, K + 1)
            document_word_topics_MC[doc].append([topic])
            document_topic_counts[doc][topic] = document_topic_counts[doc].get(topic, 0) + 1
            word_topic_counts[word][topic] = word_topic_counts[word].get(topic, 0) + 1
            total_topic_counts[topic] = total_topic_counts[topic] + 1

    return document_word_topics_MC, document_topic_counts, word_topic_counts, total_topic_counts


def compute_MC_topic_approx(document_word_topics_MC):
    """
    Given a Markov chain of word topics, compute a Monte Carlo approximation by picking mode of topics
    
    :param document_word_topics: Dictionary that maps identifiers (titles) to a Markov chain of their topics
    :return: Dictionary that maps identifiers (titles) to the Monte Carlo approx of their topics (mode)
    """
    
    document_word_topics = {title: [] for title in document_word_topics_MC.keys()}
    for doc, words in document_word_topics_MC.items():
        for i, word in enumerate(words):
            document_word_topics[doc].append(stats.mode(document_word_topics_MC[doc][i], axis=None)[0][0])

    return document_word_topics


def compute_phi_estimates(word_topic_counts, total_topic_counts, K, unique_words, beta):
    """
    Compute estimate of the phi matrix, containing word distributions per topic

    :param word_topic_counts: Dictionary that maps words to their respective counts per topic
    :param total_topic_counts: Dictionary that maps each topic to the number of times it appears in corpus
    :param K: Number of topics
    :param unique_words: The unique list of words contained in the corpus
    :param beta: Hyperparameter controlling sparsity of word distributions per topic
    :return: (K x W) matrix of word probability per topic
    """
    
    W = len(unique_words)
    phi_matrix = np.zeros((K, W))

    for w, word in enumerate(unique_words):
        for k in range(1, K + 1):
            N_wk = word_topic_counts[word][k]
            N_k = total_topic_counts[k]

            phi_matrix[k - 1, w] = (N_wk + beta) / (N_k + W * beta)

    return phi_matrix


def compute_theta_estimates(document_topic_counts, K, alpha):
    """
    Compute a matrix containing the mixture components of each document

    :param document_topic_counts: A dictionary mapping titles to topic counts in that document
    :param K: Number of topics
    :param alpha: Determines sparsity of topic distributions per document
    :return: A (K x D) NumPy array of mixture distributions per document
    """
    
    theta_matrix = np.zeros((K, len(document_topic_counts)))
    for j, (doc, topics) in enumerate(document_topic_counts.items()):
        for topic in topics:
            N_kj = document_topic_counts[doc][topic]
            N_j = sum(document_topic_counts[doc].values())
            theta_matrix[topic - 1, j] = (N_kj + alpha) / (N_j + K * alpha)

    return theta_matrix

def get_unique_words(tokens):
    """
    Provide a list of unique tokens present in the list tokens

    :param tokens: List of lists containing all of the tokens in the corpus
    :return: A list of all the unique tokens in the corpus
    """
    
    unique_words = set().union(*tokens)
    return list(unique_words)
