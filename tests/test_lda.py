import numpy as np

from collapsed_lda.lda import LatentDirichletAllocation

# Test LatentDirichletAllocation._compute_phi_estimates()


def test_compute_phi_estimates_expected_shape(lda):
    # Act
    word_topic_counts = {
        "alpha": {0: 2, 1: 0},
        "bravo": {0: 0, 1: 1},
        "charlie": {0: 1, 1: 0},
        "delta": {0: 0, 1: 1},
        "echo": {0: 1, 1: 0},
    }
    total_topic_counts = {0: 4, 1: 2}
    lda._compute_phi_estimates(
        word_topic_counts=word_topic_counts, total_topic_counts=total_topic_counts
    )
    test_phi = lda.phi_matrix

    # Assert
    assert test_phi.shape == (lda.K, 5)


def test_compute_phi_estimates_expected_values(lda):
    # Act
    word_topic_counts = {
        "alpha": {0: 2, 1: 0},
        "bravo": {0: 0, 1: 1},
        "charlie": {0: 1, 1: 0},
        "delta": {0: 0, 1: 1},
        "echo": {0: 1, 1: 0},
    }
    total_topic_counts = {0: 4, 1: 2}
    lda._compute_phi_estimates(
        word_topic_counts=word_topic_counts, total_topic_counts=total_topic_counts
    )
    test_phi = lda.phi_matrix
    # Hand-computed following from formula
    W_Beta = lda.W * lda.beta
    expected_phi = np.array(
        [
            [
                (2 + lda.beta) / (4 + W_Beta),
                lda.beta / (4 + W_Beta),
                (1 + lda.beta) / (4 + W_Beta),
                lda.beta / (4 + W_Beta),
                (1 + lda.beta) / (4 + W_Beta),
            ],
            [
                lda.beta / (2 + W_Beta),
                (1 + lda.beta) / (2 + W_Beta),
                lda.beta / (2 + W_Beta),
                (1 + lda.beta) / (2 + W_Beta),
                lda.beta / (2 + W_Beta),
            ],
        ]
    )
    assert np.allclose(test_phi, expected_phi)


# Test LatentDirichletAllocation._compute_theta_estimates()


def test_compute_theta_estimates_has_correct_shape(lda):
    doc_topic_counts = {"doc_1": {}, "doc_2": {}}
    lda._compute_theta_estimates(document_topic_counts=doc_topic_counts)

    test_theta = lda.theta_matrix
    expected_shape = (lda.K, len(doc_topic_counts))
    assert test_theta.shape == expected_shape


def test_compute_theta_estimates_has_correct_values(lda):
    doc_topic_counts = {"doc_1": {0: 5, 1: 2}, "doc_2": {0: 1, 1: 8}}
    lda._compute_theta_estimates(document_topic_counts=doc_topic_counts)

    K_alpha = lda.K * lda.alpha
    expected_theta = np.array(
        [
            [(5 + lda.alpha) / (7 + K_alpha), (1 + lda.alpha) / (9 + K_alpha)],
            [(2 + lda.alpha) / (7 + K_alpha), (8 + lda.alpha) / (9 + K_alpha)],
        ]
    )
    test_theta = lda.theta_matrix
    assert np.allclose(test_theta, expected_theta)


# Test LatentDirichletAllocation._compute_MC_topic_approx()


def test_compute_mc_topic_approx_gives_correct_values(lda):
    # 3 words in Doc 0, 2 words in Doc 1. Run for 3 iterations
    word_topics_MC = {
        "doc_0": [[1, 0, 1], [1, 1, 0], [1, 0, 0]],
        "doc_1": [[0, 0, 0], [0, 0, 1]],
    }
    expected_topics = {"doc_0": [1, 1, 0], "doc_1": [0, 0]}
    lda._compute_MC_topic_approx(document_word_topics_MC=word_topics_MC)
    test_topics = lda.document_word_topics
    assert test_topics == expected_topics


# Test LatentDirichletAllocation.get_top_n_words()


def test_get_top_0_words_returns_empty(lda):
    test_top_words = lda.get_top_n_words(n=0, return_probs=False)
    expected_top_words = {k: [] for k in range(lda.K)}
    assert test_top_words == expected_top_words


def test_get_top_n_words_gives_top_singular_word(lda):
    lda.phi_matrix = np.array(
        [[0.8, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.8]]
    )
    test_top_words = lda.get_top_n_words(n=1, return_probs=False)
    assert test_top_words == {0: ["alpha"], 1: ["echo"]}


def test_get_top_n_words_gives_expected_prob(lda):
    lda.phi_matrix = np.array(
        [[0.8, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.8]]
    )
    test_top_words = lda.get_top_n_words(n=1, return_probs=True)
    test_probs = {k: word_info[0][1] for k, word_info in test_top_words.items()}
    assert test_probs == {0: 0.8, 1: 0.8}
