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

# TODO

# Test LatentDirichletAllocation._compute_MC_topic_approx()

# TODO

# Test LatentDirichletAllocation.get_top_n_words()

# TODO
