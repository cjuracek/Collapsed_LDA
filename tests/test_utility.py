from collapsed_lda.utility.utility import get_unique_words

# Test get_unique_words


def test_get_unique_words_across_documents():
    doc_1 = ["duplicate", "word_a"]
    doc_2 = ["duplicate", "word_b"]
    test_unique_words = get_unique_words([doc_1, doc_2])
    assert sorted(test_unique_words) == ["duplicate", "word_a", "word_b"]


def test_get_unique_words_within_documents():
    test_unique_words = get_unique_words([["duplicate", "duplicate"]])
    assert test_unique_words == ["duplicate"]
