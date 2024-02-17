import string

from collapsed_lda.utility.utility import filter_extremes, get_unique_words, tokenize_doc

# Test filter_extremes()


def test_filter_extremes_within_document():
    rare_tokens = ["rare"] * 10
    common_tokens = ["common"] * 11
    doc = [*rare_tokens, *common_tokens]
    test_tokens = filter_extremes([doc], vocabulary=["rare", "common"])[0]
    assert test_tokens == common_tokens


def test_filter_extremes_across_documents():
    docs = [["rare", "common"] for i in range(10)]
    docs.append(["common"])
    filtered_docs = filter_extremes(docs, vocabulary=["rare", "common"])
    assert ["common"] in filtered_docs
    assert ["rare"] not in filtered_docs


# Test get_unique_words()


def test_get_unique_words_across_documents():
    doc_1 = ["duplicate", "word_a"]
    doc_2 = ["duplicate", "word_b"]
    test_unique_words = get_unique_words([doc_1, doc_2])
    assert sorted(test_unique_words) == ["duplicate", "word_a", "word_b"]


def test_get_unique_words_within_documents():
    test_unique_words = get_unique_words([["duplicate", "duplicate"]])
    assert test_unique_words == ["duplicate"]


# Test tokenize_doc()


def test_tokenize_doc_normal_use_case():
    doc = "This! Is a test document."
    processed_doc = tokenize_doc(doc)
    assert processed_doc == ["this", "is", "a", "test", "document"]


def test_tokenize_doc_lowercases():
    doc = "THIS IS UPPERCASE"
    processed_doc = tokenize_doc(doc)
    assert all([token.islower() for token in processed_doc])


def test_tokenize_doc_removes_punctuation():
    test_doc = f"token1 {string.punctuation} token2"
    processed_doc = tokenize_doc(test_doc)
    assert not any([punctuation in processed_doc for punctuation in string.punctuation])


def test_tokenize_doc_irregular_whitespace_removed():
    test_doc = f"token1 {string.whitespace} \x03 token2"
    processed_doc = tokenize_doc(test_doc)
    assert not any([whitespace in processed_doc for whitespace in string.whitespace + "\x03"])
