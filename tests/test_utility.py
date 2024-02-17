import string

from nltk.corpus import stopwords

from collapsed_lda.utility.utility import (
    filter_extremes,
    get_unique_words,
    preprocess_spacy_doc,
    remove_stop_words,
    tokenize_doc,
)

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


# Test remove_stop_words()


def test_remove_stop_words_removes_extra_words():
    tokens = ["test", "extra", "word"]
    extra_words = ["extra", "word"]
    filtered_tokens = remove_stop_words(tokens=tokens, extra_words=extra_words)
    assert not any([word in filtered_tokens for word in extra_words])


def test_remove_stop_words_removes_numbers():
    tokens = ["test", "100", "2014"]
    filtered_tokens = remove_stop_words(tokens=tokens, remove_numbers=True)
    assert filtered_tokens == ["test"]


def test_stop_words_removes_top_fifty_words():
    tokens = [token for token in stopwords.words("English")]
    tokens.append("test")
    filtered_tokens = remove_stop_words(tokens, tokens_have_quotes=True)
    assert filtered_tokens == ["test"]


# Test preprocess_spacy_doc()


def test_preprocess_spacy_doc_removes_stop_words(nlp):
    test_doc = nlp("Stop that!")
    test_tokens = preprocess_spacy_doc(test_doc, stop_words=["stop"])
    assert test_tokens == ["that"]


def test_preprocess_spacy_doc_removes_short_lemmas(nlp):
    test_doc = nlp("I was reading the paper.")
    test_tokens = preprocess_spacy_doc(test_doc)
    assert all([len(token) > 2 for token in test_tokens])
