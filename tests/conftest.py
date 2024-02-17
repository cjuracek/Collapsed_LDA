import pytest
import spacy

from collapsed_lda.lda import LatentDirichletAllocation


@pytest.fixture()
def lda():
    """LDA test fixture with 2 docs and 2 topics"""
    doc_to_tokens = {
        "doc_1": ["alpha", "bravo", "charlie"],
        "doc_2": ["alpha", "delta", "echo"],
    }
    lda = LatentDirichletAllocation(doc_to_tokens=doc_to_tokens, K=2, alpha=1, beta=0.01)
    return lda


@pytest.fixture()
def nlp():
    return spacy.load("en_core_web_sm")
