from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from tqdm import tqdm
import string


def parse_sgm_file(sgm_data):
    """
    Returns a dictionary with titles + articles of an SGM file

    :param sgm_data: Data read from an SGM file
    :return: A dictionary mapping titles to article contents
    """

    soup = BeautifulSoup(sgm_data, features="html5lib")
    texts = soup.find_all('text')
    title_docs = {}
    for text in texts:
        title = text.findChild('title')

        # Title is non-existent for a few articles
        if title:
            # Use contents because the text has no name; always the last element
            title_docs[title.string.strip()] = text.contents[-1]

    return title_docs


def tokenize_doc(doc):
    """
    Convert document to lowercase, remove punctuation, and split on whitespace

    :param doc: Document of text (list)
    :return: Tokens of document
    """
    
    doc = doc.lower()
    whitespace = string.whitespace + '\x03'  # End of file char
    trans = str.maketrans(whitespace, ' ' * len(whitespace), string.punctuation)
    doc_no_punc = doc.translate(trans)
    return doc_no_punc.split()


def remove_stop_words(tokens, remove_numbers=True, tokens_have_quotes=False, extra_words=None):
    """
    Remove top 50 most common stop words, along with numbers, and additional extra words

    :param tokens: List of tokens
    :param remove_numbers: Should number strings be removed from the tokens?
    :param tokens_have_quotes: Do the tokens contain quotes?
    :param extra_words: Additional words that will be removed from tokens
    :return: List of tokens with appropriate words removed
    """
    
    if extra_words is None:
        extra_words = []

    stop_words = stopwords.words('English')
    stop_words += extra_words

    if not tokens_have_quotes:
        stop_words = set([word.replace('\'', '') for word in stop_words])

    tokens_no_stop = [token for token in tokens if token not in stop_words]
    if remove_numbers:
        tokens_no_stop = [token for token in tokens_no_stop if not token.isnumeric()]
    return tokens_no_stop


def stem_tokens(tokens):
    """
    Stem tokens using the Porter stemmer

    :param tokens: List of tokens
    :return: A stemmed list of the same tokens
    """
    
    ps = PorterStemmer()
    tokens_stemmed = [ps.stem(token) for token in tokens]
    return tokens_stemmed


# Remove stop words and lemmatize
def preprocess_spacy_doc(doc, stop_words):
    # Get all lowercased lemmas from document
    lemmas = [token.lemma_.lower() for token in doc if token.text.isalpha()]

    # Remove all stop words / lemmas that are too short
    lemmas = [lemma for lemma in lemmas if lemma not in stop_words and len(lemma) > 2]
    return lemmas


# Filter out rare tokens. Per Porteous the vocabulary was filtered
# "only keeping words that occurred more than ten times"
def filter_extremes(docs, vocabulary, more_than=10):

    # Take words that appear more than "more than" times
    good_words = [word for word in tqdm(vocabulary)
                  if more_than < sum([word in doc for doc in docs])]

    tokens = [[word for word in doc if word in good_words] for doc in docs]
    return tokens


def get_unique_words(tokens):
    """
    Provide a list of unique tokens present in the list tokens

    :param tokens: List of lists containing all of the tokens in the corpus
    :return: A list of all the unique tokens in the corpus
    """

    unique_words = set().union(*tokens)
    return list(unique_words)