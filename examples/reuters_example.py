from time import time

from src.inference import *
from src.sampler import LatentDirichletAllocation
from src.utility import *
from src.utility import get_unique_words

if __name__ == "__main__":
    with open("../Data/reuters21578/reut2-000.sgm") as f:
        data = f.read()

    titles_to_articles = parse_sgm_file(data)
    titles_to_tokens = {
        title: tokenize_doc(doc) for title, doc in titles_to_articles.items()
    }

    # Remove articles whose content is 'blah blah blah'
    extra_words = ["reuter", "said", "also", "would"]
    titles_to_tokens = {
        title: remove_stop_words(tokens, extra_words=extra_words)
        for title, tokens in titles_to_tokens.items()
        if "blah" not in tokens
    }
    titles_to_tokens_stem = {
        title: stem_tokens(tokens) for title, tokens in titles_to_tokens.items()
    }

    unique_words = get_unique_words(titles_to_tokens_stem.values())
    t0 = time()
    topic, phi, theta = LatentDirichletAllocation(
        titles_to_tokens_stem, K=5, alpha=2 / 5, beta=0.01, niter=10
    )
    print("done in %0.3fs." % (time() - t0))
    print(get_top_n_words(phi, 5, unique_words))
