install:
	poetry run python -m spacy download en_core_web_sm
	poetry run python -c 'import nltk; nltk.download("stopwords")'

