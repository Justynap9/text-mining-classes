from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def stemming(text: str) -> list[str]:
    word_tokens = word_tokenize(text)
    porter = PorterStemmer()
    stem_words = []
    for w in word_tokens:
        stem_words.append(porter.stem(w))
    return stem_words
