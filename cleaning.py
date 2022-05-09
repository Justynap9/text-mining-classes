import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def cleaning(text: str) -> str:
    patterns = [r'[:;]\W+',
                '[0-9]+',
                '<.*?>',
                r'[^\w\s]']
    text = re.sub('|'.join(patterns), '', text)
    text = " ".join(text.split()).lower()
    return text


def stop_words(stem_words: list) -> list[str]:
    stopwords_en = stopwords.words('english')
    new_list = []
    for word in stem_words:
        if word not in stopwords_en:
            new_list.append(word)
    return new_list


def stemming(text: str) -> list[str]:
    word_tokens = word_tokenize(text)
    porter = PorterStemmer()
    stem_words = []
    for w in word_tokens:
        stem_words.append(porter.stem(w))
    return stem_words


def text_tokenizer(text: str) -> list:
    text = cleaning(text)
    stem_words = stemming(text)
    wordlist = stop_words(stem_words)
    new_wordlist = [w for w in wordlist if len(w) > 3]
    return new_wordlist
