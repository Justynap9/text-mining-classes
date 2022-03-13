import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def cleaning(text: str) -> str:
    emoticons = re.findall(r'[:;]\W+', text)
    patterns = [r'[:;]\W+',
                '[0-9]+',
                '<.*?>',
                r'[^\w\s]']
    text = re.sub('|'.join(patterns), '', text)
    text += " ".join(emoticons)
    text = " ".join(text.split()).lower()
    return text


def stop_words(text: str) -> str:
    stopwords_en = stopwords.words('english')
    word_tokens = word_tokenize(text)
    new_text = []
    for w in word_tokens:
        if w not in stopwords_en:
            new_text.append(w)
    return " ".join(new_text)
