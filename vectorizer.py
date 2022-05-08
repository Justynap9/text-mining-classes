from cleaning import text_tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd


data = pd.read_csv("data/True.csv")
count_vec = CountVectorizer(tokenizer=text_tokenizer)
X_transform = count_vec.fit_transform(data['title'])

tfidf_vec = TfidfVectorizer(tokenizer=text_tokenizer)
X_transform_tfidf = tfidf_vec.fit_transform(data['title'])

# Top 10 most frequent tokens

tokens_count = pd.DataFrame(X_transform.sum(axis=0),
                            columns=count_vec.get_feature_names_out())
print(tokens_count.T.sort_values(by=0, ascending=False).head(10))

# Top 10 most important tokens

tokens_tfidf = pd.DataFrame(X_transform_tfidf.sum(axis=0),
                            columns=tfidf_vec.get_feature_names_out())
print(tokens_tfidf.T.sort_values(by=0, ascending=False).head(10))

# Top 10 documents with the most tokens

tokens_count_doc = pd.DataFrame(X_transform.sum(axis=1))
print(tokens_count_doc.sort_values(by=0, ascending=False).head(10))
