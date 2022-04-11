from cleaning import text_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


data = pd.read_csv("data/True.csv")
vectorizer = CountVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer.fit_transform(data['title'][:10])
print(vectorizer.get_feature_names_out())
print(X_transform.toarray())
