import pandas as pd
from cleaning import text_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances

data = pd.read_csv('data/Text_mining.csv')
vectorizer = CountVectorizer(tokenizer=text_tokenizer)
X_transform = vectorizer.fit_transform(data['Review'])
X_transform_array = X_transform.toarray()
df = pd.DataFrame(X_transform_array,
                  columns=vectorizer.get_feature_names_out())

print(f"Cosine similarity:\n {cosine_similarity(df,df)}")
print(f"Euclidean distance:\n {euclidean_distances(df,df)}")
print(f"Cosine distance:\n {cosine_distances(df,df)}")
