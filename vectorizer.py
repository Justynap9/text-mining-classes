from cleaning import text_tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from matplotlib import pyplot as plot


data_true = pd.read_csv("data/True.csv")
data_fake = pd.read_csv("data/Fake.csv")
count_vec = CountVectorizer(tokenizer=text_tokenizer)
count_vec1 = CountVectorizer(tokenizer=text_tokenizer)
X_transform = count_vec.fit_transform(data_true['title'])
X_transform_fake = count_vec1.fit_transform(data_fake['title'])
count_vec_bin = CountVectorizer(tokenizer=text_tokenizer, binary=True)
X_transform_bin = count_vec_bin.fit_transform(data_true['title'])

tfidf_vec = TfidfVectorizer(tokenizer=text_tokenizer)
X_transform_tfidf = tfidf_vec.fit_transform(data_true['title'])

tokens_count_true = pd.DataFrame(X_transform.sum(axis=0), columns=count_vec.get_feature_names_out())
tokens_count_false = pd.DataFrame(X_transform_fake.sum(axis=0), columns=count_vec1.get_feature_names_out())
tokens_importance_true = pd.DataFrame(X_transform_tfidf.sum(axis=0),
                                      columns=tfidf_vec.get_feature_names_out())
tokens_bin_true = pd.DataFrame(X_transform_bin.sum(axis=0),
                               columns=count_vec_bin.get_feature_names_out())


def compare_titles(count_dict: dict, titles: list) -> dict:
    unique_title = {}
    for values in count_dict.values():
        for key, value in values.items():
            if key not in titles:
                unique_title[key] = value
    return unique_title


only_in_true = pd.DataFrame.from_dict(compare_titles(tokens_count_true.to_dict(orient="index"),
                                                     count_vec1.get_feature_names_out()),
                                      orient="index").sort_values(by=0, ascending=False).head(15)
only_in_fake = pd.DataFrame.from_dict(compare_titles(tokens_count_false.to_dict(orient="index"),
                                                     count_vec.get_feature_names_out()),
                                      orient="index").sort_values(by=0, ascending=False).head(15)

# Figure 1
only_in_true.plot.barh(title="Tokens appearing only in true titles")
plot.xlabel("Titles")
plot.ylabel("Appearance")
plot.show()

# Figure 2
only_in_fake.plot.barh(title="Tokens appearing only in fake titles")
plot.show()

# Figure 3
tokens_importance_true.T.sort_values(by=0, ascending=False).head(15).plot.barh(
    title="Most important tokens based on TF-IDF")
plot.ylabel("Importance")
plot.show()

# Figure 4
tokens_bin_true.T.sort_values(by=0, ascending=False).head(15).plot.barh(
    title="Crucial tokens based on binary weight")
plot.show()
