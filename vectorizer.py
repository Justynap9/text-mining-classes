from cleaning import text_tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from matplotlib import pyplot as plot
from prettytable import PrettyTable


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
title1 = "Tokens appearing only in true titles"
only_in_true.plot.barh(title=title1)
plot.xlabel("Titles")
plot.ylabel("Appearance")
plot.show()
pretty_table = PrettyTable()
pretty_table.title = title1
pretty_table.add_column("Term", only_in_true.index)
pretty_table.add_column("Count", only_in_true.iloc[:, 0])
print(pretty_table)

# Figure 2
title2 = "Tokens appearing only in fake titles"
only_in_fake.plot.barh(title=title2)
plot.show()
pretty_table2 = PrettyTable()
pretty_table2.title = title2
pretty_table2.add_column("Term", only_in_fake.index)
pretty_table2.add_column("Count", only_in_fake.iloc[:, 0])
print(pretty_table2)

# Figure 3
title3 = "Most important tokens based on TF-IDF"
tokens_importance = tokens_importance_true.T.sort_values(by=0, ascending=False).head(15)
tokens_importance.plot.barh(title=title3)
plot.ylabel("Importance")
plot.show()
pretty_table3 = PrettyTable()
pretty_table3.title = title3
pretty_table3.add_column("Term", tokens_importance.index)
pretty_table3.add_column("Importance", tokens_importance.iloc[:, 0])
print(pretty_table3)

# Figure 4
title4 = "Crucial tokens based on binary weight"
tokens_bin = tokens_bin_true.T.sort_values(by=0, ascending=False).head(15)
tokens_bin.plot.barh(title=title4)
plot.show()
pretty_table4 = PrettyTable()
pretty_table4.title = title4
pretty_table4.add_column("Term", tokens_bin.index)
pretty_table4.add_column("Importance", tokens_bin.iloc[:, 0])
print(pretty_table4)