from cleaning import text_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


data_true = pd.read_csv("data/True.csv")
data_fake = pd.read_csv("data/Fake.csv")

data_true['target'] = 1
data_fake['target'] = 0
data_news = pd.concat([data_true, data_fake])

X_train, X_test, y_train, y_test = train_test_split(data_news['title'], data_news['target'],
                                                    test_size=0.3, shuffle=True)

count_vec = CountVectorizer(tokenizer=text_tokenizer)
X_train_vectors_count = count_vec.fit_transform(X_train)
X_test_vectors_count = count_vec.transform(X_test)

# Decision Tree Classifier
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train_vectors_count, y_train)
dtc_predict = dtc_model.predict(X_test_vectors_count)
print(f'Decision Tree Model \n {classification_report(y_test, dtc_predict)}')
print(f'Confusion Matrix \n {confusion_matrix(y_test, dtc_predict)} \n')
# Random Forest Classifier
rfc_model = RandomForestClassifier()
rfc_model.fit(X_train_vectors_count, y_train)
rfc_predict = rfc_model.predict(X_test_vectors_count)
print(f'Random Forest Model \n {classification_report(y_test, rfc_predict)}')
print(f'Confusion Matrix \n {confusion_matrix(y_test, rfc_predict)} \n')
# SVM
svm_model = LinearSVC()
svm_model.fit(X_train_vectors_count, y_train)
svm_predict = svm_model.predict(X_test_vectors_count)
print(f'SVM Model \n {classification_report(y_test, svm_predict)}')
print(f'Confusion Matrix \n {confusion_matrix(y_test, svm_predict)} \n')
# AdaBoost
ab_model = AdaBoostClassifier()
ab_model.fit(X_train_vectors_count, y_train)
ab_predict = ab_model.predict(X_test_vectors_count)
print(f'AdaBoost Model \n {classification_report(y_test, ab_predict)}')
print(f'Confusion Matrix \n {confusion_matrix(y_test, ab_predict)} \n')
# Bagging
b_model = BaggingClassifier()
b_model.fit(X_train_vectors_count, y_train)
b_predict = b_model.predict(X_test_vectors_count)
print(f'Bagging \n {classification_report(y_test, b_predict)}')
print(f'Confusion Matrix \n {confusion_matrix(y_test, b_predict)} \n')
