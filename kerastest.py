# ref: https://github.com/parthkhanna150/IMDB_Sentiment/blob/master/Project2_551.ipynb
from bs4 import BeautifulSoup
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_links_characters(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub('\[[^]]*\]', '', text)
    return text

def regular_preprocess(text):
    text = remove_html(text)
    text = remove_links_characters(text)
    return text

def advanced_preprocess(text):
    return text

def preprocess(data, advanced=False):
    refined_data=[]

    if advanced:
        for data_point in data:
            refined_data.append(advanced_preprocess(data_point))
    else:
        for data_point in data:
            refined_data.append(regular_preprocess(data_point))

    return refined_data


import glob
import os
from preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

train_data = []
test_data = []

# train data
train_neg = glob.glob(os.path.join(os.getcwd(), "/data/train/pos/", "*.txt"))
for f_path in train_neg:
    with open(f_path) as f:
        train_data.append(f.read())

train_pos = glob.glob(os.path.join(os.getcwd(), "/data/train/neg/", "*.txt"))
for f_path in train_pos:
    with open(f_path) as f:
        train_data.append(f.read())

# test data
test_files = glob.glob(os.path.join(os.getcwd(), "/data/test/", "*.txt"))
for f_path in test_files:
    with open(f_path) as f:
        test_data.append(f.read())

# targets: first 12500 are pos, next 12500 are neg
targets = [0 if i<12500 else 1 for i in range(25000)]

train_data_clean = preprocess(train_data, advanced=False)
test_data_clean = preprocess(test_data, advanced=False)

print(len(train_data))

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(train_data,
                                                    targets,
                                                    train_size = 0.5,
                                                    test_size = 0.5,
                                                    random_state=551)
# splitting the data
#X_train, X_validation, y_train, y_validation = train_test_split(train_data, targets,
# train_size=0.8, test_size=0.2, random_state=551)

# Bag of Words vectorization
cv = CountVectorizer().fit(X_train)
X_train_counts = cv.transform(X_train)
X_validation_counts = cv.transform(X_validation)

# tfidf
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_validation_tfidf = tfidf_transformer.transform(X_validation_counts)

# normalization
normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_validation_normalized = normalizer_tranformer.transform(X_validation_tfidf)

def display_results(y_val, y_pred, heading):
    print(metrics.classification_report(y_val, y_pred))
    print("Accuracy % = ", metrics.accuracy_score(y_val, y_pred))

# clf_NB = MultinomialNB().fit(X_train_normalized, y_train)
# y_pred = clf_NB.predict(X_validation_normalized)
# display_results(y_validation, y_pred,"")

clf_LR = LogisticRegression().fit(X_train_normalized, y_train)
y_pred = clf_LR.predict(X_validation_normalized)
display_results(y_validation, y_pred,"")

# clf_DT = DecisionTreeClassifier().fit(X_train_normalized, y_train)
# y_pred = clf_DT.predict(X_validation_normalized)
# display_results(y_validation, y_pred,"")

clf_SVM = LinearSVC().fit(X_train_normalized, y_train)
y_pred = clf_SVM.predict(X_validation_normalized)
display_results(y_validation, y_pred,"")