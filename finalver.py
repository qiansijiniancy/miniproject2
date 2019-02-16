import os
import pandas as pd
import nltk
import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
# author: Sidi Yang

'''
Step 1: Load Data
'''
# current path
cwd = os.getcwd()

#read pos/neg files
positive_folder =[]
for x in os.listdir(cwd + '/data/train/pos/'):
    if x.endswith(".txt"):
        positive_folder.append(x)

negative_folder =[]
for x in os.listdir(cwd + '/data/train/neg/'):
    if x.endswith(".txt"):
        negative_folder.append(x)

test_folder =[]
for x in os.listdir(cwd + '/data/test/'):
    if x.endswith(".txt"):
        test_folder.append(x)

positiveReviews, negativeReviews, testReviews = [], [], []

for pfile in positive_folder:
    with open(cwd+"/data/train/pos/"+pfile, encoding="utf8") as f:
        positiveReviews.append(f.read())
for nfile in negative_folder :
    with open(cwd+"/data/train/neg/"+nfile, encoding="utf8") as f:
        negativeReviews.append(f.read())
for tfile in test_folder:
    with open(cwd+"/data/test/"+tfile, encoding="utf8") as f:
        testReviews.append(f.read())

# train_reviews
train_reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "target":1}),
    pd.DataFrame({"review":negativeReviews, "target":0}),
],
 ignore_index=True).sample(frac=1, random_state=1)


pos_reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "target":1}),
    #pd.DataFrame({"review":negativeReviews, "target":0}),
    #pd.DataFrame({"review":testReviews, "target":-1})
],
    ignore_index=True).sample(frac=1, random_state=1)

neg_reviews = pd.concat([
    #pd.DataFrame({"review":positiveReviews, "target":1}),
    pd.DataFrame({"review":negativeReviews, "target":0}),
    #pd.DataFrame({"review":testReviews, "target":-1})
],
    ignore_index=True).sample(frac=1, random_state=1)


test_reviews = pd.concat([
    #pd.DataFrame({"review":positiveReviews, "target":1}),
    #pd.DataFrame({"review":negativeReviews, "target":0}),
    pd.DataFrame({"review":testReviews, "target":0})
],
    ignore_index=True).sample(frac=1, random_state=1)


# prove it works
print(train_reviews.head())
print(train_reviews.shape)
print(test_reviews.shape)


'''
Step 2: Preprocess Data
'''
train_reviews['review'] = train_reviews['review'].str.lower()


# form corpus with train and test
corpus = pd.concat([train_reviews,test_reviews], axis=0)
print(corpus['review'].head(10))


'''
Step 3: Exact Features
'''

from sklearn.feature_extraction.text import CountVectorizer
# Transform each text into a vector of word counts
vectorizer = CountVectorizer(stop_words="english")

training_features = vectorizer.fit_transform(train_reviews["review"])
test_features = vectorizer.transform(test_reviews["review"])
print(training_features.shape)


'''
Step 4: Model Training
'''

# Training and Test data
# test_size = 0.5 cuz 25,000 + 25,000 =50,000
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(corpus['review'],
                                                    corpus['target'],
                                                    test_size=0.5)

print(len(X_train), len(y_train), len(X_test), len(y_test))

#
# ####################
# # Method 1: Multi Nominal NB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
vec = CountVectorizer()

X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

#alpha parameters stands for Add-One(Laplace) Smooting
model = MultinomialNB(alpha=1.0)

model.fit(X_train, y_train)

pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f%%' % (accuracy_score(y_test, pred) * 100))
print("The F1 accuracy score: {}%".format(f1_score(y_test, pred) * 100))

##########
#tf-idf
########
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(min_df=3).fit(X_train)
X_train_vectorized = vec.transform(X_train)
model = MultinomialNB(alpha=1.0)
model.fit(X_train_vectorized, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(vec.transform(X_test))
print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
#


#
# # ############################
# # Method 2: SVM
# from sklearn.svm import LinearSVC
# from sklearn.metrics import f1_score
# from sklearn.feature_extraction.text import CountVectorizer
# # vec = CountVectorizer()
# # X_train = vec.fit_transform(X_train)
# # X_test = vec.transform(X_test)
#
# model = LinearSVC(C=4,loss='squared_hinge')
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
#
# print("The F1 accuracy score: {}%".format(f1_score(y_test, pred) * 100))
#
# pd.DataFrame(pred).to_csv("SVM_prediction.csv")
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)