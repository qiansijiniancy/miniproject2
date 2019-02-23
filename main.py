### Mini project 2
### Comp 551, McGill University
### author: Sidi Yang

import os
import pandas as pd
import re
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


'''
Step 1: Load Data
'''
# current path
cwd = os.getcwd()

# read pos files
positive_folder =[]
pos_path = cwd + '/data/train/pos/'
for x in os.listdir(pos_path):
    if x.endswith(".txt"):
        positive_folder.append(x)
positiveReviews = []
for pfile in positive_folder:
    with open(pos_path+pfile, encoding="utf8") as f:
        positiveReviews.append(f.read())

# read neg files
negative_folder =[]
neg_path = cwd + '/data/train/neg/'
for x in os.listdir(neg_path):
    if x.endswith(".txt"):
        negative_folder.append(x)
negativeReviews = []
for nfile in negative_folder :
    with open(neg_path+nfile, encoding="utf8") as f:
        negativeReviews.append(f.read())

# read test files
# ref: https://cs.mcgill.ca/~wlh/comp551/slides/12-ensembles.pdf
def read_test():
    data = []
    folder = cwd + "/data/test/"
    for i in range(25000):
        with open(os.path.join(folder, str(i)+".txt"), "rb") as f:
            review = f.read().decode("utf-8").replace('\n','').strip().lower()
            data.append([review,i])
    return data

testReviews = read_test()
###################
### make dataframe for train and test
train_reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "target":1}),
    pd.DataFrame({"review":negativeReviews, "target":0}),
])

test_reviews = pd.concat([
    # set targets of test reviews to 0 for now
    pd.DataFrame({"review":testReviews, "target":0})])


'''
Step 2: Preprocess Data
'''
### convert to strings for pre-processing
train_reviews['review'] = train_reviews['review'].astype(str)
test_reviews['review'] = test_reviews['review'].astype(str)

def clean_text(raw_text):
  cleantext = re.sub(re.compile('<.*?>|[0-9]'), '', raw_text)
  cleantext = re.sub(re.compile('[\-\+!@#$%^&*()<>?()\|\/]'), '', cleantext)
  cleantext = cleantext.replace('<br>','')
  cleantext= cleantext.replace('</br>', '')

  return cleantext
train_reviews['review'] = train_reviews['review'].apply(clean_text)
test_reviews['review'] = test_reviews['review'].apply(clean_text)
###
### Below are stemming, lemmatization and remove stopwords
### As found during validation, they lowered the accuracy.
### Therefore, they won't be used in training final model.
# ### stemming and lemmatization
# ps = PorterStemmer()
# lem = WordNetLemmatizer()
#
# def stem_reviews(review):
#     words = review.split()
#     words = [ps.stem(word) for word in words]
#     return ' '.join(words)
#
# def lemmatize_reviews(review):
#     words = review.split()
#     lemmatized_words = [lem.lemmatize(word) for word in words]
#     return ' '.join(lemmatized_words)
#
# train_reviews['review'] = train_reviews['review'].apply(stem_reviews)
# train_reviews['review'] = train_reviews['review'].apply(lemmatize_reviews)
# test_reviews['review'] = test_reviews['review'].apply(stem_reviews)
# test_reviews['review'] = test_reviews['review'].apply(lemmatize_reviews)
#
# ## stopwords remove
# ## consider to remove this part as it is not improving at all lol!!!
# en_words = set(stopwords.words('english'))
# train_reviews['review'] = [' '.join([w for w in x.lower().split() if w not in en_words])
#     for x in train_reviews['review'].tolist()]
#
# en_words = set(stopwords.words('english'))
# test_reviews['review'] = [' '.join([w for w in x.lower().split() if w not in en_words])
#     for x in test_reviews['review'].tolist()]

'''
Step 3: Model Training
'''
### training model
### split training dataset into training set and validation set
X_train, X_validation, y_train, y_validation = train_test_split(train_reviews['review'],
                                                    train_reviews['target'],
                                                    train_size=0.8,
                                                    test_size=0.2)
### Bag of Words
cv = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
#cv = CountVectorizer(binary=True, stop_words='english',ngram_range=(1, 2)).fit(X_train)
X_train_counts = cv.transform(X_train)
X_validation_counts = cv.transform(X_validation)

### Best model during test
### Better Features
# step 1: tfidf
tfidf_transformer = TfidfTransformer(norm='l2').fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_validation_tfidf = tfidf_transformer.transform(X_validation_counts)

# step 2: normalization
#
normalizer_tranformer = Normalizer().fit(X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_validation_normalized = normalizer_tranformer.transform(X_validation_tfidf)

# use metrics.classification_report to present the results
def display_results(true_result, pred_result,head):
    print(metrics.classification_report(true_result, pred_result))
    print("Accuracy % = ", metrics.accuracy_score(true_result, pred_result))

# models
time_lr_start = datetime.datetime.now()
clf_LR = LogisticRegression().fit(X_train_normalized, y_train)
y_vali1_pred = clf_LR.predict(X_validation_normalized)
display_results(y_validation, y_vali1_pred,"Logistic Regression Results")
time_lr_end = datetime.datetime.now()
time_lr = time_lr_end - time_lr_start
print(time_lr.microseconds)
print(time_lr)

time_mnb_start = datetime.datetime.now()
clf_MB = MultinomialNB(alpha=1.0).fit(X_train_normalized, y_train)
y_vali2_pred = clf_MB.predict(X_validation_normalized)
display_results(y_validation, y_vali2_pred,"Multi Naive Bayes Results")
time_mnb_end = datetime.datetime.now()
time_mnb = time_mnb_end - time_mnb_start
print(time_mnb.microseconds)
print(time_mnb)

time_svm_start = datetime.datetime.now()
clf_SVM = LinearSVC().fit(X_train_normalized, y_train)
y_vali3_pred = clf_SVM.predict(X_validation_normalized)
display_results(y_validation, y_vali3_pred,"SVM Results")
time_svm_end = datetime.datetime.now()
time_svm = time_svm_end - time_svm_start
print(time_svm.microseconds)
print(time_svm)

# time_dt_start = datetime.datetime.now()
# clf_DT = DecisionTreeClassifier().fit(X_train_normalized, y_train)
# y_vali4_pred = clf_DT.predict(X_validation_normalized)
# display_results(y_validation, y_vali4_pred,"Decision Trees Results")
# time_dt_end = datetime.datetime.now()
# time_dt = time_dt_end - time_dt_start
# print(time_dt.microseconds)
# print(time_dt)

### apply to test data and create Kaggle submission file
X_test = test_reviews['review']
X_test_counts = cv.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)

y_test1_pred = clf_LR.predict(X_test_normalized)
pd.DataFrame(y_test1_pred).to_csv("test_data_prediction_LR.csv")

y_test2_pred = clf_MB.predict(X_test_normalized)
pd.DataFrame(y_test2_pred).to_csv("test_data_prediction_MNB.csv")

y_test3_pred = clf_SVM.predict(X_test_normalized)
pd.DataFrame(y_test3_pred).to_csv("test_data_prediction_SVM.csv")

# y_test4_pred = clf_DT.predict(X_test_normalized)
# pd.DataFrame(y_test4_pred).to_csv("test_data_prediction_DT.csv")

'''
Step 4: Pipelines
'''
### Two Pipeline
### Pipeline 1: tf-idf
### counts -> tf-idf -> normalizing -> estimator
time_p1_start = datetime.datetime.now()
pclf1 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
   # ('clf', LinearSVC()),
    ('clf', MultinomialNB()),
])


pclf1 = pclf1.fit(X_train, y_train)
y_pip_pred = pclf1.predict(X_validation)
display_results(y_vali1_pred, y_pip_pred,"")

time_p1_end = datetime.datetime.now()
time_p1 = time_p1_end - time_p1_start
print(time_p1.microseconds)
print(time_p1)

y_pip1_pred = pclf1.predict(X_test)
pd.DataFrame(y_pip1_pred).to_csv("test_data_pip1_prediction.csv")

### Pipeline 2: binary occurence
time_p2_start = datetime.datetime.now()

pclf2 = Pipeline([
    ('vect', CountVectorizer(binary=True)),
    ('norm', Normalizer()),
   # ('clf', LinearSVC()),
    ('clf', MultinomialNB()),
])

pclf2 = pclf2.fit(X_train, y_train)
y_pip_pred = pclf2.predict(X_validation)
display_results(y_vali1_pred, y_pip_pred,"")

time_p2_end = datetime.datetime.now()
time_p2 = time_p2_end - time_p2_start
print(time_p2.microseconds)
print(time_p2)


y_pip2_pred = pclf2.predict(X_test)
pd.DataFrame(y_pip2_pred).to_csv("test_data_pip2_prediction.csv")

'''
Step 5: Validation pipeline
'''
# ref: https://colab.research.google.com/drive/1LQuuM9oNuQhX16jyMoD2ekkIvJ4nefHd
# Orignially from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


### Randomized Search and Cross Validation
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as randint
from scipy.stats import uniform

params = {"vect__ngram_range": [(1,2),(1,3),(1,4)],
          "tfidf__use_idf": [True, False],
          "clf__alpha": uniform(1e-2, 1e-3)}

seed = 551 # Very important for repeatibility in experiments!

random_search = RandomizedSearchCV(pclf1, param_distributions = params, cv=2,
                                   verbose = 10, random_state = seed, n_iter = 1)
random_search = random_search.fit(X_train, y_train)
#
#
# ### CV Results and Final Eval
report(random_search.cv_results_)
y_cv_pred = random_search.predict(X_validation)
display_results(y_validation, y_cv_pred,"Validation Pipline Result")
