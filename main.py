### Mini project 2
### Comp 551, McGill University
### author: Sidi Yang

import os
import pandas as pd
import nltk
import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


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
#
# # #### stemming and lemmatization
ps = PorterStemmer()
lem = WordNetLemmatizer()

def stem_reviews(review):
    words = review.split()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

def lemmatize_reviews(review):
    words = review.split()
    lemmatized_words = [lem.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

train_reviews['review'] = train_reviews['review'].apply(stem_reviews)
train_reviews['review'] = train_reviews['review'].apply(lemmatize_reviews)
test_reviews['review'] = test_reviews['review'].apply(stem_reviews)
test_reviews['review'] = test_reviews['review'].apply(lemmatize_reviews)

### stopwords remove
### consider to remove this part as it is not improving at all lol!!!
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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
X_train, X_validation, y_train, y_validation = train_test_split(train_reviews['review'],
                                                    train_reviews['target'],
                                                    train_size=0.8,
                                                    test_size=0.2)
### Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True, stop_words='english',ngram_range=(1, 2)).fit(X_train)
X_train_counts = cv.transform(X_train)
X_validation_counts = cv.transform(X_validation)


from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

### Best model during test
### Better Features
# step 1: tfidf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(smooth_idf=False,norm='l2').fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_validation_tfidf = tfidf_transformer.transform(X_validation_counts)

# step 2: normalization
from sklearn.preprocessing import Normalizer
normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_validation_normalized = normalizer_tranformer.transform(X_validation_tfidf)

# use metrics.classification_report to present the results
def display_results(true_result, pred_result,head):
    print(metrics.classification_report(true_result, pred_result))
    print("Accuracy % = ", metrics.accuracy_score(true_result, pred_result))

# models
clf_LR = LogisticRegression().fit(X_train_normalized, y_train)
y_vali1_pred = clf_LR.predict(X_validation_normalized)
display_results(y_validation, y_vali1_pred,"Logistic Regression Results")

clf_MB = MultinomialNB(alpha=1.0).fit(X_train_normalized, y_train)
y_vali2_pred = clf_MB.predict(X_validation_normalized)
display_results(y_validation, y_vali2_pred,"Multi Naive Bayes Results")

clf_SVM = LinearSVC().fit(X_train_normalized, y_train)
y_vali3_pred = clf_SVM.predict(X_validation_normalized)
display_results(y_validation, y_vali3_pred,"SVM Results")

clf_DT = DecisionTreeClassifier().fit(X_train_normalized, y_train)
y_vali4_pred = clf_DT.predict(X_validation_normalized)
display_results(y_validation, y_vali4_pred,"Decision Trees Results")

# ### prediction and evaluation
# from sklearn import metrics
# y_vali1_pred = clf_LR.predict(X_validation_normalized)
# print(metrics.classification_report(y_validation, y_vali1_pred,
#     tr= train_reviews.target_names))
#
# from sklearn import metrics
# y_vali2_pred = clf_MB.predict(X_validation_normalized)
# print(metrics.classification_report(y_validation, y_vali2_pred,
#     target_names= train_reviews.target_names))
#
# from sklearn import metrics
# y_vali3_pred = clf_SVM.predict(X_validation_normalized)
# print(metrics.classification_report(y_validation, y_vali3_pred,
#     target_names= train_reviews.target_names))


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

y_test3_pred = clf_DT.predict(X_test_normalized)
pd.DataFrame(y_test3_pred).to_csv("test_data_prediction_DT.csv")

print(X_test.iloc[12])
print(type(test_reviews))

'''
Step 4: Pipelines
'''
### Two Pipelines

### Pipeline 1: tf-idf
### counts -> tf-idf -> normalizing -> estimator
from sklearn.pipeline import Pipeline
pclf1 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
    ('clf', MultinomialNB()),
])

pclf1.fit(X_train, y_train)
y_pip_pred = pclf1.predict(X_validation)
print(metrics.classification_report(y_vali1_pred, y_pip_pred))
display_results(y_vali1_pred, y_pip_pred,"")
y_pip1_pred = pclf1.predict(X_test)
pd.DataFrame(y_pip1_pred).to_csv("test_data_pip1_prediction_MNB.csv")

### Pipeline 2: binary occurence

from sklearn.pipeline import Pipeline
pclf2 = Pipeline([
    ('vect', CountVectorizer(binary=True)),
    ('norm', Normalizer()),
    ('clf', MultinomialNB()),
])

pclf2.fit(X_train, y_train)
y_pip_pred = pclf2.predict(X_validation)
print(metrics.classification_report(y_vali1_pred, y_pip_pred))
display_results(y_vali1_pred, y_pip_pred,"")
y_pip2_pred = pclf2.predict(X_test)
pd.DataFrame(y_pip2_pred).to_csv("test_data_pip2_prediction_MNB.csv")

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
random_search.fit(X_train, y_train)
#
#
# ### CV Results and Final Eval
report(random_search.cv_results_)
y_cv_pred = random_search.predict(X_validation)
print(metrics.classification_report(y_validation, y_cv_pred
                                    ))
#    ,target_names=train_reviews.target_names))
