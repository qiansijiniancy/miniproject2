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

# test_folder =[]
# for x in os.listdir(cwd + '/data/test/'):
#     if x.endswith(".txt"):
#         test_folder.append(x)

test_review_file = pd.read_csv('/Users/cindyang/Desktop/Imdb_sentiment_analysis/test.csv',encoding="latin1")

###################

positiveReviews, negativeReviews, testReviews = [], [], []

for pfile in positive_folder:
    with open(cwd+"/data/train/pos/"+pfile, encoding="utf8") as f:
        positiveReviews.append(f.read())
for nfile in negative_folder :
    with open(cwd+"/data/train/neg/"+nfile, encoding="utf8") as f:
        negativeReviews.append(f.read())
# for tfile in test_folder:
#     with open(cwd+"/data/test/"+tfile, encoding="utf8") as f:
#         testReviews.append(f.read())

# train_reviews
train_reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "target":1}),
    pd.DataFrame({"review":negativeReviews, "target":0}),
])
#],
# ignore_index=True).sample(frac=1, random_state=1)


pos_reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "target":1}),
    #pd.DataFrame({"review":negativeReviews, "target":0}),
#])
 ],
     ignore_index=True).sample(frac=1, random_state=1)

neg_reviews = pd.concat([
    #pd.DataFrame({"review":positiveReviews, "target":1}),
    pd.DataFrame({"review":negativeReviews, "target":0}),
#])
 ],
     ignore_index=True).sample(frac=1, random_state=1)

test_reviews = pd.concat([
    pd.DataFrame({"review":test_review_file["review"], "target":0})])
# prove it works
print(train_reviews.shape)

test_reviews = test_reviews.sort_index()
print(test_reviews['review'].head(10))
print(test_reviews['review'].tail(10))


'''
Step 2: Preprocess Data
'''
# form corpus with train and test
corpus = pd.concat([train_reviews,test_reviews], axis=0)
corpus['review'] = corpus['review'].str.lower()

print(corpus['review'].head(10))
print(type(corpus['review']))
print(len(corpus))
### https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
### https://github.com/iworeushankaonce/Imdb/blob/master/Imdb_sentiment_polarity%20(1).ipynb
### convert review contents to string & useful for pre-processing
corpus['review'] = corpus['review'].astype(str)
train_reviews['review'] = train_reviews['review'].astype(str)
test_reviews['review'] = test_reviews['review'].astype(str)

import re
def clean_htmlandsymbol(raw_text):
  cleantext_nohtml = re.sub(re.compile('<.*?>|[0-9]'), '', raw_text)
  cleantext_nosymbolandhtml = re.sub(re.compile('[\-\+!@#$%^&*()<>?()\|\/]'), '', cleantext_nohtml)

  return cleantext_nosymbolandhtml
corpus['review'] = corpus['review'].apply(clean_htmlandsymbol)
train_reviews['review'] = train_reviews['review'].apply(clean_htmlandsymbol)
test_reviews['review'] = test_reviews['review'].apply(clean_htmlandsymbol)

#
# #### stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

corpus['review'] = corpus['review'].apply(stem_sentences)
train_reviews['review'] = train_reviews['review'].apply(stem_sentences)
test_reviews['review'] = test_reviews['review'].apply(stem_sentences)
#
# #####lemma
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
def lemmatize_sentences(sentence):
    tokens = sentence.split()
    lemmatized_tokens = [lmtzr.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

corpus['review'] = corpus['review'].apply(lemmatize_sentences)

corpus['review'] = corpus['review'].apply(lemmatize_sentences)
train_reviews['review'] = train_reviews['review'].apply(lemmatize_sentences)
test_reviews['review'] = test_reviews['review'].apply(lemmatize_sentences)

### stopwords remove
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

en_words = set(stopwords.words('english'))
corpus['review'] = [' '.join([w for w in x.lower().split() if w not in en_words])
    for x in corpus['review'].tolist()]


en_words = set(stopwords.words('english'))
train_reviews['review'] = [' '.join([w for w in x.lower().split() if w not in en_words])
    for x in train_reviews['review'].tolist()]

en_words = set(stopwords.words('english'))
test_reviews['review'] = [' '.join([w for w in x.lower().split() if w not in en_words])
    for x in test_reviews['review'].tolist()]


print(corpus['review'].head(10))
print(corpus['review'].iloc[10])
print(test_reviews['review'].head(20))
print(test_reviews['review'].iloc[20])

print(type(corpus))
print(type(test_reviews))


'''
Step 3: Model Training
'''
# ####training model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
tfidf = TfidfVectorizer(binary=False, stop_words='english', max_features=150000, ngram_range = (1, 2))
X_train, X_validation, y_train, y_validation = train_test_split(train_reviews['review'],
                                                    train_reviews['target'],
                                                    train_size=0.9,
                                                    test_size=0.1)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer().fit(X_train)
X_train_counts = cv.transform(X_train)
X_validation_counts = cv.transform(X_validation)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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

clf_LR = LogisticRegression().fit(X_train_normalized, y_train)
y_pred = clf_LR.predict(X_validation_normalized)
display_results(y_validation, y_pred,"")

clf_MB = MultinomialNB(alpha=1.0).fit(X_train_normalized, y_train)
y_pred = clf_MB.predict(X_validation_normalized)
display_results(y_validation, y_pred,"")

clf_SVM = LinearSVC().fit(X_train_normalized, y_train)
y_pred = clf_SVM.predict(X_validation_normalized)
display_results(y_validation, y_pred,"")

X_train, X_test, y_train, y_test = train_test_split(corpus['review'],
                                                    corpus['target'],
                                                    train_size=0.5,
                                                    test_size=0.5)
X_test_counts = cv.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)

y_pred = clf_LR.predict(X_test_normalized)
pd.DataFrame(y_pred).to_csv("csv_test_func_test_data_prediction_LR.csv")

y_pred = clf_MB.predict(X_test_normalized)
#display_results(y_test,y_pred,"")
from sklearn.metrics import f1_score
#print("The F1 accuracy score: {}%".format(f1_score(y_test, y_pred) * 100))
#pd.DataFrame(y_pred).columns = ['Id','Category']
pd.DataFrame(y_pred).to_csv("csv_test_data_prediction_MNB.csv")

y_pred = clf_SVM.predict(X_test_normalized)
#pd.DataFrame(y_pred).columns = ['Id','Category']
pd.DataFrame(y_pred).to_csv("test_data_prediction_SVM.csv")

###     writer = csv.writer(results, delimter = ",")
###TypeError: 'delimter' is an invalid keyword argument for this function
import csv
with open('csv_Test_SVM_test_results.csv', 'w') as results:
    writer = csv.writer(results)
    writer.writerow(['Id','Category'])
    for i in range(25000):
        writer.writerow([i ,y_pred[i]])