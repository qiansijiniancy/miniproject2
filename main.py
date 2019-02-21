## feb 17 version
## Updates:
## 1. test data order (should be lol)fixed
## 2. Remove unuseful codes

import os
import pandas as pd
import nltk
import re
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
import math

#nltk.download('wordnet')
#nltk.download('stopwords')



# author: Sidi Yang , Negin Ashouri


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

test_review_file = pd.read_csv('/home/negin/Desktop/Python/ML mini proj2/miniproject2/data/test.csv',encoding="latin1")

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

for each in pos_reviews:
    output = re.sub(r'\d+', '', each)
    print("This is pos reviews without number", output)
split_words = str(pos_reviews).split() # type is list of str , split_words = feature_set

  # each['target'] = each['target'].str.replace('\d+', '')
print("this is words of each review", split_words)

for file in output:
    feature_set = set(output)
print("this is feature set####################### ", feature_set)



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
print(corpus['review'].tail(10))


'''
Step 3: Model Training
'''
### training model
### split training dataset into training set and validation set
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(binary=False, stop_words='english', max_features=150000, ngram_range = (1, 2))
X_train, X_validation, y_train, y_validation = train_test_split(train_reviews['review'],
                                                    train_reviews['target'],
                                                    train_size=0.8,
                                                    test_size=0.2)


print("This is x train##########################",X_train)



## Bernoulli Naive Bayes


def get_f_counts(data, feature_set):
    x_counts = [0] * 160
    counts = dict(Counter(words_data))
    for word, zingul in counts.items():
        x_counts[feature_set.index(word)] = zingul
    return x_counts

#Split Sentences to words

split_words = str(train_reviews).split() # type is list of str , split_words = feature_set
for each in split_words :
    words_data = re.sub(r'\d+', '', each)
  # each['target'] = each['target'].str.replace('\d+', '')
    if each['target'] == 0:
        feature_neg = each
    else :
        feature_pos = each

    print(words_data)
#remove_target =  train_reviews['target'].str.replace('\d+', '')

#set of features
feature_set = set(words_data)

#number of each feature in data
for data in words_data:
    # Extract word count feature
    feature_count = get_f_counts(data, feature_set)

#targets = {0,1}
def feature_prob(feature_set, targets)
    for feature in feature_set:
        for tartget in targets :
            prob[feature][target] = feature_count[target] / allfeature_counts[target]




for data in reviews #in main
    BNB_predict(data, target)

def BNB_sigma(X_validation): #X_validation = each review
    sum_all = 0
    for x in X_validation :
        sum_all += BNB_sumx(x)
    sigma = (math.log10(tetha1/(1-tetha1))) + sum_all

def BNB_prediction(sigma, Y_validation)
    if(sigma > 0): ]# Predict the target
        pred_tar[x_val][predict] = '1' # and classify as 1
    else if (sigma < 0)
        pred_tar[x_val][predict] = '0' # and classify as 0

    if pred_tar[x_val][predict] == feature??[x_val][target]
        Accuracy




def BNB_sumx(x):
    sum_part_one = x * (math.log10(feature_prob(x,1) / feature_prob(x,0)))
    sum_part_two = (1-x) * (math.log10((1-feature_prob(x,1))/(1- feature_prob(x,0))))
    sum_x = sum_part_one + sum_part_two
    return sum_x





import re
def to_words(X_train):
    return re.findall(r'\w+', X_train)
textsample = to_words(train_reviews)
feature_set = set(textsample)
print(feature_set)


tota1 = number of examples where y=1 / total number of examples
targets = real targets
count =0 # number of all the features we are considering
feature_set = set(['negar', 'negin' ,'negar'])
feature_name[]
#Split Sentences to words
split_words = str(train_reviews).split(" ")
print(split_words)

#extract the features
if corpus['target'] == 0 :

#creating a set of features
for target in targets :
    feature_set = set(feature_name[target])
    count[target]+=1  # number of all the features we are considering

numOfFeatures = set(feature_name[target]) #number of feature occurances in target(do we have it?)

#number of occurances of each feature in neg & pos
if corpus['target'] == 0 :
    feature_set[0][feature_name]
#going through the whole data and if it is 0, add to feature_set[0] age 1 add to feature_set[1]


#start the ocurrances in diff targets(in 1, 0)

# calculating the prob s

# Do the prediction in the prob s



#Bernoulli train

#Bernoulli Predict(how to say classify as 1 or 0)

#Results accuracy

for data in dataset:

        # Extract word count feature
        data['x_counts'] = get_x_counts(data, feature_set)

def get_x_counts(data)#, most_freq_words):
    #x_counts = [0] * 160
    counts = dict(Counter(data['text'])) #or 'review'
    for word, count in counts.items():
        if word in feature_set:
            feature_counts[feature_set.index(word)] = count

    return feature_counts

'''
#train
'''
tota1 = number of examples where y=1 / total number of examples
targets = {0,1}
for target in targets
numOfFeatures = set(feature_name[target]) #number of feature occurances of feature in target(do we have it?)
for feature in features:
    for target in targets:
        prob[feature][target] = feature[target] / number of all features in [target]  # teta[j][1] , teta[j][0]


#predict
for
sum =
delta = log(teta1/(1-teta1)) + sum
y_vali4_pred = predict(X_validation_normalized)
#Result
display_results(y_validation, y_vali4_pred,"")
'''




### Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer().fit(X_train)
X_train_counts = cv.transform(X_train)
X_validation_counts = cv.transform(X_validation)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

### Better Features
# tfidf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_validation_tfidf = tfidf_transformer.transform(X_validation_counts)

# Normalization
from sklearn.preprocessing import Normalizer
normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_validation_normalized = normalizer_tranformer.transform(X_validation_tfidf)

# use metrics.classification_report to present the results
def display_results(y_val, y_pred, heading):
    print(metrics.classification_report(y_val, y_pred))
    print("Accuracy % = ", metrics.accuracy_score(y_val, y_pred))

# models
clf_LR = LogisticRegression().fit(X_train_normalized, y_train)
y_vali1_pred = clf_LR.predict(X_validation_normalized)
display_results(y_validation, y_vali1_pred,"")

clf_MB = MultinomialNB(alpha=1.0).fit(X_train_normalized, y_train)
y_vali2_pred = clf_MB.predict(X_validation_normalized)
display_results(y_validation, y_vali2_pred,"")

clf_SVM = LinearSVC().fit(X_train_normalized, y_train)
y_vali3_pred = clf_SVM.predict(X_validation_normalized)
display_results(y_validation, y_vali3_pred,"")












### here needs to be test on Feb 18:
X_test = test_reviews['review']
X_test_counts = cv.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)

y_test1_pred = clf_LR.predict(X_test_normalized)

pd.DataFrame(y_test1_pred).to_csv("test_data_prediction_LR.csv")

y_test2_pred = clf_MB.predict(X_test_normalized)
#pd.DataFrame(y_pred).columns = ['Id','Category']
pd.DataFrame(y_test2_pred).to_csv("test_data_prediction_MNB.csv")

y_test3_pred = clf_SVM.predict(X_test_normalized)
#pd.DataFrame(y_pred).columns = ['Id','Category']
pd.DataFrame(y_test3_pred).to_csv("test_data_prediction_SVM.csv")

print(X_test.iloc[12])
print(type(test_reviews))


## k-fold cross validation
from sklearn.model_selection import KFold
