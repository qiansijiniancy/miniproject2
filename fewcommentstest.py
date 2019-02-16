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

#test_folder = [x for x in os.listdir(cwd+"/data/test/") if x.endswith(".txt")]

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

train_reviews = pd.concat([
    pd.DataFrame({"review":positiveReviews, "target":1}),
    pd.DataFrame({"review":negativeReviews, "target":0}),
    #pd.DataFrame({"review":testReviews, "target":-1})
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

## preprocessing
from nltk.corpus import stopwords
import string
# spelling correction
# convert to lower cases
train_reviews['review'] = train_reviews['review'].str.lower()
# import re
# ## ref: https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
# REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
# REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
# def preprocess_reviews(review):
#     review = [REPLACE_NO_SPACE.sub("", line.lower()) for line in review]
#     review = [REPLACE_WITH_SPACE.sub(" ", line) for line in review]
#
#     return review

# train_reviews['review'] = preprocess_reviews(train_reviews['review'])
#test_reviews['review'] = test_reviews['review'].str.lower()
# stemming
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# def stem_sentences(sentence):
#     tokens = sentence.split()
#     stemmed_tokens = [stemmer.stem(token) for token in tokens]
#     return ' '.join(stemmed_tokens)

corpus = pd.concat([train_reviews,test_reviews], axis=0)
print(corpus['review'].head(10))

# ### ref: http://python.jobbole.com/81397/
# def stem_words_array(words_array):
#     stemmer = nltk.PorterStemmer();
#     stemmed_words_array = [];
#     for word in words_array:
#         stem = stemmer.stem(word);
#         stemmed_words_array.append(stem);
#     return stemmed_words_array;
from nltk import word_tokenize


def clean_text(text):
    """
    Applies some pre-processing on the given text.

    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer
# Transform each text into a vector of word counts
vectorizer = CountVectorizer(stop_words="english",
                             preprocessor=clean_text)

training_features = vectorizer.fit_transform(train_reviews["review"])
test_features = vectorizer.transform(test_reviews["review"])

# Training
from sklearn.naive_bayes import MultinomialNB

#alpha parameters stands for Add-One(Laplace) Smooting
model = MultinomialNB(alpha=1.0)
model.fit(training_features, train_reviews["target"])
y_pred = model.predict(test_features)

# Evaluation
acc = accuracy_score(test_reviews["target"], y_pred)

print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
#
# corpus = pd.concat([train_reviews,test_reviews], axis=0)
# # # save to csv
# # reviews.to_csv('pos_neg_list.csv',index = False)
# # pos_reviews.to_csv('pos.csv',index = False)
# # neg_reviews.to_csv('neg.csv',index = False)
# #
# # #save to txt
# # with open("pos_neg_list.txt", "w") as output:
# #     output.write(str(reviews))
# #
# # with open("pos.txt", "w") as output:
# #     output.write(str(pos_reviews))
# #
# # with open("neg.txt", "w") as output:
# #     output.write(str(neg_reviews))
# # #
#
# # read top 10 for test
# # reviews_top_10 = train_reviews.head(10)
# # print(reviews_top_10)
#
# # simple multionomialNV with no preprocssing
# #train_and_test = pd.concat([train_reviews, text_reviews], axis=0)
#
# # Training and Test data
# # test_size = 0.5 cuz 25,000 + 25,000 =50,000
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(corpus['review'],
#                                                     corpus['target'],
#                                                     test_size=0.5)
#
# print(len(X_train), len(y_train), len(X_test), len(y_test))
#
# #
# # ############################
# # ## SVC
# # from sklearn.svm import LinearSVC
# # from sklearn.model_selection import GridSearchCV
# # from sklearn.metrics import f1_score
# # from sklearn.naive_bayes import MultinomialNB
# #
# # from sklearn.feature_extraction.text import CountVectorizer
# # vec = CountVectorizer()
# # X_train = vec.fit_transform(X_train)
# # X_test = vec.transform(X_test)
# #
# # model = LinearSVC(C=4,loss='squared_hinge')
# # model.fit(X_train, y_train)
# # pred = model.predict(X_test)
# # print("The F1 accuracy score: {}%".format(f1_score(y_test, pred) * 100))
# #
# #
# # ####################
# # Multi Nominal NB
# from sklearn.feature_extraction.text import CountVectorizer
# vec = CountVectorizer()
# X_train = vec.fit_transform(X_train)
# X_test = vec.transform(X_test)
#
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import f1_score
#
# #alpha parameters stands for Add-One(Laplace) Smooting
# model = MultinomialNB(alpha=1.0)
#
# model.fit(X_train, y_train)
#
# pred = model.predict(X_test)
#
#
# print("The F1 accuracy score: {}%".format(f1_score(y_test, pred) * 100))
#
# pd.DataFrame(pred).to_csv("prediction.csv")
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
#
#
#
#
# '''
# # reviews = reviews[["review", "target"]].sample(frac=1, random_state=1)
# # train = reviews[reviews.target!=-1].sample(frac=0.6, random_state=1)
# # valid = reviews[reviews.target!=-1].drop(train.index)
# # test = reviews[reviews.target==-1]
# #
# # print(train.shape)
# # print(valid.shape)
# # print(test.shape)
#
# # preprocssing
# # DataFrame object has no attribute "lower"
# #lower_comm = neg_reviews.lower()
# #def preprocessing_data(text):
#    # pass
#
# # stemming
#
# from nltk.stem.porter import *
# stemmer = PorterStemmer()
#
# # import re
# # # 这里需要有遍历
# # def preprocess(text):
# #     cleaned = text.replace('...'," ")
# #     cleaned = ''.join(cleaned)
# #     return cleaned
#
# #stemma(train_reviews['review'])
# train_reviews['review'] = preprocess(train_reviews['review'])
# #test_reviews['review'] = stem_words_array(test_reviews['review'])
# print(corpus['review'].head(10))
#
# '''
#
