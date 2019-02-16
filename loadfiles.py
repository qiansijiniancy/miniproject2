import os
import pandas as pd
import re
import numpy as np
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

reviews = pd.concat([
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
# prove it works
print(reviews.head())

# save to csv
reviews.to_csv('pos_neg_list.csv',index = False)
pos_reviews.to_csv('pos.csv',index = False)
neg_reviews.to_csv('neg.csv',index = False)

#save to txt
with open("pos_neg_list.txt", "w") as output:
    output.write(str(reviews))

with open("pos.txt", "w") as output:
    output.write(str(pos_reviews))

with open("neg.txt", "w") as output:
    output.write(str(neg_reviews))
#

















'''
# reviews = reviews[["review", "target"]].sample(frac=1, random_state=1)
# train = reviews[reviews.target!=-1].sample(frac=0.6, random_state=1)
# valid = reviews[reviews.target!=-1].drop(train.index)
# test = reviews[reviews.target==-1]
#
# print(train.shape)
# print(valid.shape)
# print(test.shape)

# preprocssing
# DataFrame object has no attribute "lower"
#lower_comm = neg_reviews.lower()
#def preprocessing_data(text):
   # pass

'''

