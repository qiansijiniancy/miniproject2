import os
import pandas as pd
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
print(reviews.head())

# save to csv
reviews.to_csv('pos_neg_list.csv',index = False)

#save to txt
with open("pos_neg_list.txt", "w") as output:
    output.write(str(reviews))
