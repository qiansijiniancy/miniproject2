import pandas as pd
import nltk

import re

df = pd.read_csv("pos_neg_list.csv")
df40 = df.head(40)
print(df40)

train_data = df40.head(20)
test_data = df40.iloc[21:-1]

import re
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

'''
Exact Features
'''

from sklearn.feature_extraction.text import CountVectorizer
# Transform each text into a vector of word counts
vectorizer = CountVectorizer(stop_words="english",
                             preprocessor=clean_text)

training_features = vectorizer.fit_transform(train_data["review"])
test_features = vectorizer.transform(test_data["review"])

print(training_features.shape)

## Bernoulli Naive Bayes


##### SVM from scikit
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC



# Training
model = LinearSVC()
model.fit(training_features, train_data["target"])
y_pred = model.predict(test_features)

# Evaluation
acc = accuracy_score(test_data["target"], y_pred)

print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

'''
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import CountVectorizer
# Transform each text into a vector of word counts
vectorizer = CountVectorizer()

training_features = vectorizer.fit_transform(train_data["review"])
test_features = vectorizer.transform(test_data["review"])

# Training
model = LinearSVC()
model.fit(training_features, train_data["target"])
y_pred = model.predict(test_features)

# Evaluation
acc = accuracy_score(test_data["target"], y_pred)

print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))

'''