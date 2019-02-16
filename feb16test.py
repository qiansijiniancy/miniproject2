## https://www.twilio.com/blog/2017/12/sentiment-analysis-scikit-learn.html
from sklearn.feature_extraction.text import CountVectorizer

data = []
data_labels = []
with open("./pos.txt") as f:
    for i in f:
        data.append(i)
        data_labels.append('pos')

with open("./neg.txt") as f:
    for i in f:
        data.append(i)
        data_labels.append('neg')
vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() # for easy usage
# test_size = 0.5 cuz 25,000 + 25,000 =50,000
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80,
        random_state=1234)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()

log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)

import random
j = random.randint(0,len(X_test)-7)
for i in range(j,j+7):
    print(y_pred[0])
    ind = features_nd.tolist().index(X_test[i].tolist())
    print(data[ind].strip())


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
