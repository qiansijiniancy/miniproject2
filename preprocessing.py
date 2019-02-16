#import loadfiles
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import *


ps = PorterStemmer()

# author: Sidi Yang

# read reviews
cwd = os.getcwd()

# # present first five
# print(imdb_reviews.head(5))
#
# #read review content
# review_content = imdb_reviews["review"]
#
# #preprocessing on the content
# #review_content = review_content.lower().split()
# #print(review_content.head(5))
# tokens = imdb_reviews


def load_doc(filename):
    # 以只读方式打开文件
    file = open(filename, encoding = "utf-8")
    # 读取所有文本
    text = file.read()
    # 关闭文件
    file.close()
    return text
