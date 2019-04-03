# --------------------Load libraries and packages--------------------
import sys
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
# --------------------Load libraries and packages--------------------

# --------------------Load the input file--------------------
if len(sys.argv) != 2:
    exit('Params missing: test_file_path')

file_path = sys.argv[1]

test_corpus = []
if file_path[-4:] != ".txt":
    exit("Must be a .txt file")
else:
    with open(file_path, 'r') as f:
        for line in f:
            test_corpus.append(line.strip())
raw_test_corpus = test_corpus.copy()
# --------------------Load the input file--------------------

# --------------------Construct stopwords for stock tickers--------------------

# read stock list and build a list of stopwords for these stock tickers
stock_list = pd.read_csv('train/data/stocks_cleaned.csv')
stock_list.columns = ['ticker', 'company']


def build_stoplist(df):
    stoplist = set()
    for index, row in df.iterrows():
        stoplist.add(row.ticker.lower())
        stoplist.update(row.company.lower().split())
    return stoplist


stock_stop = build_stoplist(stock_list)
stop = set(stopwords.words('english'))
# --------------------Construct stopwords for stock tickers--------------------


# --------------------Clean Text--------------------
def pre_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_word(word):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def pre_text(text):
    '''
    This function cleans the text
    '''
    processed_text = []
    text = text.lower()
    # remove link
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', text)
    # remove 2 more dots
    text = re.sub(r'\.{2,}', ' ', text)
    text = text.strip(' >"\'')
    words = text.split()
    # remove stopwords
    words = [word for word in words if word not in stock_stop and word not in stop]
    # remove too long or too short word
    for word in words:
        word = pre_word(word)
        if is_word(word) and len(word) >= 2 and len(word) <= 10:
            processed_text.append(word)
    # remove punctuation
    new_text = ' '.join(processed_text)
    new_text = re.sub(r"[^\w\s]", "", new_text)
    return new_text


for i in range(0, len(test_corpus)):
    test_corpus[i] = pre_text(test_corpus[i])

# --------------------Clean Text--------------------

# --------------------Predict--------------------
logmodel = joblib.load('model/classifier.pkl')
countvector = joblib.load('model/countvector.pkl')

# construct count vector for test data
count_matrix_test = countvector.transform(test_corpus)
df_count_test = pd.DataFrame(count_matrix_test.toarray())
y_pred = logmodel.predict(df_count_test)

# print the result
print("There are {} text to be predict.".format(len(test_corpus)))
for i in range(0, len(raw_test_corpus)):
    print("\t The text is: '{}'".format(raw_test_corpus[i]))
    print("\t The sentiment is: {}".format("Positive" if y_pred[i] == 1 else "Negative"))
# --------------------Predict--------------------
