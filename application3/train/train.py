# --------------------Load libraries and packages--------------------
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

# --------------------Load and concatenate data--------------------
data1 = pd.read_excel('data/investing_comments_cleaned.xls', header=None)
data2 = pd.read_excel('data/options_comments_cleaned.xls', header=None)
data3 = pd.read_excel('data/overall_comments_cleaned.xls', header=None)
data4 = pd.read_excel('data/SecurityAnalysis_comments_cleaned.xls', header=None)
data5 = pd.read_excel('data/stocks_comments_cleaned.xls', header=None)
data6 = pd.read_excel('data/wsb_comments_cleaned.xls', header=None)
data7 = pd.read_excel('data/supplement_comments_cleaned.xls', header=None)
data = pd.concat([data1, data2, data3, data4, data5, data6, data7], ignore_index=True)
data.columns = ['text', 'sentiment']
# --------------------Load and concatenate data--------------------


# --------------------Construct stopwords for stock tickers--------------------

# read stock list and build a list of stopwords for these stock tickers
stock_list = pd.read_csv('data/stocks_cleaned.csv')
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


# --------------------Clean Text in the dataframe--------------------
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


# clean the data
data.text = data.text.apply(pre_text)

# --------------------Clean Text in the dataframe--------------------


# --------------------Train model--------------------
# We found the Count Vector with Logistic Regression has the best performace
# We could also try TF-IDF representation if the corpus is large enough
all_words = []
for i in data.text:
    words = word_tokenize(i)
    for word in words:
        all_words.append(word)

labels = []
for i in data.sentiment:
    labels.append(i)
y = pd.Series(labels)

countvector = CountVectorizer(input=all_words, lowercase=True, min_df=2, ngram_range=(1, 1))
count_matrix = countvector.fit_transform(data.text)
feature_names = countvector.get_feature_names()
df_count = pd.DataFrame(count_matrix.toarray(), columns=feature_names)

# tfidf_vectorizer = TfidfVectorizer(input=all_words,lowercase=True, min_df=2, ngram_range=(1, 1))
# tfidf_matrix = tfidf_vectorizer.fit_transform(data.text)
# feature_names = tfidf_vectorizer.get_feature_names()
# df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns = feature_names)

logmodel = LogisticRegression()
logmodel.fit(df_count, y)
# --------------------Train model--------------------


# --------------------Save the trained model--------------------
joblib.dump(countvector, '../model/countvector.pkl')
joblib.dump(logmodel, '../model/classifier.pkl')
# --------------------Save the trained model--------------------
