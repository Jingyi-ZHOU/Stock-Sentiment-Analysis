# --------------------Load libraries and packages--------------------
import re
from bs4 import BeautifulSoup
import urllib
import pandas as pd
import requests
import json
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
# --------------------Load libraries and packages--------------------


# --------------------Web Scrapping--------------------
url = 'https://listingcenter.nasdaq.com/IssuersPendingSuspensionDelisting.aspx'
response = urllib.request.urlopen(url)
soup = BeautifulSoup(response, 'html.parser')
table = soup.find('table', {'class': 'rgMasterTable'})

stock_list = []
for tr in table.tbody.findAll('tr'):
    row = []
    for td in tr.findAll('td'):
        text = td.getText()
        if text == '\xa0':
            text = 'None'
        row.append(text)
    stock_list.append(row)

df = pd.DataFrame(stock_list)
df.columns = ['Issuer Name', 'Symbol', 'Reason', 'Status', 'Effective Date', 'Form 25 Date']

# Filter results with the status, as some may resume trading.
suspended_df = df[df['Status'] == "Suspended"]

# 1. Build query structure
# query_list0 is for Google News; query_list1 is for Globenewswire
ticker = df.to_dict('index')


def build_query_list():
    query_list0 = []
    query_list1 = []
    for i in ticker:
        comp = re.sub(r'[,\.]', ' ', ticker[i]['Issuer Name']).split()
        str_comp = '+'.join(comp)
        str_comp1 = ' '.join(comp)
        comp.append(ticker[i]['Symbol'])
        query = '+'.join(comp)
        query1 = ' '.join(comp)
        query_list0.append(query)
        query_list1.append(query1)
    return (query_list0, query_list1)


(query_list0, query_list1) = build_query_list()


# After testing different combinations, we found that company_name+ticker is best for Globenewswire, while company_name+ticker+"stock"+"suspended" is bester for Google News.

# After testing different combinations of selecting the top-k retrieved results, we found that simply selecting the top 1 result from both news sources give a nice peroformance.

# 2.1 Google News Crawler

# return top 1 results
def parseSingleNews(query):
    url = 'https://news.google.com/rss/search?q=' + query + '+stock+suspend&hl=en-US&gl=US&ceid=US:en'
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, 'html.parser')

    channel = soup.find('channel')
    news_list = []
    for item in soup.findAll('item')[0:1]:
        des = ''
        abst = ''
        description = item.find('description').get_text()
        match = re.search(r'(href=")(.*)(" target)', description)
        match1 = re.search(r'(<p>)(.*)(</p>)', description)
        if (match):
            des = match.group(2)
        if (match1):
            abst = match1.group(2)
        news_item = {
            'title': item.find('title').get_text(),
            'pubdate': item.find('pubdate').get_text(),
            'link': des,
            'abstract': abst
        }
        news_list.append(news_item)
    return news_list


def allQuery(query_list):
    dict = {}
    for q in query_list:
        q1 = q.replace('+', ' ')
        news_list = parseSingleNews(q)
        dict[q1] = news_list
    return dict


google_news = allQuery(query_list0)


# 2.2 Globenewswire Crawler

query_list2 = []
for q in query_list1:
    qry = {'keyword': q}
    query_list2.append(qry)

# return top 1 results


def parseSingleNews1(query):
    r = requests.get('https://globenewswire.com/Search', params=query)
    result = BeautifulSoup(r.text, 'html.parser')
    contents = result.find_all("div", attrs={"class": "results-link"})
    news_list = []
    for content in contents[0:1]:
        a = content.find("h1", attrs={"class": "post-title16px"}).find("a")
        title = a.text
        pubdate = content.find("span", attrs={"class": "dt-green"}).get_text()
        link = a['href']
        abstract = content.find_all("p")[1].text
        news_item = {
            'title': title,
            'pubdate': pubdate,
            'link': 'https://globenewswire.com/' + link,
            'abstract': abstract
        }
        news_list.append(news_item)
    return news_list


def allQuery1(query_list):
    dict = {}
    for q in query_list:
        news_list = parseSingleNews1(q)
        dict[q['keyword']] = news_list
    return dict


globe_news = allQuery1(query_list2)

# --------------------Web Scrapping--------------------


# --------------------Merge two sets--------------------
all_news = {}
for key in google_news:
    all_news[key] = google_news[key]
    all_news[key] += globe_news[key]

# save to a dataframe
columns = ["Issuer Name", "Symbol", "Reason", "news_title", "news_link", "news_abstract", "pos_sentiment_score", "neg_sentiment_score", "sentiment_label"]
df_results = pd.DataFrame(columns=columns)

for index, rows in suspended_df.iterrows():
    name = suspended_df.loc[index, "Issuer Name"]
    ticker = suspended_df.loc[index, "Symbol"]
    reason = suspended_df.loc[index, "Reason"]

    n = re.sub(r'[,\.]', ' ', name)
    k = n.strip() + " " + ticker.strip()
    k = re.sub(' +', ' ', k)

    allnews = all_news.get(k)
    for news in allnews:
        news_title = news['title']
        news_link = news['link']
        news_abstract = news['abstract']
        df_results = df_results.append({"Issuer Name": name, "Symbol": ticker, "Reason": reason,
                                        "news_title": news_title, "news_link": news_link, "news_abstract": news_abstract}, ignore_index=True)

# --------------------Merge two sets--------------------


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

# --------------------Clean Text--------------------


# --------------------Predict--------------------
# load the model
logmodel = joblib.load('model/classifier.pkl')
countvector = joblib.load('model/countvector.pkl')

for index, row in df_results.iterrows():
    # clean the title
    processed_text = pre_text(df_results.loc[index, "news_title"])
    # convert to count vector
    count_matrix_test = countvector.transform([processed_text])
    df_count_test = pd.DataFrame(count_matrix_test.toarray())

    label = logmodel.predict(df_count_test)[0]
    pos_score = logmodel.predict_proba(df_count_test)[0][1]
    neg_score = logmodel.predict_proba(df_count_test)[0][0]

    df_results.loc[index, "pos_sentiment_score"] = pos_score
    df_results.loc[index, "neg_sentiment_score"] = neg_score
    df_results.loc[index, "sentiment_label"] = "Positive" if label == 1 else "Negative"

# Save to a CSV file

df_results.to_csv("output.csv", index=False)
# --------------------Predict--------------------
