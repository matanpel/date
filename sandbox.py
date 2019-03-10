

#######       dateXGB train         ################



import pandas as pd
import string
from fastparquet import ParquetFile
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import Counter
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np


# clean data - create corpus


def hood(text):
    sentence = text.split()
    indices = (i for i, word in enumerate(sentence) if '201' in word)
    neighbors = []
    for ind in indices:
        if ind < 3:
            strt = 0
        else:
            strt = ind - 3
        neighbors.append(
            ' '.join(sentence[strt:ind]) + ' ' + ' '.join([sentence[ind]]) + ' ' + ' '.join(sentence[ind + 1:ind + 4]))
    return neighbors


def clean_data(X):
    my_punc = list(string.punctuation)
    #my_punc.remove('-')
    #my_punc.remove('/')
    my_punc = ''.join(my_punc)
    translator = str.maketrans(my_punc, ' ' * len(my_punc))

    corpus = []
    for row in X.itertuples():
        text = row.text
        if text != pd.isnull(text):
            text = str(text).translate(translator)
            # text = text.replace('\n',' ')
            text = hood(text)
            corpus.append(' '.join(text))
    return (corpus)


def freq_calc(date_part, df):
    my_counter = Counter(df[date_part])
    freq_table = pd.DataFrame(columns=[date_part, 'freq'])
    freq_table[date_part] = my_counter.keys()
    freq = []
    for key in my_counter:
        freq.append(my_counter[key] / df.shape[0])
    freq_table['freq'] = freq
    return (freq_table)


#######################################################################################

# read data

df = ParquetFile('/run/user/1000/gvfs/smb-share:server=nas01.local,share=rnd/data/parquet_data/130K_gold.parq').to_pandas()

df['day'] = df['invoice_date'].str.slice(0, 2)
df['month'] = df['invoice_date'].str.slice(3, 5)
df['year'] = df['invoice_date'].str.slice(6, 10)

df.loc[:, 'hood'] = clean_data(pd.DataFrame(df.text))
df = df[df['hood'] != '']

# split data
y = df['day']#.astype('uint8')  ############################# part of date to predict (day\month\year)
X = df[['hood']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True, lowercase=True, max_features=1000)  ########### MODEL
#vectorizer.fit(X.hood)
#word_matrix = vectorizer.transform(X.hood)
vectorizer.fit(X_train.hood)
word_matrix = vectorizer.transform(X_train.hood)

tfidf = TfidfTransformer(norm='l2')  ############# MODEL
tfidf.fit(word_matrix)
tfidf_matrix = tfidf.transform(word_matrix)


test_word_matrix = vectorizer.transform(X_test.hood)
test_tfidf_matrix = tfidf.transform(test_word_matrix)


print('start XG-Boost')
clf = xgb.XGBClassifier(seed=42)
#clf = clf.fit(tfidf_matrix, y)
clf = clf.fit(tfidf_matrix, y_train)

y_pred = clf.predict(test_tfidf_matrix)

print('accuracy: ', accuracy_score(y_test, y_pred, normalize=True))
print('precision: ', precision_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred)))
print('recall: ', recall_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred)))


'''
flow = {'vect': vectorizer,
        'tfidf': tfidf,
        'clf': clf}
with open('/data/Date/flow_Date_month_xgb.pkl', 'wb') as fp:
    pickle.dump(flow, fp)
'''



