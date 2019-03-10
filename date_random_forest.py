
import pandas as pd
import numpy as np
import string
from fastparquet import ParquetFile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


#clean data - create corpus


def hood(text):
    sentence = text.split()
    indices = (i for i,word in enumerate(sentence) if '201' in word)
    neighbors = []
    for ind in indices:
        if ind < 3:
            strt = 0
        else:
            strt = ind-3
        neighbors.append(' '.join(sentence[strt:ind]) +' '+ ' '.join([sentence[ind]]) + ' ' + ' '.join(sentence[ind+1:ind+4]))
    return neighbors



def clean_data(X):
    my_punc = list(string.punctuation)
    #my_punc.remove('-')
    #my_punc.remove('/')
    my_punc = ''.join(my_punc)
    translator = str.maketrans(my_punc, ' '*len(my_punc))

    corpus=[]
    for row in X.itertuples():
        text=row.Text
        if text != pd.isnull(text):
            text = str(text).translate(translator)
            #text = ' '.join(word for word in text.split() if len(word) < 10) # remove words longer then the word 'september'
            #text = ' '.join(word for word in text.split() if not word.isdigit() or int(word) < 32) # remove numbers bigger than 31
            #text = ' '.join([item for item in text.split() if item.isdigit() or not any(filter(str.isdigit, item))]) # filter words with letters & numbers --- that not good 08Aug
            #text = text.replace('\n',' ')
            text = text.replace(' L_B ', ' ')  # in Parquet '\n' == ' L_B '
            text = text.replace(' comma ', ',')  # in Parquet ',' == ' comma '
            text = hood(text)
            corpus.append(' '.join(text))
    return(corpus)

#######################################################################################

#read data


df = ParquetFile('/run/user/1000/gvfs/smb-share:server=nas01.local,share=rnd/data/parquet_data/date_alg_results_and_ocr.parq').to_pandas()
#df = df.loc[(df['conclusion'] == 'N\A') & (df['post_match_date'] != 'N\A')]
df = df.loc[df['post_match_date'] != 'N\A']

print('data loaded')

df['day'] = df['post_match_date'].str.slice(0,2)
df['month'] = df['post_match_date'].str.slice(3,5)
df['year'] = df['post_match_date'].str.slice(6,10)


df.loc[:, 'hood'] = clean_data(pd.DataFrame(df.Text))
df = df[df['hood'] != '']


print('invoices to train: ', len(df))

#split data
y = df['day'].astype('uint8') ############################# date part
X = df[['imaginary_id','conclusion','Text']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print('data split')

#X_train = clean_data(X_train)

print('data cleaned')

vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True, lowercase=True, max_features=200) ########### MODEL
vectorizer.fit(X_train.Text)
word_matrix = vectorizer.transform(X_train.Text)

print('vectorizer done')

tfidf = TfidfTransformer(norm='l2') ############# MODEL
tfidf.fit(word_matrix)
tfidf_matrix = tfidf.transform(word_matrix)

print('tfidf done')

#X_test = clean_data(X_test)
test_word_matrix = vectorizer.transform(X_test.Text)
test_tfidf_matrix = tfidf.transform(test_word_matrix)


clf = RandomForestClassifier(n_estimators = 250, criterion = 'entropy', random_state = 42,n_jobs=-1) ############ MODEL
clf.fit(tfidf_matrix, y_train)

print('Random Forest done')



y_pred = clf.predict(test_tfidf_matrix)
conc_mat=pd.DataFrame({'imaginary_id':X_test.imaginary_id,'alg_pred':y_pred,'True':y_test, 'omri_alg': X_test.conclusion})

print ('RF:')
print ('----------------------')
print ('\tAccuracy: %1.3f' % accuracy_score(y_test, y_pred, normalize=True))
print ('\tPrecision: %1.3f' % precision_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred)))
print ('\tRecall: %1.3f' % recall_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred)))


"""
df1 = pd.DataFrame({'x_test': X_test, 'y_test': y_test, 'y_pred': y_pred})
df1.loc[:, 'is_correct'] = (df1.y_test == df1.y_pred)
df1.groupby(['y_test','is_correct']).size().unstack()
"""