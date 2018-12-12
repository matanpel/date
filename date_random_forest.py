
import pandas as pd
import numpy as np
import string
import re
import glob
import pickle
from fastparquet import ParquetFile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


#clean data - create corpus
def clean_data(X):
    my_punc=list(string.punctuation)
    my_punc.remove('#')
    my_punc=''.join(my_punc)
    translator = str.maketrans('', '', my_punc)
    
    corpus=[]
    for row in X.itertuples():
        text=row.Text
        if text != pd.isnull(text):
            text = text.replace('/',' ')
            text = text.replace('-',' ')

            text = str(text).translate(translator)
            #text = text.replace('\n',' ')
            text = text.replace(' L_B ',' ')  # in Parquet '\n' == ' L_B '
            text = text.replace(' comma ',',') # in Parquet ',' == ' comma '
            #text=re.sub(r'\d+', '', text)  #remove numbers  ### do we want to? might hold information
            corpus.append(text)
    return(corpus)

#######################################################################################

#read data

#df = pd.read_csv('Y:/data/suppliers/suppliers_texts.csv')
dates = ParquetFile('Y:/data/parquet_data/rails_data.parq').to_pandas(columns=['imaginary_id','invoice_date']).dropna()
dates['day'] = dates['invoice_date'].str.slice(0,2)
dates['month'] = dates['invoice_date'].str.slice(3,5)
dates['year'] = dates['invoice_date'].str.slice(6,10)

dates = dates[(dates['month'].isin(['01','02','03','04','05','06'])) & (dates['year']=='2018')]

print('dates loaded')

#text = ParquetFile('Y:/data/parquet_data/text_extractions_barmoach.parq').to_pandas() ## Parquet file not loading
path = 'C:/Users/User/DS/parquet/ocr_text_csv/ocr_text*.csv'
text = pd.concat([pd.read_csv(f, encoding = "ISO-8859-1", usecols=['#imaginary_id','text'], dtype={'#imaginary_id': str, 'text': str}) for f in glob.glob(path)], ignore_index = True)
text.columns = ['imaginary_id', 'Text']


print('text extractions loaded')

df = pd.merge(text,dates, on='imaginary_id')

print('data merged')
print('invoices to train: ', len(df))

#split data
y = df['month'] ############################# date part
X = df[['imaginary_id','Text']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print('data split')

X_train = clean_data(X_train)

print('data cleaned')

vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True, lowercase=True, max_features=5000) ########### MODEL
vectorizer.fit(X_train)

print('vectorizer done')

word_matrix = vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm='l2') ############# MODEL
tfidf.fit(word_matrix)

print('tfidf done')

tfidf_matrix = tfidf.transform(word_matrix)

X_test=clean_data(X_test)
test_word_matrix = vectorizer.transform(X_test)
test_tfidf_matrix = tfidf.transform(test_word_matrix)


clf = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 42,n_jobs=-1) ############ MODEL
clf.fit(tfidf_matrix, y_train)

print('Random Forest done')


## random forest:
y_pred = clf.predict(test_tfidf_matrix)
X_test, y_test, y_pred #### make DataFrame and analyze ###############################################

print ('RF:')
print ('----------------------')
print ('\tAccuracy: %1.3f' % accuracy_score(y_test, y_pred, normalize=True))
print ('\tPrecision: %1.3f' % precision_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred)))
print ('\tRecall: %1.3f' % recall_score(y_test, y_pred, average='weighted',labels=np.unique(y_pred)))


