import pandas as pd
import matplotlib.pyplot as plt
import re
import json
import ast
import pickle
import numpy as np
import matplotlib.patches as mpatches
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF#, LatentDirichletAllocation
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer#, SnowballStemmer
import api_module
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
from api_module import utils 



def pickle_object(obj, name):
    '''
    pickle any object in python to store it

    Input:
        obj: Python object to be pickled
        name: resulting name of pickle file. NOTE: No need to add .pkl in the name
    Output:
        None. Function writes straight to disk.
    '''
    with open(name+".pkl", 'wb') as f:
        pickle.dump(obj, f, protocol=4)



def unpickle_object(pkl):
    '''
    fucntion to unpickle any object in python
    Input:
        pickle object from disk.
    Output:
        unpickled file now available in python namespace
    '''
    return pickle.load(open(pkl, 'rb'))


def unique_pairs(df):
    '''
    Return lower triangle of a correlation matrix
    Input:
        df - Pandas DataFrame
    Output:
        drop_cols - columns for lower triangle in correlation matrix
    '''
    drop_cols = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            drop_cols.add((cols[i], cols[j]))
    return drop_cols



def extract_top_correlations(df, num_values):
    '''
    Get the top correlated columns in the dataframe
    Input:
        df - Pandas DataFrame
        num_values - number of top columns to display
    Output:
        corr_matrix[0:num_values] - correlation matrix for top columns
    '''
    corr_matrix = df.corr().abs().unstack()
    unique_cols = unique_pairs(df)
    corr_matrix = corr_matrix.drop(labels = unique_cols).sort_values(ascending = False)
    return corr_matrix[0:num_values]



def tokenizing_function_en(document):
    """
    Create tokens from each sentence (splits words)
    Input:
        document - the text in the dataframe
    Output:
        tokenized - tokenized words list from document
    """
    stemmer = PorterStemmer()
    tokenized = [stemmer.stem(word) for sentence in sent_tokenize(document) for word in word_tokenize(sentence) if len(word) > 1]
    return tokenized


def eng_clean(document):
    '''
    Regular expression quick cleaning function to
    remove everything that isn't in the english alphabet and urls
    Input:
        document - the text in the dataframe
    Output:
        text - the cleaned document
    '''
    text = re.sub(r'http\S+', '', document, flags=re.MULTILINE) # remove urls
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = ' '.join([i for i in text.split()])
    text = re.sub('[^a-zA-Z]+', ' ', document)
    text = re.sub(r'\b\w{1,3}\b', '', text)
    for sign in ['\n', '\x0c']:
        text = text.replace(sign, ' ')
    return text.lower().strip()



def preprocess(df, text_col, clean_col, cleaning_func): #default right now is set to eng_clean
    '''
    preprocess data to return cleaned matrix to apply tf-idf function to.
    Input:
        PATH - path to datafile
    Output:
        df - Pandas DataFrame
    '''
    #df = pd.read_csv(PATH)

    df.fillna('None Present', inplace=True)
    df[clean_col] = df[text_col].apply(lambda x : cleaning_func(x))
    return df


def create_tfidf(df, column_name, num_features, tokenizing_function, ngram=1, stop_words='english'):
    '''
    Create matrix with word weightings for a specified number of words in each document,
    uses counts normalized by counts of documents containing the word.
    This matrix will be used for clustering the documents.
    Input:
        df - Pandas DataFrame
        column_name - columns containing corpus to be transformed
        num_features - (Int) number of words to be used from the corpus for clustering documents
    Output:
        tfidf_vectorizer - vectorizer
        tfidf_sparse - sparse matrix of vectorized words
    '''
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,ngram),
                                       stop_words=stop_words,
                                       max_df=0.8,
                                       min_df=2,
                                       max_features=num_features,
                                       sublinear_tf = True,
                                       tokenizer=tokenizing_function)
    tfidf_sparse = tfidf_vectorizer.fit_transform(df[column_name])
    return tfidf_vectorizer, tfidf_sparse


def create_cv(df, column_name, num_features, ngram=1):
    '''
    Create matrix with word matrix for a specified number of words in each document,
    uses the counts per document.
    This matrix will be used for clustering the documents.
    Input:
        df - Pandas DataFrame
        column_name - columns containing corpus to be transformed
        num_features - (Int) number of words to be used from the corpus for clustering documents
    Output:
        count_vectorizer - vectorizer
        count_sparse - sparse matrix of vectorized words
    '''
    count_vectorizer = CountVectorizer(ngram_range=(1, ngram),
                                       stop_words = 'english',
                                       max_df = 0.7,
                                       min_df = 2,
                                       max_features = num_features,
                                       tokenizer = tokenizing_function)
    count_sparse = count_vectorizer.fit_transform(df[column_name])
    return count_vectorizer, count_sparse


def create_scaled_matrix(matrix):
    '''
    Scales matrix using StandardScaler
    Input:
        matrix - Numpy Array to scale
    Output:
        matrix - scaled Numpy Array
    '''
    scaler = StandardScaler()
    return scaler.fit_transform(matrix)


def getLinearColor(num_classes):    
    plt.style.use('fivethirtyeight')
    result = plt.cm.nipy_spectral(np.linspace(0, 1, num_classes))
    color = []
    for row in result:
        ansRow = 0
        for cell in row:
            ansRow <<= 8
            ansRow += int(cell * 255)
        color.append(ansRow)
    
    return color


def plot_json(clf_df):
    '''
    Creates a 2D plot of the data
    Input:
        clf_df - Pandas DataFrame        
    Output:
        Json Object - per row
    '''
    data = clf_df
    json_data = []
    for row in data.values:
        json_data.append({'x':row[0], 'y':row[1], 'c':row[2], 'text':row[3]})
        # json_data.append({'x':row[0], 'y':row[1], 'c':row[2]})
    
    return json_data


def data_toJson(data):
    '''
    Creates a 2D plot json
    Input:
        data - Pandas DataFrame        
    Output:
        Json Object - {
            'tbheader' :[{'colname':'no', 'name':'#'},{'colname':'class1', 'name':'Class 1'}, ...],
            'tbdata' :[{'no': 1, 'class1':'...', 'class1':'...'}, ...]
        }
    '''    
    json_data = {}
    rownum = 0
    tbdata = []
    tbdispCol = ["no"]
    tbheader = [{'colname':'no', 'name':'#'}]
    for row in data:
        if rownum == 0:
            cellnum = 0
            for cell in row:                                            
                cellIdxStr = str(cellnum + 1)
                tbheader.append({'colname':'class' + cellIdxStr, 'name':'Class ' + cellIdxStr})
                cellnum = cellnum + 1            
                tbdispCol.append('class' + cellIdxStr)
            json_data['tbheader'] = tbheader
        
        tbrow = {'no':str(rownum + 1)}
        cellnum = 0
        for cell in row:            
            if not cell:
                cell = ""

            tbrow['class' + str(cellnum + 1)] = cell
            cellnum = cellnum + 1
        tbdata.append(tbrow)
        rownum = rownum + 1
    json_data['tbdata'] = tbdata
    json_data['tbdispCol'] = tbdispCol
    return json_data

def getModelPath(filename):
    return api_module.MODELS_FOLDER + filename
    