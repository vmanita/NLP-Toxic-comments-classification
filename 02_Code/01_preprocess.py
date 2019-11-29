#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:50:15 2019

@author: Manita
"""
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import copy
import itertools
import string
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
import re
import string
from nltk.tokenize import ToktokTokenizer
from tqdm import tqdm
from nltk.stem import SnowballStemmer
from collections import defaultdict
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
from matplotlib import pyplot as plt
import warnings

#warnings.filterwarnings('ignore')

# Import data Functions +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def import_data(path):
    data = pd.read_csv(path)
    data['target'] = data.max(numeric_only=True, axis = 1)
    return data

def sampling(data, sample_size, seed):
    #random.seed(seed)
    data_copy=copy.deepcopy(data['id'])
    idxs=random.sample(range(len(data)), sample_size)
    sample_idxs=[data_copy[i] for i in idxs]
    sample = data.loc[data['id'].isin(sample_idxs)]
    return sample

stemmer = SnowballStemmer('english')
stopwords_english = stopwords.words('english')
lemmatizer = nltk.WordNetLemmatizer()
toktok = ToktokTokenizer()

def clean_text(comment, remove_stopwords=True, stem_words=True, tok = ToktokTokenizer(), emoji = True, link = True):
    
    # tokenize    
    comment = tok.tokenize(comment)
    
    # lowercase and remove ponctuation
    comment = [comment.lower() for comment in comment]
    comment = [comment.translate(str.maketrans('', '', string.punctuation)) for comment in comment]
    
    # stopwords
    if remove_stopwords:
        comment = [word for word in comment if word not in stopwords_english]
    
    # stem
    if stem_words:
        comment = [stemmer.stem(word) for word in comment]
        
    # emojis
    if emoji:           
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
        comment = [emoji_pattern.sub(r'', word) for word in comment]
    
    # links
    if link:
        comment = [re.sub(r'(http)([\w]+)', '', comment) for comment in comment]
        comment = [comment for comment in comment if comment.isalnum()]
        
    comment = " ".join(comment) 
    return comment


random_seed = 0
random.seed(random_seed)
sample_size = 100000

# Import Training data: train.csv +++++++++++++++++++++++++++++++++++++++++++++

data = import_data('/Users/Manita/OneDrive - NOVAIMS/Text Mining/jigsaw-toxic-comment-classification-challenge/train.csv')

# Plotting the distribution of clean vs toxic comments ++++++++++++++++++++++++

tmp = data.copy()
d = {'target':{1:'Toxic',0:'Clean'}}
tmp.replace(d, inplace=True)
plt.figure(figsize = (10,10)) 
sns.countplot(x="target", data=tmp)
plt.title('Clean vs Toxic comments', fontweight = 'bold')
plt.show()


# Plotting the distribution of toxicity levels in all comments ++++++++++++++++

plot = data.copy()
plot = pd.melt(plot, id_vars="id", var_name="toxicity_levels",
               value_vars = ['toxic', 'severe_toxic', 'obscene',
                             'threat', 'insult', 'identity_hate'], value_name="target")
    
plot = plot.groupby('toxicity_levels')['target'].apply(lambda x: (x==1).sum()).reset_index(name='count')
plt.figure(figsize = (10,10)) 
p = sns.factorplot(x = 'toxicity_levels', y = 'count',
                   data = plot, kind = 'bar', order=["toxic", "obscene", "insult",
                                                     "severe_toxic","identity_hate","threat"])
plt.xticks(rotation=80)
plt.show()


# Balance Dataset +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
Here we are retrieving all ofensive commentaries and sampling an equal number
of clean ones
We tried running our classifier with the unbalanced dataset but the results were 
not good.
''' 

toxic_size = len(data.loc[data.target == 1])
sample_toxic = data.loc[data.target == 1]
sample_not_toxic = sampling(data.loc[data.target == 0].reset_index(drop=True), toxic_size, random_seed)

data_sample = pd.concat([sample_toxic,sample_not_toxic],axis=0)

# Creating new features +++++++++++++++++++++++++++++++++++++++++++++++++++++++

data_sample['char_count'] = data_sample['comment_text'].apply(len)
data_sample['word_count'] = data_sample['comment_text'].apply(lambda x: len(x.split()))
data_sample['word_density'] = data_sample['char_count'] / (data_sample['word_count']+1)
data_sample['punctuation_count'] = data_sample['comment_text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
data_sample['title_word_count'] = data_sample['comment_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
data_sample['upper_case_word_count'] = data_sample['comment_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

# These features in comment did not add anything to the model prediction ability
#data_sample['stop_count'] = data_sample['comment_text'].apply(lambda x: len([x for x in x.split() if x in stopwords_english]))
#data_sample['numeric_count'] = data_sample['comment_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

# Applying the cleaning data functions ++++++++++++++++++++++++++++++++++++++++

comments = [clean_text(comment) for comment in tqdm(data_sample['comment_text'])]

data_sample['comment_text'] = comments





















