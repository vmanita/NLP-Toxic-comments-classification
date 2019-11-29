#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.preprocessing import MaxAbsScaler  
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV


# Setting a list with new features ++++++++++++++++++++++++++++++++++++++++++++

numeric_labels = ['char_count',
                      'word_count',
                      'word_density',
                      'punctuation_count',
                      'title_word_count',
                      'upper_case_word_count']#,
                      #'stop_count']#,
                      #'numeric_count']

                      
# Splitting the trainning data between train and validation +++++++++++++++++++
                      
train, valid, y_train, y_valid = train_test_split(data_sample[['comment_text']+numeric_labels],
                                                  data_sample['target'], test_size=0.2,
                                                  random_state=random_seed)  

train['comment_text'] = train['comment_text'].astype('str')
valid['comment_text'] = valid['comment_text'].astype('str')

# Vectorizing the text into matrixes ++++++++++++++++++++++++++++++++++++++++++
'''
We tried 4 different vectorizers and ended up only using the regular TF IDF matrix
All others are in comment
'''

# TF MATRIX --
#count_vec = CountVectorizer(min_df = 5, max_df = 0.7)
#x_train = count_vec.fit_transform(train['comment_text'])#.toarray()
#x_valid = count_vec.transform(valid['comment_text'])#.toarray()

# TF-IDF MATRIX --
tfidf_vec = TfidfVectorizer(min_df = 5, max_df = 0.7)
x_train = tfidf_vec.fit_transform(train['comment_text'])#.toarray()
x_valid = tfidf_vec.transform(valid['comment_text'])#.toarray()

# NGRAM TF-IDF --
#tfidf_vec_ngram = TfidfVectorizer(min_df = 5, max_df = 0.7, ngram_range=(2,3))
#x_train = tfidf_vec_ngram.fit_transform(train['comment_text'])#.toarray()
#x_valid = tfidf_vec_ngram.transform(valid['comment_text'])#.toarray()

# CHAR NGRAM TF-IDF --
#tfidf_vec_ngram_chars = TfidfVectorizer(min_df = 5, max_df = 0.7,analyzer='char',ngram_range=(2,3))
#x_train = tfidf_vec_ngram_chars.fit_transform(train['comment_text'])#.toarray()
#x_valid = tfidf_vec_ngram_chars.transform(valid['comment_text'])#.toarray()


# transform vectorizer into readble matrix ++++++++++++++++++++++++++++++++++++

train_matrix = train[numeric_labels].values
valid_matrix = valid[numeric_labels].values

# Combine vectorizer with other features ++++++++++++++++++++++++++++++++++++++

train_input = hstack((x_train, train_matrix))
valid_input = hstack((x_valid, valid_matrix))

# Scale Variables +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

scaler = MaxAbsScaler()
train_input=scaler.fit_transform(train_input)
valid_input=scaler.transform(valid_input)   
        

# Feature reduction +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
Here we reduce the dimensionality by selecting the best features to proceed with
the models.
We used a Recusive Feature Elimination technique from Sklearn.
Due to the high computational effort and long run-times, we decided not to do 
cross-validation at this point and exporting the results to an excel file to
eliminate the need to run this again
'''

'''
estimator = LogisticRegression()
selector = RFECV(estimator, step=10)
selector = selector.fit(train_input, y_train)



###creating rfecv from model results
rfecv=pd.DataFrame({ 'rank':selector.ranking_, 
                    'support':selector.support_})
rfecv.reset_index(drop=False, inplace=True)
rfecv.rename(columns={'index':'col_num'}, inplace=True)

# kepping track of wich olumns were selected
    #change path here#
    
rfecv.to_excel('C:/Users/PC/Documents/MAA/TM/tm proj/rfecv.xlsx')
'''


# Change path
path = '/Users/Manita/OneDrive - NOVAIMS/text_mining_shared/scripts FINAL/rfecv.xlsx'

# import result of rfecv if already known:
rfecv=pd.read_excel(path, usecols=[1,2,3])


# kepping only values that were selected
rfecv_true = rfecv[rfecv.support==True].drop(labels=['rank', 'support'], axis=1)


# removing non selected columns from dataset
idx=list(rfecv_true['col_num'])


# Saving the data in a dataframe withfeature selection ++++++++++++++++++++++++
train_input2=train_input[:,idx]
valid_input2=valid_input[:,idx]
train_input2.shape

