#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:47:33 2019

@author: Manita
"""
from tqdm import tqdm
#if we wan to pass new sentences to apply the model 

sentences = ["hey you're stupid"]

sentences = pd.DataFrame({'comment_text':sentences})

sentences['char_count'] = sentences['comment_text'].apply(len)
sentences['word_count'] = sentences['comment_text'].apply(lambda x: len(x.split()))
sentences['word_density'] = sentences['char_count'] / (sentences['word_count']+1)
sentences['punctuation_count'] = sentences['comment_text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
sentences['title_word_count'] = sentences['comment_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
sentences['upper_case_word_count'] = sentences['comment_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

# clean
to_predict = [clean_text(sentence) for sentence in tqdm(sentences['comment_text'])]

# Vectorize
matrix = tfidf_vec.transform(to_predict)
# Combine
input_ = hstack((matrix, sentences[numeric_labels].values))

#scale
input_=scaler.transform(input_)


#feature selection
input_=input_[:,idx]

# predict
predictions = model.predict(input_)
print(predictions)

# Return Dataframe with toxicity
result = pd.DataFrame({'Comment':sentences.comment_text,
                       'Classification':['Toxic' if prediction == 1 else 'Clean' for prediction in predictions]})

print(result.Classification)


















