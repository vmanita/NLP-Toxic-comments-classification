#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 02:46:13 2019

@author: Manita
"""


# Testing Data ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
We will now import the testing data and test the performance of our final and
tuned model
'''
# Import Test data

test_data = pd.read_csv('/Users/Manita/OneDrive - NOVAIMS/Text Mining/jigsaw-toxic-comment-classification-challenge/test.csv')
y_test = pd.read_csv('/Users/Manita/OneDrive - NOVAIMS/Text Mining/jigsaw-toxic-comment-classification-challenge/test_labels.csv')

# Define target
y_test['target'] = y_test.max(numeric_only=True, axis = 1)
y_test = y_test.loc[y_test.target >= 0]

# Filter rows (remove -1 labels)
idx_to_keep = y_test.id
test_data = test_data.loc[test_data['id'].isin(idx_to_keep)]

# Add target
test_data = test_data.merge(y_test,left_on='id',right_on='id')[['id','comment_text','target']]

# Get sample of data to predict
# Given the high ammount of data in the testing set, we will sample it for 10 000 observations

random_seed = 0
sample_size = 10000
sample_test_data = sampling(test_data, sample_size, random_seed)

# new variables

sample_test_data['char_count'] = sample_test_data['comment_text'].apply(len)
sample_test_data['word_count'] = sample_test_data['comment_text'].apply(lambda x: len(x.split()))
sample_test_data['word_density'] = sample_test_data['char_count'] / (sample_test_data['word_count']+1)
sample_test_data['punctuation_count'] = sample_test_data['comment_text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
sample_test_data['title_word_count'] = sample_test_data['comment_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
sample_test_data['upper_case_word_count'] = sample_test_data['comment_text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

# Clean Data
new_comments = [clean_text(comment) for comment in tqdm(sample_test_data['comment_text'])]

sample_test_data['comment_text'] = new_comments

# Vectorize
test = tfidf_vec.transform(sample_test_data['comment_text'])
x_ = sample_test_data[numeric_labels].values
# combine
test = hstack((test, x_))

# scale
test=scaler.transform(test)

# if using feature selection apply it:
test=test[:,idx]

# Test Model ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

x_test = test.copy()
y_test = sample_test_data['target']


# load the model
toxic_model = pickle.load(open(path + filename, 'rb'))
#result = toxic_model.score(x_test, y_test)

test_predict = toxic_model.predict(x_test)
print(classification_report(y_test,test_predict)) 
print('>>> Accuracy:',accuracy_score(test_predict, y_test),'<<<')


def confusion_mx(y, y_predict, labels = [0,1]):
    cm = confusion_matrix(y, y_predict, labels)
    plt.figure(figsize=(10,10))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax,fmt='g')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels([0, 1])
    ax.yaxis.set_ticklabels([0, 1])
    plt.show()
    

confusion_mx(y_test,test_predict)

from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

# AUROC CURVE

y_scores = toxic_model.predict_proba(x_test)[:, 1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

def plot_roc_auc(roc_auc):
    plt.figure(figsize=(15,10))
    plt.title('AUROC Curve', fontweight = 'bold', fontsize = 20)
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc, linewidth=3)
    plt.legend(loc = 'lower right',fontsize = 20)
    plt.plot([0, 1], [0, 1],linestyle='--', linewidth=3)
    plt.axis('tight')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('True Positive Rate', fontsize = 15)
    plt.xlabel('False Positive Rate', fontsize = 15)
    plt.show()

plot_roc_auc(roc_auc)


# PR CURVE

from sklearn.utils.fixes import signature

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_scores)
average_precision

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure(figsize=(10,7))
plt.step(recall, precision, color='black', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='black', **step_kwargs)

plt.xlabel('Recall', fontsize = 15)
plt.ylabel('Precision', fontsize = 15)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision), fontweight = 'bold', fontsize = 18)
plt.show()











