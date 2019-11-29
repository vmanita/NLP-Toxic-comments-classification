#!/usr/bin/env python3
# -*- coding: utf-8 -*-

models = [LogisticRegression(random_state = random_seed, max_iter=5000,solver ='liblinear'),
          MultinomialNB(),
          RandomForestClassifier(random_state = random_seed),
          SGDClassifier(loss="hinge", penalty="l2", max_iter=5, random_state = random_seed),
          MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=random_seed),
          DecisionTreeClassifier(random_state = random_seed),
          XGBClassifier(random_state = random_seed)]
 

# Plotting the different models +++++++++++++++++++++++++++++++++++++++++++++++
'''
We have a list 'models' containing different models.
We will proceed to run the classification of data with and without feature selection
and choose the one that returns best results
Cross-validation was also used to reduce the bias of running with the same seed
and providing better results and a better generalization ability.
'''

# Without feature selection  ++++++++++++++++++++++++++++++++++++++++++++++++++
    
splits=10         
CV = KFold(n_splits=splits, random_state=random_seed)
cv_df = pd.DataFrame(index=range(splits * len(models)))

entries = []
for model in tqdm(models):
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, train_input, y_train, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# Plot the results

my_pal = {model_name: "r" if model_name == "LogisticRegression" else "grey" for model_name
          in cv_df.model_name.unique()}

plt.figure(figsize = (12,5))
sns.boxplot(x='model_name', y='accuracy', data=cv_df,boxprops=dict(alpha=.8),
            linewidth=2, palette=my_pal)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=6, jitter=True, edgecolor="black", linewidth=1, palette=my_pal)
plt.xlabel('Model Name')
plt.show()


print(np.round(
        cv_df.groupby('model_name').agg('mean').
        sort_values(by=['accuracy'], 
                    ascending=False)['accuracy'],decimals=4))

print(np.round(
        cv_df.groupby('model_name').agg('std').
        sort_values(by=['accuracy'], 
                    ascending=True)['accuracy'],decimals=4))



# Selecting the best model according to the accuracy
# train the best model

model1 = linear_model.LogisticRegression(random_state = random_seed, max_iter=5000,solver ='liblinear')

model1.fit(train_input, y_train)
predictions = model1.predict(valid_input)
print(confusion_matrix(y_valid,predictions))  
print(classification_report(y_valid,predictions)) 
print('>>> Accuracy:',accuracy_score(predictions, y_valid),'<<<')



# With feature selection  +++++++++++++++++++++++++++++++++++++++++++++++++++++

splits=10         
CV = KFold(n_splits=splits, random_state=random_seed)
cv_df2 = pd.DataFrame(index=range(splits * len(models)))

entries = []
for model in tqdm(models):
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, train_input2, y_train, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df2 = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# Plot the results

my_pal = {model_name: "r" if model_name == "LogisticRegression" else "grey" for model_name
          in cv_df2.model_name.unique()}


plt.figure(figsize = (12,5))
sns.boxplot(x='model_name', y='accuracy', data=cv_df2,boxprops=dict(alpha=.8), 
            linewidth=2,  palette=my_pal)

sns.stripplot(x='model_name', y='accuracy', data=cv_df2, 
              size=6, jitter=True, edgecolor="black", linewidth=1)
plt.xlabel('Model Name')
plt.show()


print(np.round(
        cv_df2.groupby('model_name').agg('mean').
        sort_values(by=['accuracy'], 
                    ascending=False)['accuracy'],decimals=4))

print(np.round(
        cv_df2.groupby('model_name').agg('std').
        sort_values(by=['accuracy'], 
                    ascending=True)['accuracy'],decimals=4))


# Selecting the best model according to the accuracy
# train the best model

model2 =  SGDClassifier(loss="hinge", penalty="l2", max_iter=5, random_state = random_seed)

model2.fit(train_input2, y_train)
predictions2 = model2.predict(valid_input2)
print(confusion_matrix(y_valid,predictions2))  
print(classification_report(y_valid,predictions2)) 
print('>>> Accuracy:',accuracy_score(predictions2, y_valid),'<<<')

'''
We noticed that running the model with feature selection provided better results,
besides the faster running-times and smaller computational effort.
We proceeded with the model and dataset with feature selection applied
'''

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# final model with feature selection
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

model =  SGDClassifier(loss="hinge", penalty="l2", max_iter=5, random_state = random_seed)

model.fit(train_input2, y_train)
predictions2 = model.predict(valid_input2)
print(confusion_matrix(y_valid,predictions2))  
print(classification_report(y_valid,predictions2)) 
print('>>> Accuracy:',accuracy_score(predictions2, y_valid),'<<<')

# Tune parameters with grid search
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

grid = {
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
    'n_iter': [1000], # number of epochs
    'loss': ['log'], # logistic regression,
    'penalty': ['l2'],
    'n_jobs': [-1]
}


sgd = SGDClassifier(random_state = random_seed)
clf = GridSearchCV(sgd, grid, cv=2)

clf.fit(train_input2, y_train)

clf.best_params_

predictions2 = clf.best_estimator_.predict(valid_input2)
print(confusion_matrix(y_valid,predictions2))  
print(classification_report(y_valid,predictions2)) 
print('>>> Accuracy:',accuracy_score(predictions2, y_valid),'<<<')


# Save model and export it ++++++++++++++++++++++++++++++++++++++++++++++++++++


import pickle

# save the model
filename = 'Toxic_Classifier.sav'
path = '/Users/Manita/OneDrive - NOVAIMS/text_mining_shared/scripts FINAL/'
pickle.dump(clf.best_estimator_, open(path + filename, 'wb'))


# load the model
toxic_model = pickle.load(open(path + filename, 'rb'))
result = toxic_model.score(valid_input2, y_valid)
print(result)








