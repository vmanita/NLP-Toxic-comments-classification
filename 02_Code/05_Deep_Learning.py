#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:24:35 2019

@author: Manita
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    
    plt.figure(figsize = (12,5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label = 'Trainning acc')
    plt.plot(x, val_acc, 'r', label = 'Validation acc')
    plt.title('Trainning and Validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label = 'Trainning loss')
    plt.plot(x, val_loss, 'r', label = 'Validation loss')
    plt.title('Trainning and Validation loss')
    plt.legend()
    

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

'''
#------------------------------------------------------------------------------
# Com este modelo 2 iteracoes bastam 0.89  acc
model = Sequential()
model.add(layers.Dense(10, input_dim = input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.summary()

history = model.fit(x_train, y_train,
                    epochs = 10,
                    validation_data = (x_valid, y_valid),
                    batch_size=32)

loss, accuracy = model.evaluate(x_train, y_train, verbose = False)
print("Trainning Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_valid, y_valid, verbose = False)

print("Testing Accuracy: {:.4f}".format(accuracy))
plot_history(history)
 
predictions = model.predict(x_valid)
y_pred = np.round(predictions)
print(metrics.confusion_matrix(y_valid, y_pred))
print(classification_report(y_valid,y_pred)) 
print('>>> Accuracy:',accuracy_score(y_pred, y_valid),'<<<')
#------------------------------------------------------------------------------
'''


model_dp = Sequential()
model_dp.add(Dense(512, input_shape=(train_input2.shape[1],)))
model_dp.add(Activation('relu'))
model_dp.add(Dropout(0.5))
model_dp.add(layers.Dense(1, activation='sigmoid'))
model_dp.summary()
    
model_dp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 1

history = model_dp.fit(train_input2, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data = (valid_input2, y_valid))
    
loss, accuracy = model_dp.evaluate(train_input2, y_train, verbose = False)
print("Trainning Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model_dp.evaluate(valid_input2, y_valid, verbose = False)
print("Validation Accuracy: {:.4f}".format(accuracy))

history.history

plot_history(history)
 
predictions_dp = model_dp.predict(valid_input2)
y_pred = np.round(predictions_dp)
print(metrics.confusion_matrix(valid_input2, y_pred))
print(classification_report(valid_input2,y_pred)) 
print('>>> Accuracy:', accuracy_score(y_pred, y_valid),'<<<')   

    
    
    
    
    
    
    
    