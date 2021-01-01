# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:07:51 2021

@author: Peilin Yang
"""

from keras.datasets import imdb
import numpy as np
from keras import models 
from keras import layers

# train label and test label are list of 0 and 1, neg or pos
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 0: padding, 1: start of sequence, 2: unknown
# if cannot find i, replace it with "?"
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# Pre-processing Data: standardize the matrix
def vectorize_sequences(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension)) 
    for i, sequence in enumerate(sequences): 
        results[i, sequence] = 1. 
    return results

x_train = vectorize_sequences(train_data) 
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32') 
y_test = np.asarray(test_labels).astype('float32')

# Define Network Model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:] 
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# X is txt, Y is 0/1
history = model.fit(partial_x_train, partial_y_train, epochs=20,batch_size=512, validation_data=(x_val, y_val))

# Loss Function

import matplotlib.pyplot as plt
history_dict = history.history 
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
plt.title('Training and validation loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend()

# Accuracy
plt.clf()
acc = history_dict['acc'] 
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend()
plt.show()

# A new model overcome overfitting
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512) 
results = model.evaluate(x_test, y_test)

# Predict the Model
model.predict(x_test)