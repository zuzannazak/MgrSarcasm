import json
import tensorflow as tf
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

#Sources:
#  https://towardsdatascience.com/tensorflow-sarcasm-detection-in-20-mins-b549311b9e91

sentences = []
labels = []

with open('data-test-balanced.csv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        labels.append(row[0])
        sentences.append(row[1])

#print(pd.DataFrame({'sentence' : sentences[0:10], 'label':labels[0:10]}))


# Splitting the dataset into Train and Test
training_size = round(len(sentences) * .75)
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
#
# Setting tokenizer properties
vocab_size = 50000
oov_tok = "<oov>"

# Fit the tokenizer on Training data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Setting the padding properties
max_length = 100
trunc_type='post'
padding_type='post'
#
# Creating padded sequences from train and test data
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
#
# Setting the model parameters
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# Converting the lists to numpy arrays for Tensorflow 2.x
training_padded = np.array(training_padded)
training_labels = np.array(training_labels, dtype='int32')
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels, dtype='int32')

print(training_padded.dtype)

# # Training the model
num_epochs = 10
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)


sentence = ["Yes, definitely",
            "I like cats."]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))