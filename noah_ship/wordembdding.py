import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense,Embedding, Input, Concatenate

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load train data
df = pd.read_csv('train_data.csv')

# remove id column
df.drop('Id', axis=1, inplace=True)

# convert to numpy array
data = df.values

# split into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# split into X_train, y_train, X_test, y_test
X_train = train_data[:, 0].astype(str)
y_train = train_data[:, 1].astype(np.float64)

X_test = test_data[:, 0].astype(str)
y_test = test_data[:, 1].astype(np.float64)

# one hot encode y_train and y_test
y_train = tf.one_hot(y_train, depth=9)
y_test = tf.one_hot(y_test, depth=9)

# parameters
vocab_size = 557
embedding_dim = 4
max_length = 30000

#  tokenize and pad sequences with chunks size is 50
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

# standardize X_train and X_test
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# save tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

input_layer = Input(shape=(max_length,))
# embedding layer with chunks size is 50
embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_layer)

# 3 parallel conv1d layers and maxpooling layers
conv1 = Conv1D(100, 3, activation='relu')(embedding_layer)
pool1 = GlobalMaxPooling1D()(conv1)

conv2 = Conv1D(100, 5, activation='relu')(embedding_layer)
pool2 = GlobalMaxPooling1D()(conv2)

conv3 = Conv1D(100, 7, activation='relu')(embedding_layer)
pool3 = GlobalMaxPooling1D()(conv3)

# concatenate the 3 parallel layers
concat = Concatenate()([pool1, pool2, pool3])


# dense layer
dense = Dense(9, activation='relu')(concat)

# output layer
output_layer = Dense(9, activation='sigmoid')(dense)

# build model
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=25, batch_size=10, validation_data=(X_test, y_test))

# save model
model.save('model.h5')

# evaluate model
loss, acc = model.evaluate(X_test, y_test)

print("Loss: ", loss)
print("Accuracy: ", acc)




