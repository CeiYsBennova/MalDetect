import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


# load train data
test_data = pd.read_csv('test_data.csv')

# remove id column
test_data.drop('Id', axis=1, inplace=True)

# convert to numpy array
test_data = test_data.values

X_test = test_data[:, 0].astype(str)
y_test = test_data[:, 1].astype(np.float64)


y_test = tf.one_hot(y_test, depth=9)

# parameters
vocab_size = 557
embedding_dim = 4
max_length = 30000

#load tokenizer
with open('tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)


X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')
model = load_model('model.h5')

# evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %f' % acc)
print('Test Loss: %f' % loss)

# predict and classify for 9 classes
y_pred = model.predict(X_test)
# print y_pred values compare to y_test values
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_pred)
print(y_test)
# print confusion matrix
print(confusion_matrix(y_test, y_pred))
