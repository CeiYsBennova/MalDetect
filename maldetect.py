import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Flatten, Embedding, Input, Concatenate
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from tensorflow.keras.models import load_model



df = pd.read_csv('train_datav2.csv')

# remove id column
df.drop('Id', axis=1, inplace=True)

# convert to numpy array
data = df.values

# split into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# split into X_train, y_train, X_test, y_test
X_train = train_data[:, 0].astype(str)
y_train = train_data[:, 1]

X_test = test_data[:, 0].astype(str)
y_test = test_data[:, 1]

y_train = y_train.astype(np.float64).reshape((-1, 1))
y_test = y_test.astype(np.float64).reshape((-1, 1))



# parameters
vocab_size = 557    
embedding_dim = 4
max_length = 30000

'''
# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
'''
# tokenize and pad sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

# standardize X_train and X_test
#X_train = tf.keras.utils.normalize(X_train, axis=1)
#X_test = tf.keras.utils.normalize(X_test, axis=1)

#standardscaler = StandardScaler()
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# model summary be like:
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None,N,1)                   0   
# _________________________________________________________________
# embedding (Embedding)        (None,N,4)                   V*4
# _________________________________________________________________
# conv1d (Conv1D)              (None,N,100)                 100*3*4
# _________________________________________________________________
# global_max_pooling1d (Global (None,100)                   0
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None,N,100)                 100*5*4
# _________________________________________________________________
# global_max_pooling1d_1 (Glob (None,100)                   0
# _________________________________________________________________
# conv1d_2 (Conv1D)            (None,N,100)                 100*7*4
# _________________________________________________________________
# global_max_pooling1d_2 (Glob (None,100)                   0
# _________________________________________________________________
# feature_concat (Concatenate) (None,300)                   0
# _________________________________________________________________
# dense_1 (Dense)              (None,9)                     300*9
# softmax (Softmax)            (None,9)                     0
# =================================================================
# Total params:

# build model
input_layer = Input(shape=(max_length, ))
embedding_layer = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_layer)

conv1 = Conv1D(100, 3, activation='relu')(embedding_layer)
pool1 = GlobalMaxPooling1D()(conv1)

conv2 = Conv1D(100, 5, activation='relu')(embedding_layer)
pool2 = GlobalMaxPooling1D()(conv2)

conv3 = Conv1D(100, 7, activation='relu')(embedding_layer)
pool3 = GlobalMaxPooling1D()(conv3)

concat = Concatenate()([pool1, pool2, pool3])

dense = Dense(64, activation='relu')(concat)
output_layer = Dense(1, activation='sigmoid')(dense)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#create checkpoint
filepath = "checkpointv2.hdf5"
callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# fit model
model.fit(X_train, y_train, epochs=30, batch_size=10, validation_data=(X_test, y_test), callbacks=[callback])

# save model
model.save('model_newv2.h5')

# save tokenizer
with open('tokenizer_newv2.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#acurracy
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred, average='weighted'))
print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall: ', recall_score(y_test, y_pred, average='weighted'))


# load test data
test_data = pd.read_csv('test_datav2.csv')

# remove id column
test_data.drop('Id', axis=1, inplace=True)

# convert to numpy array
test_data = test_data.values

X_test = test_data[:, 0].astype(str)
y_test = test_data[:, 1].astype(np.float64).reshape((-1, 1))


X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

model = load_model('model_newv2.h5')


# predict and classify for 9 classes
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(y_pred)
print(y_test)

print(confusion_matrix(y_test, y_pred))

# print classification report
print(classification_report(y_test, y_pred))

# print accuracy score
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred, average='weighted'))
print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall: ', recall_score(y_test, y_pred, average='weighted'))



