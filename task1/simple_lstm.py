#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, InputLayer
from keras.callbacks import EarlyStopping, ModelCheckpoint


path_in = 'sample/'
path_out = 'sample/'
data_dim = 1
batch_size = 1024
timesteps = 431
classes = 9


def lstm():
    model = Sequential()
    model.add(InputLayer((timesteps, data_dim)))
    model.add(LSTM(64, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=True, activation='tanh'))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))
    return model


train_labels = np.eye(classes)[np.load(path_in + 'train_labels.npy')]
train_data = np.load(path_in + 'train_data.npy')[:, :, None]
test_data = np.load(path_in + 'test_data.npy')[:, :, None]


model = lstm()
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
model.fit(train_data, train_labels, batch_size, epochs=10,
    callbacks=[EarlyStopping(), ModelCheckpoint('lstm.h5',
    save_best_only=True, save_weights_only=True)], validation_split=0.05)


test_labels = model.predict(train_data, batch_size)
test_labels = np.argmax(test_labels, axis=1)
np.save(path_out + 'test_labels_lstm.npy', test_labels)
