import itertools
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, losses
import idx2numpy
from scripts import protected_functions as pf
from sklearn.metrics import classification_report

import pandas as pd

DATA_DIR = '/Users/ivanpetej/Projects/protected-calibration/data/'



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_train_dist = idx2numpy.convert_from_file(DATA_DIR + 'train-images-idx3-ubyte')
y_train_dist = idx2numpy.convert_from_file(DATA_DIR + 'train-labels-idx1-ubyte')

x_test_dist = idx2numpy.convert_from_file(DATA_DIR + 't10k-images-idx3-ubyte')
y_test_dist = idx2numpy.convert_from_file(DATA_DIR + 't10k-labels-idx1-ubyte')

x_test_dist = tf.pad(x_test_dist, [[0, 0], [2, 2], [2, 2]]) / 255
x_test_dist = tf.expand_dims(x_test_dist, axis=3, name=None)

x_val = x_train[-2000:, :, :, :]
y_val = y_train[-2000:]
x_train = x_train[:-2000, :, :, :]
y_train = y_train[:-2000]

model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 5, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

print(x_test.shape, y_test.shape)
print(x_test_dist.shape, y_test_dist.shape)


print('Fitting model')
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test)

test_dist_loss, test_dist_acc = model.evaluate(x_test_dist, y_test_dist)

p_pred = model.predict(x_test_dist)
p_pred_test = model.predict(x_test)
n_test = len(p_pred)

# log_sj_martingale, log_cj_martingale = pf.calc_martingale_multiclass(p_pred, y_test_dist, n_test, k=10, plot_charts=True)


p_prime = pf.calibrate_probs_multiclass(p_pred, y_test_dist, n_test, k=10)

print('Test set')
print(classification_report(y_test, np.argmax(p_pred_test, axis=1), labels=range(10)))
print('')
print('Distorted set - base classifier')
print(classification_report(y_test_dist, np.argmax(p_pred, axis=1), labels=range(10)))
print('')
print('Distorted set - protected classifier')
print(classification_report(y_test_dist, np.argmax(p_prime, axis=1), labels=range(10)))


