import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import numpy as np

from scripts import protected_functions as pf

# testing with standard MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

selected_labels = [1, 7]

x_train, y_train, x_test, y_test = pf.make_binary(x_train, y_train, x_test, y_test, selected_labels, 1, 3)

x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]])/255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]])/255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

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
model.summary()

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_val, y_val))

model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)[:, selected_labels]
p_pred = y_pred[:, 1]

y_test[y_test == selected_labels[0]] = 0
y_test[y_test == selected_labels[1]] = 1

pf.calc_martingale(p_pred, y_test, len(y_test), k=2, plot_charts=True)

p_prime, cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot = pf.calibrate_probs(p_pred, y_test, len(y_test), k=2)



