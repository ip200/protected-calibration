import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, losses
import idx2numpy
from scripts import protected_functions as pf

import pandas as pd

DATA_DIR = '/Users/ivanpetej/Projects/protected-calibration/data/'


def run_digits(selected_digits, k):
    print(selected_digits)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train, x_test, y_test = pf.make_binary(x_train, y_train, x_test, y_test, selected_digits, 1, 0)

    x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
    x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255

    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)

    x_train_dist = idx2numpy.convert_from_file(DATA_DIR + 'train-images-idx3-ubyte')
    y_train_dist = idx2numpy.convert_from_file(DATA_DIR + 'train-labels-idx1-ubyte')

    x_test_dist = idx2numpy.convert_from_file(DATA_DIR + 't10k-images-idx3-ubyte')
    y_test_dist = idx2numpy.convert_from_file(DATA_DIR + 't10k-labels-idx1-ubyte')

    # train_ind = np.concatenate((np.where(y_train_dist == selected_digits[0])[0],
    #                             np.random.choice(np.where(y_train_dist == selected_digits[1])[0],
    #                                              int(len(y_train_dist[y_train_dist == selected_digits[1]])))))
    test_ind = np.concatenate((np.where(y_test_dist == selected_digits[0])[0],
                               np.random.choice(np.where(y_test_dist == selected_digits[1])[0],
                                                int(len(y_test_dist[y_test_dist == selected_digits[1]])))))
    #
    # x_train_dist = x_train_dist[train_ind]
    # y_train_dist = y_train_dist[train_ind]
    #
    x_test_dist = x_test_dist[test_ind]
    y_test_dist = y_test_dist[test_ind]

    # x_train_dist, y_train_dist, x_test_dist, y_test_dist = pf.make_binary(x_train_dist, y_train_dist, x_test_dist, y_test_dist, selected_digits, 1, 0)
    #
    # x_train_dist = tf.pad(x_train_dist, [[0, 0], [2, 2], [2, 2]]) / 255
    # x_train_dist = tf.expand_dims(x_train_dist, axis=3, name=None)

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

    print('Fitting model')
    history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val))

    test_loss, test_acc = model.evaluate(x_test, y_test)

    test_dist_loss, test_dist_acc = model.evaluate(x_test_dist, y_test_dist)

    y_pred = model.predict(x_test_dist)[:, selected_digits]
    p_pred = y_pred[:, 1]

    y_test_dist[y_test_dist == selected_digits[0]] = 0
    y_test_dist[y_test_dist == selected_digits[1]] = 1

    n_test = len(y_test_dist)

    log_sj_martingale, log_cj_martingale = pf.calc_martingale(p_pred, y_test_dist, n_test, k, plot_charts=False)

    p_prime, cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot = pf.calibrate_probs(
        p_pred, y_test_dist, n_test, k
    )

    return [p_prime, p_pred,
            y_test_dist], test_loss, test_acc, test_dist_loss, test_dist_acc, p_pred, p_prime, y_test_dist, n_test, \
           log_sj_martingale[0][-1], log_cj_martingale[-1], cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot


selected_digits = [1, 7]
res, test_loss, test_acc, test_dist_loss, test_dist_acc, p_pred, p_prime, y_test_dist, N_test, log_sj_martingale, \
log_cj_martingale, cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot = run_digits(selected_digits, k=2)


results = pd.DataFrame()
col_names = ['selected_digits', 'test_loss', 'test_acc', 'test_dist_loss', 'test_dist_acc', 'N_test', 'log_sj_martingale', 'log_cj_martingale', 'cum_loss_base', 'roc_auc_base', 'cum_loss_prot', 'roc_auc_prot']
for i in range(9):
    for j in range(i + 1, 10):
        selected_digits = [i,j]
        _, test_loss, test_acc, test_dist_loss, test_dist_acc, p_pred, p_prime, y_test_dist, N_test, log_sj_martingale, log_cj_martingale, cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot = run_digits(selected_digits, k=2)
        opa = pd.DataFrame([[selected_digits, test_loss, test_acc, test_dist_loss, test_dist_acc, N_test, log_cj_martingale, log_cj_martingale, cum_loss_base, roc_auc_base, cum_loss_prot, roc_auc_prot]], columns=col_names)
        results = results.append(opa)
        results.to_csv('results_partitioned.csv')


