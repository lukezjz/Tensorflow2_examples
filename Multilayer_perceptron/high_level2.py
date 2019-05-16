import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    digits = load_digits()
    # digits = sklearn.datasets.load_digits()
    digits_y = np.identity(10)[digits.target]
    # y = np.identity(n_class)[labels]
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits_y, test_size=0.2, random_state=1)
    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.2, random_state=1)
    print('x_train: {}; x_test: {}'.format(x_train.shape, x_test.shape))
    print('y_train: {}; y_test: {}'.format(x_train.shape, y_test.shape))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=30, input_dim=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, steps_per_epoch=5000, validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
