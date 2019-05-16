import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


EPOCHS = 5000
LEARNING_RATE = .02


def main():
    digits = load_digits()
    # digits = sklearn.datasets.load_digits()
    digits_y = np.identity(10)[digits.target]
    # y = np.identity(n_class)[labels]
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits_y, test_size=0.2, random_state=1)
    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.2, random_state=1)
    print('x_train: {}; x_test: {}'.format(x_train.shape, x_test.shape))
    print('y_train: {}; y_test: {}'.format(x_train.shape, y_test.shape))

    inputs = tf.keras.Input(shape=(64,))
    intermediate = tf.keras.layers.Dense(30, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(intermediate)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())

    for epoch in range(EPOCHS):
        with tf.GradientTape() as tape:
            y_predict = model(x_train)
            loss = tf.reduce_mean(tf.metrics.categorical_crossentropy(y_true=y_train, y_pred=y_predict))

        if epoch % 100 == 0:
            y = tf.argmax(y_train, axis=1)
            y_ = tf.argmax(y_predict, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_), tf.float32))
            print('Epoch [{}/{}], loss: {:.3f}, accuracy: {:.3f}'.format(epoch, EPOCHS, loss, accuracy))

        grads = tape.gradient(loss, model.variables)
        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
        optimizer.apply_gradients(zip(grads, model.variables))


if __name__ == '__main__':
    main()
