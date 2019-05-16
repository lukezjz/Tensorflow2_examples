import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


EPOCHS = 100
LEARNING_RATE = .02


class Model(object):
    def __init__(self):
        self.w1 = tf.Variable(tf.random.truncated_normal([64, 30]))
        self.b1 = tf.Variable(tf.random.truncated_normal([30]))
        self.w2 = tf.Variable(tf.random.truncated_normal([30, 10]))
        self.b2 = tf.Variable(tf.random.truncated_normal([10]))
        # tensor = tf.random.truncated_normal(shape)
        # tensor = tf.constant(num, shape)
        # parameter = tf.Variable(tensor)

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        # x = tf.cast(x, dtype)
        fc1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        # out = tf.matmul(x1, x2)
        # out = tf.nn.relu(x)
        y = tf.matmul(fc1, self.w2) + self.b2
        return y


def loss_func(model, x, y):
    y_ = model(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
    cross_entropy = tf.reduce_mean(cross_entropy)
    # x = tf.reduce_mean(x)
    return cross_entropy


def accuracy_func(logits, labels):
    preds = tf.argmax(logits, axis=1)
    labels = tf.argmax(labels, axis=1)
    # axis describes which axis of the input Tensor to reduce across
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
    # tf.equal(y1, y2)
    return accuracy


def main():
    digits = load_digits()
    # digits = sklearn.datasets.load_digits()
    digits_y = np.identity(10)[digits.target]
    # y = np.identity(n_class)[labels]
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits_y, test_size=0.2, random_state=1)
    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.2, random_state=1)
    print('x_train: {}; x_test: {}'.format(x_train.shape, x_test.shape))
    print('y_train: {}; y_test: {}'.format(x_train.shape, y_test.shape))

    model = Model()

    for epoch in range(EPOCHS):
        with tf.GradientTape() as tape:
            loss = loss_func(model, x_train, y_train)
            accuracy = accuracy_func(model(x_test), y_test)
        trainable_variables = [model.w1, model.b1, model.w2, model.b2]
        grads = tape.gradient(loss, trainable_variables)
        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
        optimizer.apply_gradients(zip(grads, trainable_variables))

        if epoch % 10 == 0:
            print('Epoch [{}/{}], loss: {:.3f}, accuracy: {:.3f}'.format(epoch, EPOCHS, loss, accuracy))


if __name__ == '__main__':
    main()
