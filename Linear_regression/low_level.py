import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


EPOCHS = 100
LEARNING_RATE = 0.01


class Model(object):
    def __init__(self):
        self.w = tf.Variable(tf.random.uniform([1]))
        self.b = tf.Variable(tf.random.uniform([1]))

    def __call__(self, x):
        y = self.w * x + self.b
        return y


def loss_fn(model, x, y):
    y_ = model(x)
    return tf.reduce_mean(tf.square(y_ - y))


def main():
    x = np.random.uniform(0, 10., size=(100, 1))
    noise = np.random.normal(0, 2., size=(100, 1))
    y = 3.0 * x + 2. + noise

    model = Model()
    plt.scatter(x, y)
    plt.plot(x, model(x), c='r')
    plt.show()

    for epoch in range(EPOCHS):
        with tf.GradientTape() as tape:
            loss = loss_fn(model, x, y)
        dw, db = tape.gradient(loss, [model.w, model.b])
        model.w.assign_sub(LEARNING_RATE * dw)
        model.b.assign_sub(LEARNING_RATE * db)
        print('Epoch [{}/{}], loss [{:.3f}], w/b [{:.3f}/{:.3f}]'.format(epoch, EPOCHS, loss, float(model.w.numpy()), float(model.b.numpy())))

    plt.scatter(x, y)
    plt.plot(x, model(x), c='r')
    plt.show()


if __name__ == '__main__':
    main()
