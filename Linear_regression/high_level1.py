import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


EPOCHS = 100
LEARNING_RATE = 0.0001


def main():
    x = np.random.uniform(0, 10., size=(100, 1))
    noise = np.random.normal(0, 2., size=(100, 1))
    y = 3.0 * x + 2. + noise

    model = tf.keras.layers.Dense(units=1)
    plt.scatter(x, y)
    plt.plot(x, model(x), c='r')
    plt.show()

    for epoch in range(EPOCHS):
        with tf.GradientTape() as tape:
            y_ = model(x)
            loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(y, y_))
        grads = tape.gradient(loss, model.variables)
        optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        optimizer.apply_gradients(zip(grads, model.variables))
        print('Epoch [{}/{}], loss [{:.3f}]'.format(epoch, EPOCHS, loss))

    plt.scatter(x, y)
    plt.plot(x, model(x), c='r')
    plt.show()


if __name__ == '__main__':
    main()
