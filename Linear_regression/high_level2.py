import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    x = np.random.uniform(0, 10., size=(100, 1))
    noise = np.random.normal(0, 2., size=(100, 1))
    y = 3.0 * x + 2. + noise

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_dim=1))
    print(model.summary())
    plt.scatter(x, y)
    plt.plot(x, model(x), c='r')
    plt.show()

    model.compile(optimizer='sgd', loss='mse')
    model.fit(x, y, steps_per_epoch=1000)
    plt.scatter(x, y)
    plt.plot(x, model(x), c='r')
    plt.show()


if __name__ == '__main__':
    main()
