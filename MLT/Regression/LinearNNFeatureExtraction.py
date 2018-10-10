import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pickle as pkl


class LinearNN():

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.y.shape[1], input_shape=(self.X.shape[1],)))

        model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss=tf.keras.losses.mean_squared_error)
        self.model = model
        model.summary()

    def train(self, epochs=400, batch_size=200):
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size)

    def show_results(self):
        pred_y = self.model.predict(self.X)
        real_y = self.y

        error = np.mean(np.abs(pred_y-real_y))
        print("Mean error", error)

    def plot_weights(self):
        weights = self.model.get_weights()[0]
        plt.imshow(weights, cmap="RdBu")
        plt.colorbar()
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    X = np.load("X.npy")
    y = np.load("y.npy")
    test = LinearNN(X, y)
    test.build_model()
    test.train()
    test.show_results()
    test.plot_weights()

