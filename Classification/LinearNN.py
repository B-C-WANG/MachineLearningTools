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
        model.add(tf.keras.layers.Activation("softmax"))
        model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        self.model = model
        model.summary()

    def train(self, epochs=400, batch_size=200):
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size)

    def show_results(self):
        pred_y = np.argmax(self.model.predict(self.X), axis=1)
        real_y = np.argmax(self.y, axis=1)

        error = np.sum(np.equal(pred_y, real_y)) / pred_y.shape[0]
        print("Accuracy", error)

        correct_by_label = np.zeros(shape=(max(real_y) + 1))
        all_by_label = np.zeros(shape=max(real_y) + 1)
        for i in range(len(real_y)):
            all_by_label[real_y[i]] += 1
            if real_y[i] == pred_y[i]: correct_by_label[real_y[i]] += 1
        correct_ratio = correct_by_label / all_by_label
        print("Accuracy by label:\n", "\n".join(list(correct_ratio.astype("str"))))

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

