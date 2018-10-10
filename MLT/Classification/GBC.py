#coding: utf-8

# GradientBoost分类器，非常有效

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from MLT.Classification.plot_confusion_matrix import plot_confusion_matrix
import pickle as pkl

class GBC():
    def __init__(self,X,y,y_is_onehot=False):
        self.X = X
        self.y = y
        if y_is_onehot == True:
            self.y = np.argmax(self.y,axis=1)


    def fit(self):
        self.model = GradientBoostingClassifier(n_estimators=50)
        self.model.fit(self.X,self.y)

    def show_result(self,label=None):
        pred_y = self.model.predict(self.X)
        real_y = self.y
        if label == None:
            label = list(set(real_y))
        plot_confusion_matrix(real_y,pred_y,labels = label)
        error = np.sum(np.equal(pred_y, real_y)) / pred_y.shape[0]
        print("Accuracy", error)

        correct_by_label = np.zeros(shape=(max(real_y) + 1))
        all_by_label = np.zeros(shape=max(real_y) + 1)
        for i in range(len(real_y)):
            all_by_label[real_y[i]] += 1
            if real_y[i] == pred_y[i]: correct_by_label[real_y[i]] += 1
        correct_ratio = correct_by_label / all_by_label
        print("Accuracy by label:\n", "\n".join(list(correct_ratio.astype("str"))))

    def plot_importance(self,label=None):
        plt.barh(list(range(len(self.model.feature_importances_))),self.model.feature_importances_)
        if label != None:
            for i in range(len(label)):
                string = label[i]
                plt.text(0.005,i-0.4,string,horizontalalignment='center')

        plt.show()


if __name__ == '__main__':

    X = np.load("X.npy")
    y = np.load("y.npy")
    model = GBC(X,y,y_is_onehot=True)
    model.fit()
    model.show_result()
    model.plot_importance()