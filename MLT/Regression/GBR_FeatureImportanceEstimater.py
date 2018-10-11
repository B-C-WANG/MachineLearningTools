

# GBR非常好用，可以调整参数实现测试集训练集误差平衡，同时能够展示出Feature的重要性

import numpy as np
import matplotlib.pyplot as plt
X = np.load("X.npy")
y = np.load('y.npy')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class GBRFIE(object):
    def __init__(self,X,y,test_split_ratio=0.3):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_split_ratio, shuffle=True)

    def fit(self,n_estimators=40):

        model = GradientBoostingRegressor(n_estimators=n_estimators)
        model.fit(self.x_train, self.y_train)
        self.model = model

    def show_pred_train_test(self):
        model = self.model
        # 绿色为训练集，红色为测试集
        pred_train_y = model.predict(self.x_train)
        real_y = self.y_train
        plt.plot(pred_train_y,real_y,"go")


        pred_y = model.predict(self.x_test)
        real_y = self.y_test

        error = np.mean(np.abs(pred_y-real_y))
        print("Mean error", error)
        plt.plot(pred_y,real_y,"ro")
        plt.show()

    def print_out_importance(self):
        importance = self.model.feature_importances_
        print(importance)