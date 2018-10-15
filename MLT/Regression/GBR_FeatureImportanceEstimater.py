

# GBR非常好用，可以调整参数实现测试集训练集误差平衡，同时能够展示出Feature的重要性

import numpy as np
import matplotlib.pyplot as plt
import tqdm
X = np.load("X.npy")
y = np.load('y.npy')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class GBRFIE(object):
    def __init__(self,X,y,test_split_ratio=0.3):
        self.test_split_ratio = test_split_ratio
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_split_ratio, shuffle=True)

    def fit(self,n_estimators=40):

        model = GradientBoostingRegressor(n_estimators=n_estimators)
        model.fit(self.x_train, self.y_train)
        self.model = model

    @staticmethod
    def search_n_estimators_to_error(X,y,split_ratio=0.3,search_min=20,search_max=160,repeat_for_each_n=10):  # 找error与fit参数的关系
        '''

        :param search_min:
        :param search_max:
        :param repeat_for_each_n: get average of error repeat for n times.
        :return:
        '''
        test_error_l = []
        train_error_l = []
        x = []
        for i in tqdm.trange(search_min, search_max):
            tr = []
            te = []
            for _ in range(repeat_for_each_n):  # 训练5次找平均值
                model = GBRFIE(X,y,split_ratio)
                model.fit(n_estimators=i)
                train_error,test_error = model.show_pred_train_test(train_error=True, plot_fig=False)
                tr.append(train_error)
                te.append(test_error)
            train_error_l.append(np.mean(tr))
            test_error_l.append(np.mean(te))
            x.append(i)
        return x, train_error_l, test_error_l

    def show_pred_train_test(self,train_error=False,plot_fig=True,save_fig_filename=False,point_size=60):
        model = self.model
        # 绿色为训练集，红色为测试集
        pred_train_y = model.predict(self.x_train)
        real_train_y = self.y_train
        if plot_fig:
            plt.scatter(pred_train_y, real_train_y, s=point_size)


        pred_y = model.predict(self.x_test)
        real_y = self.y_test

        error = np.mean(np.abs(pred_y-real_y))


        if plot_fig:
            plt.scatter(pred_y,real_y,s=point_size)
            if isinstance(save_fig_filename,str):
                plt.savefig(save_fig_filename,dpi=600)
            plt.show()
        if train_error == False:
            return error
        else:
            train_error = np.mean(np.abs(pred_train_y-real_train_y))
            return train_error,error

    def print_out_importance(self):
        importance = self.model.feature_importances_
        print(importance)