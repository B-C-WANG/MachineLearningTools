import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
from sklearn.decomposition import PCA


class PCA_Analysis():

    def __init__(self, X, y, PCA_N_components, feature_label=None):
        self.X = X
        self.y = y
        self.N_components = PCA_N_components
        self.feature_label = feature_label
        if self.feature_label is None:
            self.feature_label = [str(i) for i in range(self.X.shape[1])]

    def do_pca(self):
        pca = PCA(n_components=self.N_components, whiten=False)
        self.reduced_x = pca.fit_transform(self.X)
        self.comp = pca.components_

    def plot_pca_components_weights(self):
        # 权重图中正和负权重的颜色，包括字体和条形图
        OPP_COLOR = "firebrick"
        NEG_COLOR = "dodgerblue"

        if (self.N_components == 1):

            ax1 = plt.subplot(122)
            ax2 = plt.subplot(121)

            # 分别绘制正数和负数区域
            minus_index = []  # 负数的区域单独加上，之后会影响text颜色
            zero_index = []

            def plt_one_direction_bar(ax, direction):

                x = []
                y = []
                for i in range(self.comp.shape[1]):

                    if (direction == 1 and self.comp[0, i] > 0) or \
                            (direction == -1 and self.comp[0, i] < 0):
                        if (self.comp[0, i]) < 0: minus_index.append(i)
                        if (self.comp[0, i]) == 0: zero_index.append(i)
                        x.append(i)
                        y.append(self.comp[0, i])
                    else:
                        x.append(i)
                        y.append(0)
                if (direction == 1):
                    ax.barh(x, y, color=OPP_COLOR)
                elif (direction == -1):
                    ax.barh(x, y, color=NEG_COLOR)
                return ax

            ax1.get_yaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

            ax1.set_xlim(0, -0.35)
            ax1.set_xlim(0, 0.35)

            ax1.spines['top'].set_visible(False)

            ax1.spines['left'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            ax2.spines['top'].set_visible(False)

            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(False)

            plt_one_direction_bar(ax1, 1)

            plt_one_direction_bar(ax2, -1)

            plt.subplots_adjust(wspace=0.11, hspace=0)

            for i in range(len(self.feature_label)):
                string = self.feature_label[i]
                if i in minus_index:
                    color = NEG_COLOR
                else:
                    color = OPP_COLOR

                plt.text(0.005, i - 0.4, string, horizontalalignment='center', color=color)

            plt.show()


        else:
            plt.imshow(self.comp)
            plt.show()


    def show_pca_classification_results(self):

        # foreach label, use different color, add regard pca components as (x), (x,y) or (x,y,z)
        if (self.N_components >3):
            raise ValueError("Can not plot more than 4 dimension")

        total_label_pos_data = {}
        for sample_index in range(self.reduced_x.shape[0]):

            try:
                total_label_pos_data[list(y[sample_index]).index(1)].append(self.reduced_x[sample_index, :])
            except:
                total_label_pos_data[list(y[sample_index]).index(1)] = []

        color = ["dodgerblue", "firebrick", "green", "cyan", "magenta", "yellow"]
        total_used_sample = 0
        if self.N_components == 3: ax = plt.subplot(111, projection="3d")

        for label_index in range(len(total_label_pos_data.keys())):

            data = np.array(total_label_pos_data[label_index])

            data = data.reshape(self.N_components, -1)

            if (self.N_components == 1):
                plt.plot(np.random.randint(low=0, high=1000, size=data[0, :].shape), data[0, :], "o",
                         color=color[label_index], alpha=0.5)
                total_used_sample += data.shape[1]
            elif (self.N_components == 2):
                plt.plot(data[0, :], data[1, :], "o", color=color[label_index])
            elif (self.N_components == 3):
                ax.scatter(data[0, :], data[1, :], data[2, :], "x", color=color[label_index])

        plt.show()


if __name__ == '__main__':

    # 1. load data, X: shape (n_sample,n_feature), y: shape (n_sample, one-hot)
    # example data shape: X (4000,49), y:(4000,6)
    X = np.load("X.npy")
    y = np.load("y.npy")
    # 2. do pca
    test = PCA_Analysis(X, y, 3)
    test.do_pca()
    # 3. show the weights of feature to final reduced_x
    test.plot_pca_components_weights()
    # 4. plot classification results
    test.show_pca_classification_results()
