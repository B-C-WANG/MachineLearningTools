from soapml.Dataset import  Dataset
from MLT.VAE.VAE1D import VAE1D
import numpy as np
from MLT.Regression.LinearNNFeatureExtraction import LinearNN
from MLT.Regression.GBR_FeatureImportanceEstimater import GBRFIE

def gbr_feature():
    model = GBRFIE(X, y, test_split_ratio=0.3)
    model.fit(n_estimators=40)
    error = model.show_pred_train_test(plot_fig=True, point_size=100)
    print(error)
    model.print_out_importance()

def linear_NN():
    X = np.load("X.npy")
    y = np.load('y.npy')

    model = LinearNN(X, y)
    model.build_model()
    model.train(epochs=4000)
    model.show_results()
    model.plot_weights()


def vae1d_demo():
    dataset = Dataset.load("cnecdaDataset_test.smld")
    datax = dataset.datasetx
    datay = dataset.datasety
    batch_size = 100
    # make it can be // by batch size
    more_sample = datax.shape[0] % batch_size
    sample_num = datax.shape[0] - more_sample
    datax = datax[:sample_num, :]
    datay = datay[:sample_num]
    original_dim = datax.shape[1]
    intermediate_dim = 256
    # normalize data
    datax /= np.max(datax)
    model = VAE1D(original_dim, intermediate_dim, batch_size)
    model.fit(x_train=datax, epochs=30)
    encode_result = model.encode(datax)
    model.plot_encoded_result_as_dimension_reduction(encode_result, y=datay)


