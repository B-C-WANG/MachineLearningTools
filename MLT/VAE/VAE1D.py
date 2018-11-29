import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import colors
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
import pickle as  pkl

class CustomVariationalLayer(Layer):
    def __init__(self, original_dim,z_mean,z_log_var,**kwargs):
        self.is_placeholder = True
        self.original_dim=original_dim
        self.z_mean=z_mean
        self.z_log_var = z_log_var
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):

        #xent_loss =  original_dim * metrics.binary_crossentropy(x, x_decoded_mean) # the loss function should be changed
        xent_loss =  self.original_dim * metrics.mae(x, x_decoded_mean) # the loss function should be changed



        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]

        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)

        # We won't actually use the output.
        return x




class VAE1D():
    def __init__(self,original_dim,intermediate_dim,batch_size,latent_dim=2,epsilon_std=1.0,optimizer="rmsprop"):
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = 2
        self.epsilon_std = epsilon_std
        self.batch_size = batch_size

        self.x = Input(batch_shape=(batch_size, original_dim))
        self.h = Dense(intermediate_dim, activation='relu')(self.x)
        self.z_mean = Dense(latent_dim)(self.h)
        self.z_log_var = Dense(latent_dim)(self.h)
        self.z = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_var])

        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')
        self.h_decoded = decoder_h(self.z)
        self.x_decoded_mean = decoder_mean(self.h_decoded)

        self.y = CustomVariationalLayer(self.original_dim,self.z_mean,self.z_log_var)([self.x, self.x_decoded_mean])  # custom层主要看call，输入x和decoded结果，然后输出原始和解码后的loss
        self.vae = Model(self.x, self.y)
        self.vae.compile(optimizer=optimizer)
        self.vae.summary()

        self.encoder = Model(self.x,self.z_mean)

        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

    def fit(self,x_train,epochs,shuffle=True,**kwargs):
        self.vae.fit(x_train,shuffle=shuffle,batch_size=self.batch_size,epochs=epochs,*kwargs)

    def encode(self,x,**kwargs):
        return self.encoder.predict(x,batch_size=self.batch_size,*kwargs)

    def plot_encoded_result_as_dimension_reduction(self,encode_result,y=None):
        assert isinstance(encode_result,np.ndarray)
        assert encode_result.shape[1] == 2
        fig = plt.figure()
        ax = plt.subplot(111)
        norm = colors.Normalize()
        cm = plt.cm.get_cmap('RdYlBu')
        if y is not None:
            ax.scatter(encode_result[:, 0],encode_result[:, 1], norm=norm, c=y, cmap=cm)
        else:
            ax.scatter(encode_result[:, 0], encode_result[:, 1], norm=norm,cmap=cm)
        plt.show()

    def save(self,filename="model.pkl"):
        with open(filename, "wb") as  f:
            pkl.dump(self,f)

    @staticmethod
    def load(filename="model.pkl"):
        with open(filename, "rb")  as f:
            return pkl.load(f)

    def sampling(self,args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
