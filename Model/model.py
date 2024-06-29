#this defines custom layers for the model. 
'''The aim is to develop a graph based deep learning method to forecast the price direction on 21st day given 20 time steps of variables of a set of stocks. This stocks...
... are believed to influence the pricing of each other, and hence it is expected that the graph based method can decipher the complex inter-relations'''

from spektral.layers import GCNConv as GraphConv
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout, LSTM, Concatenate, Input, Layer, Lambda, Reshape, Conv1D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from spektral.utils import normalized_adjacency
import scipy.stats as st
import numpy as np

def squeeze_end2axes_operator( x4d ):
    shape = tf.shape( x4d )
    x3d = tf.reshape( x4d, [shape[0], shape[1], shape[2] * shape[3]])
    return x3d

def squeeze_end2axes_shape( x4d_shape ):
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if ( None in [ in_cols, in_filters] ):
        output_shape = ( in_batch, in_rows, None )
    else:
        output_shape = ( in_batch, in_rows, in_cols * in_filters )
    return output_shape

'''This layer learns a lower dimensional feature set for the input features, and uses a autoencoder concept'''

class autoEncoderLayer(keras.layers.Layer):

    def __init__(self, gamma1 = 1e-2, **kwargs):
        super(autoEncoderLayer, self).__init__(**kwargs)
        self.gamma = gamma1

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
            trainable=True, name = 'encoder-alpha',
        )
        self.alpha1 = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=None),
            trainable=True, name = 'encoder-alpha',
        )
        self.epsi = self.add_weight(
            shape=(input_shape[-3], input_shape[-2], 1), initializer="random_normal", trainable=True, regularizer = tf.keras.regularizers.L1(l1=1e-2),
            name = 'decoder-epsi',
        )
        self.epsi1 = self.add_weight(
            shape=(input_shape[-3], input_shape[-2], input_shape[-1]), initializer="random_normal", trainable=True, regularizer = tf.keras.regularizers.L1(l1=1e-2),
            name = 'decoder-epsi',
        )

    def call(self, inputs):
        encoder = tf.matmul(inputs, self.alpha) + epsi
        decoder = tf.matmul(encoder, self.alpha1) + epsi1
        loss_fn = tf.math.square(tf.math.substract(decoder-inputs))
        self.add_loss(loss_fn)
        return encoder

    def get_config(self):
        config = super(autoEncoderLayer, self).get_config()
        config.update({"units": 'Nil'})
        return config

'''This layer intends to learn the graph structure using a matrix logit function, the output is a non-directional and non-weighted graph'''
class graph_AutoGen(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(graph_AutoGen, self).__init__(**kwargs)

    def build(self, input_shape):
        self.theta = self.add_weight(
            shape=(input_shape[-2], input_shape[-2]),
            regularizer = tf.keras.regularizers.L2(l2=1e-2),
            initializer="random_uniform",
            trainable=True, name = 'graph_theta',
        )
        self.threshold = self.add_weight(
            shape=(input_shape[-2], input_shape[-2]),
            regularizer = tf.keras.regularizers.L2(l2=1e-2),
            initializer="random_uniform",
            trainable=True, name = 'graph_theta',
        )

    def customGraphActivation(self, x):
      condition = tf.greater(x, tf.sigmoid(self.threshold))
      return tf.where(condition, tf.ones(tf.shape(x)), tf.zeros(tf.shape(x)))

    def call(self, inputs):
        e = inputs

        m = tf.matmul(tf.matmul(tf.transpose(e, perm = [0, 2, 1]), theta), e)

        a = self.customGraphActivation(tf.sigmoid(m))

        return a

    def get_config(self):
        config = super(graph_AutoGen, self).get_config()
        config.update({"units": 'graph_auto'})
        return config

'''This layer calls in the GCN implementation from Spektral. LSTM is used to learn any time dependencies in the node features (which are time steps)'''
def gcn_lstm_Layer(inputs, outshape = 8, gcn_units_1 = 4, lstm_units_1 = 8, activation = 'tanh'):
    gcn_units1 = gcn_units_1
    lstm_units1 = lstm_units_1
    activation = tf.keras.activations.deserialize(activation, custom_objects=None)
    outshape = outshape

    x, a = inputs
    x1 = GraphConv(gcn_units1, activation=activation, dropout_rate = 0, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3), 
            bias_regularizer=tf.keras.regularizers.L2(l2=1e-3))([x, a])
    x2 = LSTM(lstm_units1, activation=activation, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3), 
             recurrent_regularizer=tf.keras.regularizers.L2(l2=1e-3), bias_regularizer=tf.keras.regularizers.L2(l2=1e-3))(tf.transpose(x1, perm = [0, 2, 1]))
    return activation(tf.transpose(out, perm = [0, 2, 1]))

'''This function builds the model by taking in input as the dimension of each of the training examples i.e. timesteps*n_nodes*n_features'''
def get_model(sample_graph, gamma = 1e-3):
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-09, amsgrad=False, name='Adam')
    gamma_featsel = gamma
    n_nodes = sample_graph[0].shape[0]
    n_variables = sample_graph[0].shape[2]
    sequence_length = sample_graph[0].shape[1]
    inp_seq1 = Input((n_nodes, sequence_length, n_variables))
    x = autoEncoderLayer(gamma1 = gamma_featsel)(inp_seq1)
    x = Reshape((x.shape[1], x.shape[2]), input_shape = (x.shape[1], x.shape[2], x.shape[3]))(x)
    a = graph_AutoGen()(x)
    x = gcn_lstm_Layer([x, a], outshape = 8, gcn_units_1 = 4, lstm_units_1 = 8, activation = 'tanh')
    x = Dropout(0.45)(x)
    out = Dense(2, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3), bias_regularizer=tf.keras.regularizers.L2(l2=1e-3))(x)
    model = Model(inp_seq1, out)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    return model
