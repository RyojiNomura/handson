import sys
sys.path.append('..')
import tensorflow as tf
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = tf.Variable(tf.random.normal((I, H), mean=0.0, stddev=0.01, dtype='float'))
        b1 = tf.Variable(tf.zeros(H, dtype='float'))
        W2 = tf.Variable(tf.random.normal((H, O), mean=0.0, stddev=0.01, dtype='float'))
        b2 = tf.Variable(tf.zeros(O, dtype='float'))

        self.layers = [
                Affine(W1, b1),
                Sigmoid(),
                Affine(W2, b2),
                ]
        self.loss_layer = SoftmaxWithLoss()
        self.params = []
        for layer in self.layers:
            self.params += layer.params
            
    def predict(self, x):
        x = tf.dtypes.cast(x, dtype='float')
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        x = tf.dtypes.cast(x, dtype='float')
        t = tf.dtypes.cast(t, dtype='int32')
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = tf.dtypes.cast(dout, dtype='float')
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        self.grads = []
        for layer in self.layers:
            self.grads += layer.grads

        return dout

