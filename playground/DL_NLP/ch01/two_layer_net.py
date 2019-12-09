import sys
sys.path.append('..')
import tensorflow as tf
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * tf.random.uniform((I, H), dtype='double')
        b1 = tf.zeros((H, ), dtype='double')
        W2 = 0.01 * tf.random.uniform((H, O), dtype='double')
        b2 = tf.zeros((O, ), dtype='double')

        self.layers = [
                Affine(W1, b1),
                Sigmoid(),
                Affine(W2, b2),
                ]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

