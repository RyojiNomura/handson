# coding: utf-8
from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error
import tensorflow as tf


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [tf.zeros_like(W), tf.zeros_like(b)]
        self.x = None
    
    def forward(self, x):
        W, b = self.params
        out = tf.matmul(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = tf.matmul(dout, tf.transpose(W))
        dW = tf.matmul(tf.transpose(self.x), dout)
        db = tf.math.reduce_sum(dout, axis=0)
        self.grads[0] = dW
        self.grads[1] = db
        return dx
    
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [tf.zeros_like(W)]
        self.x = None
        
    def forward(self, x):
        W, = self.params
        out = tf.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = tf.matmul(dout, tf.transpose(W))
        dW = tf.matmul(tf.transpose(self.x), dout)
        self.grads[0] = dW
        return dx

class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = tf.reduce_sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.num_label = self.y.shape[1]
        if self.t.ndim == 1:
            self.t = tf.one_hot(self.t, num_label)
        
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        t = self.t
        if t.ndim == 1:
            t = tf.one_hot(t, self.num_label)
        t = tf.dtypes.cast(t, dtype='float')

        dx = self.y
        dx_ones = tf.ones_like(self.y) * t
        dx -= dx_ones
        dx *= dout
        dx = dx / batch_size
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class Dropout:
    '''
    http://arxiv.org/abs/1207.0580
    '''
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None