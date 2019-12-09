from abc import ABCMeta, abstractmethod
import tensorflow as tf

class Layer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def backward(self):
        pass

class MatMul(Layer):
    def __init__(self, W):
        self.params = [W]
        self.grads = [tf.zeros(W.shape)]
        self.x = None
        
    def forward(self, x):
        W,  = self.params
        out = tf.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = tf.matmul(dout, tf.transpose(W))
        dW = tf.matmul(tf.transpose(self.x), dout)
        self.grads[0] = dW
        return dx

class Affine(Layer):
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [tf.zeros(W.shape), tf.zeros(b.shape)]
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
        db = tf.reduce_sum(dout, axis=0)

        self.grads[0] = dW
        self.grads[1] = db

class Softmax:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None
    
    def forward(self, x):
        self.out = 1 / (1 + tf.math.exp(x))
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = tf.reduce_sum(dx, axis=1, keepdim=True)
        dx -= self.out * sumdx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + tf.math.exp(x))

        if self.t.shape == self.y.shep:
            self.t = tf.math.argmax(self.t)

        loss = cross_entropy_error(self.y, self.t)
        return loss