# coding: utf-8
import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1 / (1 + tf.math.exp(-x))

def relu(x):
    return tf.dtypes.cast(tf.math.greater(x, 0), 'float') * x

def softmax(x):
    if x.ndim == 2:
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = tf.math.exp(x)
        x /= tf.math.reduce_sum(x, axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - tf.math.reduce_max(x, axis=0, keepdims=True)
        x = tf.math.exp(x) / tf.reduce_sum(tf.math.exp(x))
    return x

def cross_entropy_error(y, t):
    num_label = y.shape[1]
    t = tf.dtypes.cast(t, dtype='float')
    
    if t.ndim == 1:
        t = tf.one_hot(t, num_label)
        
    batch_size = y.shape[0]

    return -tf.reduce_sum(tf.math.log(tf.reduce_max(y*t, axis=1))) / batch_size
