# coding: utf-8
import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + tf.math.exp(-x))

def relu(x):
    return tf.math.greater(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = tf.math.exp(x)
        x /= tf.math.reduce_max(x, axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = tf.math.exp(x) / tf.reduce_sum(tf.math.exp(x))
    return x

def cross_entropy_error(y, t):
    # if y.ndim == 1:
    #     t = tf.reshape(t, (1, t.shape[0]))
    #     y = tf.reshape(y, (1, y.shape[0]))
        
    # # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    # if t.shape == y.shape:
    #     t = tf.math.argmax(t, axis=1)
             
    batch_size = y.shape[0]

    return -tf.reduce_sum(tf.math.log(y[tf.range(0, batch_size), t] + 1e-7)) / batch_size