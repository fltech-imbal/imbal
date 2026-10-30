import tensorflow as tf

def safe_norm(x, axis):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + 1e-12)