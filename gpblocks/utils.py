import tensorflow as tf 


def to_default_float(x):
    return tf.cast(x, dtype=tf.float64)
