import tensorflow as tf

def pcc(y_true, y_pred):
    y_true = tf.reshape(y_true, tf.shape(y_pred))

    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    return 1 - (
        tf.reduce_sum(y_true_centered * y_pred_centered) /
        (tf.norm(y_true_centered) * tf.norm(y_pred_centered))
    )