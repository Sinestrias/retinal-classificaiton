import tensorflow as tf

def make_weighted_bce(pos_weight):
    pos_weight_tf = tf.constant(pos_weight, dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        term1 = pos_weight_tf * y_true * tf.math.log(y_pred)
        term2 = (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return -tf.reduce_mean(term1 + term2)

    return weighted_bce
