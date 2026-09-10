import tensorflow as tf
import tensorflow_probability as tfp
from .safe_norm import safe_norm

EPSILON = 1e-6

def ratio_loss(train_label_min, train_label_max, lambda_val=1, ratio_bound=3, unit=False):

    def loss(labels, representations, weights=None):
        if unit:
            train_label_range = train_label_max - train_label_min
            ideal_ratio = 1.8 / train_label_range
        else:
            ideal_ratio = ratio_bound
        ideal_ratio = tf.cast(ideal_ratio, tf.float32)

        distance_to_next_label = labels[1:] - labels[:-1]

        distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)

        ratio = tf.reduce_sum(distance_to_next_representation) / (tf.reduce_sum(distance_to_next_label) + 1e-12)
        ratio = tf.clip_by_value(ratio, 1e-7, 1e7)

        loss_value = (tf.keras.ops.log10(ratio)/(2*tf.keras.ops.log(ideal_ratio)))**2

        return loss_value * lambda_val

    return loss

def no_reprepresention_loss(labels, representations, weight=None, unit=False):
    return 0.0

def minimize_variance(labels, representations, weight=None):
    distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    distance_to_next_label = tf.reshape(distance_to_next_label, [-1, 1])
    distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
    distance_to_next_representation = tf.reshape(distance_to_next_representation, [-1, 1])
    ratios = distance_to_next_representation / (distance_to_next_label + EPSILON)

    std_dev = tf.math.reduce_std(tf.keras.ops.log10(ratios))
    return std_dev

def augmented_pcc(labels, representations, weight=None, unit=False):
    labels_reshaped = tf.reshape(labels, (-1, 1))
    extended_representations = tf.concat([representations, labels_reshaped], axis=1)

    return 1 - tf.reduce_mean(tf.math.abs(tfp.stats.correlation(extended_representations)))

def distance_pcc(labels, representations, weight=None, unit=False):
    # print(tf.shape(representations))
    # print(tf.shape(labels))
    distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    distance_to_next_label = tf.reshape(distance_to_next_label, (-1, 1))

    distance_to_next_representation = safe_norm(
        representations[1:] - representations[:-1],
        axis=1
    )
    distance_to_next_representation = tf.reshape(
        distance_to_next_representation, (-1, 1)
    )

    return 1 - tfp.stats.correlation(
        distance_to_next_label,
        distance_to_next_representation,
        sample_axis=0,
        event_axis=1
    )

def distance_pcc_decorrelation(labels, representations, weight=None, unit=False):

    distance_to_next_label = tf.abs(labels[1:] - labels[:-1])

    distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)

    labels_reshaped = tf.reshape(labels, (-1, 1))
    extended_representations = tf.concat([representations, labels_reshaped], axis=1)

    return 1 - tfp.stats.correlation(distance_to_next_label, tf.expand_dims(distance_to_next_representation, axis=-1)) + tf.reduce_mean(tf.math.abs(tfp.stats.correlation(extended_representations)))

def cauchy_schwartz(labels, representations, weight=None, unit=False):
    # print(labels)
    distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    # distance_to_first_label = tf.abs(labels[1:] - labels[0])

    distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
    # distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=1)

    # combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0)
    # combined_label_distances = tf.squeeze(combined_label_distances)
    # combined_representation_distances = tf.concat(
    #     [distance_to_next_representation, distance_to_first_representation],
    #     axis=0)
    # a = combined_label_distances
    # b = combined_representation_distances
    a = distance_to_next_label
    b = distance_to_next_representation

    # print(tf.reduce_mean(tf.multiply(a, a)) * tf.reduce_mean(tf.multiply(b, b)) - tf.reduce_mean(tf.multiply(a, b))**2)
    return tf.reduce_sum(tf.multiply(a, a)) * tf.reduce_sum(tf.multiply(b, b)) - tf.reduce_sum(tf.multiply(a, b))**2


def maximize_entropy(labels, representations, weight=None):
    distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    distance_to_first_label = tf.abs(labels[1:] - labels[0])
    distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=-1)
    distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)

    combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0) + EPSILON
    combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_first_representation],
                                                  axis=0) + EPSILON
    combined_representation_distances = tf.reshape(combined_representation_distances, tf.shape(combined_label_distances))

    ratios = combined_label_distances / combined_representation_distances
    ratios = ratios / tf.reduce_sum(ratios)
    return tf.reduce_sum(ratios * tf.math.log(ratios)) - tf.cast(tf.math.log(1 / tf.size(ratios)), dtype=tf.float32)


def decorrelation(labels, representations, weight=None, eps=1e-6):
    # Center the representations
    mean = tf.reduce_mean(representations, axis=0, keepdims=True)
    centered = representations - mean

    # Covariance
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        tf.shape(representations)[0] - 1, representations.dtype
    )

    # Std with epsilon to avoid div-by-zero on constant features
    std = tf.sqrt(tf.linalg.diag_part(cov) + eps)
    corr = cov / (std[:, None] * std[None, :] + eps)

    # Belt-and-suspenders: zero out any NaN/Inf that still appears
    corr = tf.where(tf.math.is_finite(corr), corr, tf.zeros_like(corr))

    n = tf.shape(corr)[0]
    mask = tf.ones_like(corr) - tf.eye(n, dtype=corr.dtype)

    denom = tf.reduce_sum(mask)
    return tf.math.divide_no_nan(tf.reduce_sum(tf.abs(corr) * mask), denom)

def maximize_entropy_w_ratio_loss(train_label_min, train_label_max, lambda_val=1, unit=False, decorr=False):
    current_ratio_loss = ratio_loss(train_label_min, train_label_max, lambda_val, unit=unit)

    def loss_function(labels, representations, weight=None):
        return maximize_entropy(labels, representations, weight) + current_ratio_loss(labels, representations, weight)

    def loss_function_decorrelation(labels, representations, weight=None):
        return maximize_entropy(labels, representations, weight) + current_ratio_loss(labels, representations, weight) + decorrelation(labels, representations, weight)

    return loss_function_decorrelation if decorr else loss_function

def cauchy_schwartz_w_ratio_loss(train_label_min, train_label_max, lambda_val=1, unit=False, decorr=False):
    current_ratio_loss = ratio_loss(train_label_min, train_label_max, lambda_val, unit=unit)

    def loss_function(labels, representations, weight=None):
        return cauchy_schwartz(labels, representations, weight) + current_ratio_loss(labels, representations, weight)

    def loss_function_decorrelation(labels, representations, weight=None):
        return cauchy_schwartz(labels, representations, weight) + current_ratio_loss(labels, representations, weight) + decorrelation(labels, representations, weight)
    return loss_function_decorrelation if decorr else loss_function

def distance_pcc_w_ratio_loss(train_label_min, train_label_max, lambda_val=1, unit=False, decorr=False):
    current_ratio_loss = ratio_loss(train_label_min, train_label_max, lambda_val, unit=unit)

    def loss_function(labels, representations, weight=None):
        return distance_pcc(labels, representations, weight) + current_ratio_loss(labels, representations, weight)

    def loss_function_decorrelation(labels, representations, weight=None):
        return distance_pcc(labels, representations, weight) + current_ratio_loss(labels, representations, weight) + decorrelation(labels, representations, weight)

    return loss_function_decorrelation if decorr else loss_function

def minimize_variance_w_ratio_loss(train_label_min, train_label_max, lambda_val=1, unit=False, decorr=False):
    current_ratio_loss = ratio_loss(train_label_min, train_label_max, lambda_val, unit=unit)

    def loss_function(labels, representations, weight=None):
        return minimize_variance(labels, representations, weight) + current_ratio_loss(labels, representations, weight)

    def loss_function_decorrelation(labels, representations, weight=None):
        return minimize_variance(labels, representations, weight) + current_ratio_loss(labels, representations, weight) + decorrelation(labels, representations, weight)

    return loss_function_decorrelation if decorr else loss_function

def distance_difference(labels, representations, weight=None, unit=False):
    distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    distance_to_next_representation = safe_norm(
        representations[1:] - representations[:-1],
        axis=1
    )
    return tf.norm(distance_to_next_label - distance_to_next_representation)

def distance_difference_w_ratio_loss(train_label_min, train_label_max, lambda_val=1, unit=False, decorr=False):
    current_ratio_loss = ratio_loss(train_label_min, train_label_max, lambda_val, unit=unit)

    def loss_function(labels, representations, weight=None):
        return distance_difference(labels, representations, weight) + current_ratio_loss(labels, representations, weight)

    def loss_function_decorrelation(labels, representations, weight=None):
        return distance_difference(labels, representations, weight) + current_ratio_loss(labels, representations,
                                                                                       weight) + decorrelation(labels,
                                                                                                               representations,
                                                                                                               weight)

    return loss_function_decorrelation if decorr else loss_function