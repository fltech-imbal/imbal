import tensorflow as tf
import tensorflow_probability as tfp
from .safe_norm import safe_norm
from math import pi

EPSILON = 1e-6
LABEL_RANGE = None

def minimize_variance(label_distances, representation_distances):
    ratios = representation_distances / (label_distances + EPSILON)
    std_dev = tf.math.reduce_std(tf.keras.ops.log(ratios))
    return std_dev

def distance_pcc(label_distances, representation_distances):
    return 1 - tfp.stats.correlation(
        label_distances,
        representation_distances,
        sample_axis=0,
        event_axis=1
    )

def cauchy_schwartz(label_distances, representation_distances):
    a = label_distances
    b = representation_distances
    return tf.reduce_sum(tf.multiply(a, a)) * tf.reduce_sum(tf.multiply(b, b)) - tf.reduce_sum(tf.multiply(a, b))**2

def maximize_entropy(label_distances, representation_distances):
    ratios = representation_distances / (label_distances + EPSILON)
    ratios = ratios / tf.reduce_sum(ratios)
    return tf.reduce_sum(ratios * tf.math.log(ratios)) - tf.cast(tf.math.log(1 / tf.size(ratios)), dtype=tf.float32)

def distance_difference(label_distances, representation_distances, alpha):
    return tf.reduce_mean(tf.abs(representation_distances - alpha * label_distances))

def cosine_similarity(label_distances, representation_distances):
    first_vectors_normalized = tf.linalg.l2_normalize(representation_distances[:-1], axis=1, epsilon=1e-8)
    second_vectors_normalized = tf.linalg.l2_normalize(representation_distances[1:], axis=1, epsilon=1e-8)

    similarities = tf.reduce_sum(first_vectors_normalized * second_vectors_normalized, axis=1)
    return 1 - tf.reduce_mean(similarities)

def enforced_cosine(label_distances, representations):
    representation_differences = representations[1:] - representations[:-1]
    first_vectors_normalized = tf.linalg.l2_normalize(representation_differences[:-1], axis=1, epsilon=1e-8)
    second_vectors_normalized = tf.linalg.l2_normalize(representation_differences[1:], axis=1, epsilon=1e-8)

    label_difference = (label_distances[:-1] + label_distances[1:])
    normalized_difference_length = label_difference / LABEL_RANGE

    ideal_cosines = tf.reshape(tf.math.cos(normalized_difference_length * 3), (-1,))

    similarities = tf.reduce_sum(first_vectors_normalized * second_vectors_normalized, axis=1)
    return tf.reduce_mean(tf.square(ideal_cosines - similarities))


def get_distance_pairs(
    labels,
    representations,
    include_global_anchor
):
    pairwise_label_distances = tf.abs(labels[1:] - labels[:-1])
    pairwise_label_distances = tf.reshape(pairwise_label_distances, [-1, 1])
    pairwise_representation_distances = safe_norm(representations[1:] - representations[:-1], axis=1)
    pairwise_representation_distances = tf.reshape(pairwise_representation_distances, [-1, 1])

    if include_global_anchor:
        anchor_label_distances = tf.abs(labels[1:] - labels[0])
        anchor_label_distances = tf.reshape(pairwise_label_distances, [-1, 1])
        anchor_representation_distances = safe_norm(representations[1:] - representations[0], axis=1)
        anchor_representation_distances = tf.reshape(pairwise_representation_distances, [-1, 1])
        pairwise_label_distances = tf.concat([pairwise_label_distances, anchor_label_distances], axis=0)
        pairwise_representation_distances = tf.concat([pairwise_representation_distances, anchor_representation_distances], axis=0)

    return pairwise_label_distances, pairwise_representation_distances


def ratio_loss(train_label_min, train_label_max, unit=False):
    def loss(label_distances, representation_distances, fixed_ratio=1):
        if unit:
            train_label_range = train_label_max - train_label_min
            ideal_ratio = 1.8 / train_label_range
        else:
            ideal_ratio = fixed_ratio
        ideal_ratio = tf.cast(ideal_ratio, tf.float32)

        ratio = tf.reduce_sum(representation_distances) / (tf.reduce_sum(label_distances) + 1e-12)
        ratio = tf.clip_by_value(ratio, 1e-7, 1e7)

        loss_value = (tf.keras.ops.log10(ratio) - tf.keras.ops.log10(ideal_ratio)) ** 2

        return loss_value

    return loss

def learnable_ratio_loss(label_distances, representation_distances, learned_ratio):
    ratio = tf.reduce_sum(representation_distances) / (tf.reduce_sum(label_distances) + 1e-12)
    ratio = tf.clip_by_value(ratio, 1e-7, 1e7)

    ratio_loss_value = (tf.keras.ops.log(ratio) - tf.keras.ops.log(learned_ratio))**2

    return ratio_loss_value

def decorrelation(representations, eps=1e-6):
    mean = tf.reduce_mean(representations, axis=0, keepdims=True)
    centered = representations - mean

    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        tf.shape(representations)[0] - 1, representations.dtype
    )

    std = tf.sqrt(tf.linalg.diag_part(cov) + eps)
    corr = cov / (std[:, None] * std[None, :] + eps)

    corr = tf.where(tf.math.is_finite(corr), corr, tf.zeros_like(corr))

    n = tf.shape(corr)[0]
    mask = tf.ones_like(corr) - tf.eye(n, dtype=corr.dtype)

    denominator = tf.reduce_sum(mask)
    return tf.math.divide_no_nan(tf.reduce_sum(tf.abs(corr) * mask), denominator)

def loss_function_builder(
    loss,
    train_label_min,
    train_label_max,
    fixed_ratio=0,
    learnable_ratio=False,
    ratio_max=5,
    ratio_lambda=1,
    unit_representation=False,
    use_decorrelation=False,
    decorrelation_lambda=1,
    include_global_anchor=False,
    include_ratio=False,
    use_raw_representations=False,
):
    global LABEL_RANGE
    LABEL_RANGE = train_label_max - train_label_min

    if fixed_ratio != 0 and learnable_ratio:
        raise RuntimeError("Cannot have fixed and learnable ratio")

    if fixed_ratio:
        ratio_loss_function = ratio_loss(
            train_label_min,
            train_label_max,
            unit=unit_representation
        )
    elif learnable_ratio > 0:
        ratio_loss_function = learnable_ratio_loss
    else:
        ratio_loss_function = lambda x, y, z: 0

    if use_decorrelation:
        decorrelation_loss_function = decorrelation
    else:
        decorrelation_loss_function = lambda x : 0

    def constructed_loss(
        labels,
        representations
    ):
        ratio = fixed_ratio
        if learnable_ratio:
            scalar = representations[0, -1]
            representations = representations[:, :-1]
            ratio = (ratio_max - 1 / ratio_max) * tf.sigmoid(scalar) + 1 / ratio_max

        label_distance_pairs, representation_distance_pairs = get_distance_pairs(
            labels,
            representations,
            include_global_anchor=include_global_anchor
        )

        if include_ratio:
            if use_raw_representations:
                rep_loss_value = loss(label_distance_pairs, representations, ratio)
            else:
                rep_loss_value = loss(label_distance_pairs, representation_distance_pairs, ratio)
        else:
            if use_raw_representations:
                rep_loss_value = loss(label_distance_pairs, representations)
            else:
                rep_loss_value = loss(label_distance_pairs, representation_distance_pairs)
        ratio_loss_value = ratio_loss_function(label_distance_pairs, representation_distance_pairs, ratio)
        decorrelation_loss_value = decorrelation_loss_function(representations)
        # print()
        # print(rep_loss_value, ratio_loss_value, decorrelation_loss_value)
        # print(rep_loss_value + ratio_loss_value * ratio_lambda + decorrelation_loss_value * decorrelation_lambda)
        return rep_loss_value + ratio_loss_value * ratio_lambda + decorrelation_loss_value * decorrelation_lambda

    return constructed_loss