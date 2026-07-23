import tensorflow as tf
import tensorflow_probability as tfp
from .safe_norm import safe_norm

EPSILON = 1e-6

def minimize_variance(labels, representations, weight=None, unit=False):
    distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    distance_to_first_label = tf.abs(labels[1:] - labels[0])

    distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
    distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)

    combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0) + EPSILON
    combined_label_distances = tf.expand_dims(combined_label_distances, axis=-1)
    combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_first_representation],
                                                  axis=0) + EPSILON
    combined_representation_distances = tf.reshape(combined_representation_distances, combined_label_distances.shape)

    ratios = combined_representation_distances / combined_label_distances

    std_dev = tf.math.reduce_std(tf.keras.ops.log10(ratios))
    return std_dev

def augmented_pcc(labels, representations, weight=None, unit=False):
    labels_reshaped = tf.reshape(labels, (-1, 1))
    extended_representations = tf.concat([representations, labels_reshaped], axis=1)

    return 1 - tf.reduce_mean(tf.math.abs(tfp.stats.correlation(extended_representations)))

def distance_pcc(labels, representations, weight=None, unit=False):
    if unit:
        distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
        distance_to_first_label = tf.abs(labels[1:] - labels[0])

        distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
        distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)

        combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0) + EPSILON
        combined_label_distances = tf.expand_dims(combined_label_distances, axis=-1)
        combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_first_representation],
                                                      axis=0) + EPSILON
        combined_representation_distances = tf.reshape(combined_representation_distances, combined_label_distances.shape)

        return 1 - tfp.stats.correlation(combined_label_distances, combined_representation_distances)
    else:
        distance_to_next_label = tf.expand_dims(safe_norm(labels[1:] - labels[:-1], axis=1) + EPSILON, axis=-1)
        distance_to_next_representation = tf.expand_dims(
            safe_norm(representations[1:] - representations[:-1], axis=1) + EPSILON, axis=-1)

        return 1 - tfp.stats.correlation(distance_to_next_label, distance_to_next_representation)

def maximize_entropy(labels, representations, weight=None, unit=False):

    if unit:
        distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
        distance_to_first_label = tf.abs(labels[1:] - labels[0])
        distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=-1)
        distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)

        combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0) + EPSILON
        combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_first_representation],
                                                      axis=0) + EPSILON
        combined_representation_distances = tf.reshape(combined_representation_distances, combined_label_distances.shape)

        ratios = combined_label_distances / combined_representation_distances
        ratios = ratios / tf.reduce_sum(ratios)
        return tf.reduce_sum(ratios * tf.math.log(ratios)) - tf.cast(tf.math.log(1 / tf.size(ratios)), dtype=tf.float32)
    else:
        distance_to_next_label = safe_norm(labels[1:] - labels[:-1], axis=1) + EPSILON
        distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1) + EPSILON

        ratios = distance_to_next_representation / distance_to_next_label
        ratios = ratios / tf.reduce_sum(ratios)
        return tf.reduce_sum(ratios * tf.math.log(ratios)) - tf.cast(tf.math.log(1 / tf.cast(tf.size(ratios), tf.float32)), dtype=tf.float32)