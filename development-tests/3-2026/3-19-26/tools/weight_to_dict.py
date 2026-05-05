import tensorflow as tf

def create_weight_tensor_fast(y_true: tf.Tensor, weight_dict) -> tf.Tensor:
    """
    Creates a tensor of weights corresponding to the values in y_true based on the provided weight_dict.

    Args:
        y_true (tf.Tensor): The tensor containing the ground truth labels.
        weight_dict (Dict[float, float], optional): A dictionary mapping label values to their corresponding weights.

    Returns:
        tf.Tensor: A tensor of weights corresponding to y_true.
    """
    if weight_dict is None:
        return tf.ones_like(y_true, dtype=tf.float32)
    # Convert the weight dictionary to sorted tensors
    unique_labels = tf.constant(sorted(weight_dict.keys()), dtype=tf.float32)
    weight_values = tf.constant([weight_dict[label] for label in sorted(weight_dict.keys())], dtype=tf.float32)
    # Flatten y_true if it has more than one dimension and cast to same dtype as unique_labels
    y_true_flat = tf.cast(tf.reshape(y_true, [-1]), dtype=unique_labels.dtype)
    # Use tf.searchsorted to find the indices of y_true in unique_labels
    indices = tf.searchsorted(unique_labels, y_true_flat, side='left')
    # Handle the case where the search goes out of bounds
    num_unique_labels = tf.shape(unique_labels)[0]
    num_unique_labels = tf.cast(num_unique_labels, indices.dtype)  # Ensure the type matches
    indices = tf.clip_by_value(indices, 0, num_unique_labels - 1)
    # Gather the weights using the found indices
    y_true_weights = tf.gather(weight_values, indices)
    # Reshape the weights to match the original y_true shape if necessary
    y_true_weights = tf.reshape(y_true_weights, tf.shape(y_true))

    return y_true_weights