import tensorflow as tf


@tf.function(reduce_retracing=True)
def cosine_similarity(a, b):
    denom = tf.norm(a) * tf.norm(b)
    if denom < 10e-6:
        return tf.constant(0.0)
    return tf.reduce_sum(a * b) / denom

@tf.function(reduce_retracing=True)
def project(vec_1, vec_2):
    denominator = tf.reduce_sum(vec_2 * vec_2)

    return tf.math.divide_no_nan(
        tf.reduce_sum(vec_1 * vec_2),
        denominator
    ) * vec_2

@tf.function(reduce_retracing=True)
def get_class_centroids(gradients, labels, num_classes):
    labels = tf.reshape(labels, [-1])

    centroids = tf.math.unsorted_segment_sum(
        data=gradients,
        segment_ids=labels,
        num_segments=num_classes
    )

    normalized_centroids = tf.math.l2_normalize(
        centroids,
        axis=1
    )

    return centroids, normalized_centroids

def spherical_kmeans(vectors, labels, num_classes, max_iterations=100, tolerance=1e-5):
    vectors = tf.convert_to_tensor(vectors, dtype=tf.float32)

    # Normalize all input vectors
    normalized_vectors = tf.math.l2_normalize(vectors, axis=1)

    n = tf.shape(vectors)[0]

    # Randomly initialize centroids from input vectors
    indices = tf.random.shuffle(tf.range(n))[:num_classes]
    centroids = tf.gather(normalized_vectors, indices)

    previous_labels = tf.fill([n], -1)

    for _ in tf.range(max_iterations):
        # Cosine similarity between every vector and every centroid
        similarities = tf.matmul(
            normalized_vectors,
            centroids,
            transpose_b=True
        )

        # Assign each vector to its most similar centroid
        labels = tf.argmax(
            similarities,
            axis=1,
            output_type=tf.int32
        )

        # Calculate new centroids
        new_centroids = tf.math.unsorted_segment_mean(
            normalized_vectors,
            labels,
            num_classes
        )

        # Normalize centroid directions
        new_centroids = tf.math.l2_normalize(
            new_centroids,
            axis=1
        )

        # Preserve old centroids for empty clusters
        counts = tf.math.unsorted_segment_sum(
            tf.ones([n], dtype=tf.float32),
            labels,
            num_classes
        )

        new_centroids = tf.where(
            counts[:, None] > 0,
            new_centroids,
            centroids
        )

        converged = tf.reduce_all(labels == previous_labels)

        centroids = new_centroids
        previous_labels = labels

        if converged:
            break

    normalized_centroids = tf.math.l2_normalize(
        centroids,
        axis=1
    )

    return centroids, normalized_centroids

@tf.function(reduce_retracing=True)
def conflict_only_end_sum_with_cosine(class_gradients, normalized_class_gradients, num_classes):
    return tf.vectorized_map(
        lambda i: gi_minus_proj_sum_cos_times_gj(
            normalized_class_gradients, i
        ),
        tf.range(num_classes)
    )

@tf.function(reduce_retracing=True)
def conflict_only_end_sum(class_gradients, normalized_class_gradients, num_classes):
    return tf.vectorized_map(
        lambda i: gi_minus_proj_sum_gj(
            normalized_class_gradients, i
        ),
        tf.range(num_classes)
    )

@tf.function(reduce_retracing=True)
def gi_minus_proj_sum_cos_times_gj(vectors, index):
    current = vectors[index]

    # Compute cosine similarities against every vector
    similarities = tf.linalg.matvec(vectors, current) / (
        tf.norm(vectors, axis=1) * tf.norm(current) + 1e-12
    )

    # Exclude the current vector
    similarities = tf.tensor_scatter_nd_update(
        similarities,
        [[index]],
        [0.0]
    )

    # Keep only negative similarities
    similarities = tf.minimum(similarities, 0.0)

    # Weighted sum of conflicting vectors
    sum_conflicting = tf.reduce_sum(
        vectors * similarities[:, None],
        axis=0
    )

    # Project sum_conflicting onto current
    projected_vector = (
        tf.reduce_sum(current * sum_conflicting)
        / (tf.reduce_sum(current * current) + 1e-12)
    ) * current

    return current - projected_vector

@tf.function(reduce_retracing=True)
def gi_minus_proj_sum_gj(vectors, index):
    current = vectors[index]

    # Compute cosine similarities against every vector
    similarities = tf.linalg.matvec(vectors, current) / (
        tf.norm(vectors, axis=1) * tf.norm(current) + 1e-12
    )

    # Exclude the current vector
    similarities = tf.tensor_scatter_nd_update(
        similarities,
        [[index]],
        [0.0]
    )

    conflict_mask = tf.cast(similarities < 0, vectors.dtype)
    sum_conflicting = tf.reduce_sum(
        vectors * conflict_mask[:, None],
        axis=0
    )

    # Project sum_conflicting onto current
    projected_vector = (
        tf.reduce_sum(current * sum_conflicting)
        / (tf.reduce_sum(current * current) + 1e-12)
    ) * current

    return current - projected_vector

@tf.function(reduce_retracing=True)
def middle_sum(class_gradients, normalized_class_gradients, num_classes):
    return tf.vectorized_map(
        lambda i: gi_minus_sum_proj_gj(
            normalized_class_gradients, i, num_classes
        ),
        tf.range(num_classes)
    )

@tf.function(reduce_retracing=True)
def conflict_only_middle_sum(class_gradients, normalized_class_gradients, num_classes):
    return tf.vectorized_map(
        lambda i: conflict_only_gi_minus_sum_proj_gj(
            normalized_class_gradients, i, num_classes
        ),
        tf.range(num_classes)
    )

@tf.function(reduce_retracing=True)
def two_fists_one_heart(class_gradients, normalized_class_gradients, num_classes):
    return middle_sum(class_gradients, normalized_class_gradients, num_classes)

@tf.function(reduce_retracing=True)
def ts_mof(class_gradients, normalized_class_gradients, num_classes):
    middle_sums = conflict_only_middle_sum(class_gradients, normalized_class_gradients, num_classes)

    existence = tf.cast(class_gradients != 0, class_gradients.dtype)

    return tf.reduce_sum(middle_sums * existence, axis=0) / (tf.reduce_sum(existence, axis=0) + 1e-8)


@tf.function(reduce_retracing=True)
def gi_minus_sum_proj_gj(vectors, index, num_classes):
    current = vectors[index]

    projections = tf.vectorized_map(
        lambda i: project(current, vectors[i]),
        tf.range(num_classes)
    )

    sum_projections = tf.reduce_sum(projections, axis=0) - current

    return current - sum_projections

@tf.function(reduce_retracing=True)
def conflict_only_gi_minus_sum_proj_gj(vectors, index, num_classes):
    current = vectors[index]

    projections = tf.vectorized_map(
        lambda i: project(current, vectors[i]),
        tf.range(num_classes)
    )

    # Compute cosine similarities against every vector
    similarities = tf.linalg.matvec(vectors, current) / (
        tf.norm(vectors, axis=1) * tf.norm(current) + 1e-12
    )

    # Exclude the current vector
    similarities = tf.tensor_scatter_nd_update(
        similarities,
        [[index]],
        [0.0]
    )

    conflict_mask = tf.cast(similarities < 0, vectors.dtype)

    sum_projections = tf.reduce_sum(projections * conflict_mask[:, None], axis=0) - current

    return current - sum_projections