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

    return centroids

def spherical_kmeans(vectors, num_classes, max_iterations=100, tolerance=1e-5):
    vectors = tf.convert_to_tensor(vectors, dtype=tf.float32)

    normalized_vectors = tf.math.l2_normalize(vectors, axis=1)
    n = tf.shape(vectors)[0]

    indices = tf.random.shuffle(tf.range(n))[:num_classes]
    centroids = tf.gather(normalized_vectors, indices)

    previous_labels = tf.fill([n], -1)

    for _ in tf.range(max_iterations):
        similarities = tf.matmul(
            normalized_vectors,
            centroids,
            transpose_b=True
        )

        labels = tf.argmax(
            similarities,
            axis=1,
            output_type=tf.int32
        )

        # Sum vectors within each cluster
        new_centroids = tf.math.unsorted_segment_sum(
            normalized_vectors,
            labels,
            num_classes
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

    return centroids

@tf.function(reduce_retracing=True)
def middle_sum(class_gradients, num_classes, conflict_only=False):
    return tf.vectorized_map(
        lambda i: gi_minus_sum_proj_gj(
            class_gradients, i, num_classes, conflict_only=conflict_only
        ),
        tf.range(num_classes)
    )

@tf.function(reduce_retracing=True)
def gi_minus_sum_proj_gj(vectors, index, num_classes, conflict_only=False):
    current = vectors[index]

    projections = tf.vectorized_map(
        lambda i: project(current, vectors[i]),
        tf.range(num_classes)
    )

    if conflict_only:
        similarities = tf.linalg.matvec(vectors, current) / (
                tf.norm(vectors, axis=1) * tf.norm(current) + 1e-12
        )

        conflict_mask = tf.cast(similarities < 0, vectors.dtype)

        sum_projections = tf.reduce_sum(
            projections * conflict_mask[:, None],
            axis=0
        )
    else:
        sum_projections = tf.reduce_sum(projections, axis=0) - current

    return current - sum_projections

@tf.function(reduce_retracing=True)
def two_fists_one_heart(class_gradients, num_classes):
    return tf.reduce_sum(middle_sum(class_gradients, num_classes), axis=0)

@tf.function(reduce_retracing=True)
def ts_mof(class_gradients, num_classes):
    middle_sums = middle_sum(class_gradients, num_classes, conflict_only=True)
    existence = tf.cast(class_gradients != 0, class_gradients.dtype)

    return tf.reduce_sum(middle_sums * existence, axis=0) / (tf.reduce_sum(existence, axis=0) + 1e-8)

def end_sum(class_gradients, num_classes):
    end_sums = tf.vectorized_map(
        lambda i: gi_minus_proj_sum_gj(
            class_gradients, i, conflict_only=True, cosine_weighted=False
        ),
        tf.range(num_classes)
    )
    return tf.reduce_sum(end_sums, axis=0)

def end_sum_w_cosine(class_gradients, num_classes):
    end_sums = tf.vectorized_map(
        lambda i: gi_minus_proj_sum_gj(
            class_gradients, i, conflict_only=True, cosine_weighted=True
        ),
        tf.range(num_classes)
    )
    return tf.reduce_sum(end_sums, axis=0)


@tf.function(reduce_retracing=True)
def gi_minus_proj_sum_gj(vectors, index, conflict_only=False, cosine_weighted=False):
    current = vectors[index]

    if conflict_only:
        similarities = tf.linalg.matvec(vectors, current) / (
                tf.norm(vectors, axis=1) * tf.norm(current) + 1e-12
        )

        conflict_mask = tf.cast(similarities < 0, vectors.dtype)

        if cosine_weighted:
            sum_others = tf.reduce_sum(
                vectors * conflict_mask[:, None] * similarities[:, None] * -1,
                axis=0
            )
        else:
            sum_others = tf.reduce_sum(
                vectors * conflict_mask[:, None],
                axis=0
            )

        projection = project(current, sum_others)
    else:
        sum_others = tf.reduce_sum(vectors, axis=0) - current
        projection = project(current, sum_others)

    return current - projection


def conflict_reduction_builder(
    binning_method='label',
    conflict_reduction_method=None,
    num_classes=3,
    normalize_before_conflict_reduction=False
):
    def function(
        x_batch,
        y_reg_batch,
        y_pseudo_class_batch,
        model,
        loss_fn
    ):
        @tf.function(reduce_retracing=True)
        def get_sample_gradient(inputs):
            x, y = inputs
            with tf.GradientTape() as tape:
                prediction = model(x[None, :], training=False)
                sample_loss = loss_fn(tf.reshape(y, [1, -1]), prediction)

            gradient = tape.gradient(sample_loss, model.trainable_variables)
            flattened_gradient = tf.concat(
                [
                    tf.reshape(g, [-1])
                    for g in gradient
                    if g is not None
                ],
                axis=0
            )

            return flattened_gradient, sample_loss

        per_sample_gradients, per_sample_loss = tf.vectorized_map(
            get_sample_gradient,
            (x_batch, y_reg_batch),
        )

        batch_loss = tf.reduce_mean(per_sample_loss)

        original_gradient = tf.reduce_mean(per_sample_gradients, axis=0)
        if binning_method == 'label':
            binned_centroids = get_class_centroids(per_sample_gradients, y_pseudo_class_batch, num_classes)
        elif binning_method == 'clustering':
            binned_centroids = spherical_kmeans(per_sample_gradients, num_classes)
        else:
            raise NotImplementedError

        if normalize_before_conflict_reduction:
            binned_centroids = tf.linalg.l2_normalize(binned_centroids, axis=1)

        if conflict_reduction_method is None:
            final_gradient = original_gradient
        else:
            final_gradient = conflict_reduction_method(binned_centroids, num_classes)
            final_gradient = tf.linalg.l2_normalize(final_gradient) * tf.norm(original_gradient)

        return (
            batch_loss,
            original_gradient,
            final_gradient,
            binned_centroids
        )

    return function