import tensorflow as tf
import keras
from keras import layers
from imbal.regression import tsne_visualization, plot_true_vs_predictions
from matplotlib import pyplot as plt
import numpy as np

ALPHA = 1
LEARNING_RATE = 1e-5
EPOCHS = 2000
REPRESENTATION_LAYER_INDEX = -2
MANUALLY_SORT_EVERY_BATCH = True

tf.config.run_functions_eagerly(True)

def build_sep_ec_model(layer_dimensions, representation_layer_dimension):
    inputs = keras.Input(shape=(x_train.shape[1],))

    x = inputs
    for index, num_units in enumerate(layer_dimensions):
        x = layers.Dense(num_units, activation='relu', kernel_initializer='he_normal')(x)
        if index == len(layer_dimensions) - 1:
            x = layers.Flatten()(x)

    representation_layer = layers.Dense(representation_layer_dimension)(x)
    output_layer = layers.Dense(1, kernel_initializer='he_normal')(representation_layer)

    model = keras.Model(inputs=inputs, outputs=[output_layer, representation_layer], name="SEP_EC")
    return model

# 2. Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

def compute_regression_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

# def compute_squared_representation_loss(labels, representations):
#     EPSILON = 1e-9
#
#     label_differences = labels[:, None, :] - labels[None, :, :]
#     representation_differences = representations[:, None, :] - representations[None, :, :]
#
#     label_distances = tf.linalg.norm(label_differences, axis=-1) + EPSILON
#     representation_distances = tf.linalg.norm(representation_differences, axis=-1) + EPSILON
#
#     ratios = label_distances / representation_distances
#
#     n = tf.shape(labels)[0]
#     mask = tf.linalg.band_part(tf.ones((n, n), dtype=ratios.dtype), 0, -1) - tf.eye(n, dtype=ratios.dtype)
#     valid_ratios = tf.boolean_mask(ratios, mask > 0)
#     std_dev = tf.math.reduce_std(valid_ratios)
#
#     return std_dev

def compute_representation_loss(labels, representations):
    EPSILON = 1e-9

    # # Attempt 1
    # labels_shifted = tf.roll(labels, shift=-1, axis=0)
    # representations_shifted = tf.roll(representations, shift=-1, axis=0)
    #
    # label_distances = tf.linalg.norm(labels - labels_shifted, axis=1) + EPSILON
    # representation_distances = tf.linalg.norm(representations - representations_shifted, axis=1) + EPSILON
    #
    # ratios = label_distances / representation_distances
    #
    # std_dev = tf.math.reduce_std(ratios)
    # return std_dev

    # Attempt 2
    distance_to_next_label = tf.linalg.norm(labels[1:] - labels[:-1], axis=1)
    distance_to_last_label = tf.linalg.norm(labels[1:-1] - labels[-1], axis=-1)
    distance_to_first_label = tf.linalg.norm(labels[1:] - labels[0], axis=-1)

    distance_to_next_representation = tf.linalg.norm(representations[1:] - representations[:-1], axis=1)
    distance_to_last_representation = tf.linalg.norm(representations[1:-1] - representations[-1], axis=-1)
    distance_to_first_representation = tf.linalg.norm(representations[1:] - representations[0], axis=-1)

    combined_label_distances = tf.concat([distance_to_next_label, distance_to_last_label, distance_to_first_label], axis=0) + EPSILON
    combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_last_representation, distance_to_first_representation], axis=0) + EPSILON

    ratios = combined_label_distances / combined_representation_distances

    # n = tf.shape(ratios)[0]
    # endpoint_weight = (n - 1)/3
    # weights = tf.ones_like(ratios)
    # weights = tf.concat([weights[:-1], [endpoint_weight]], axis=0)
    #
    # weighted_mean = tf.reduce_sum(weights * ratios) / tf.reduce_sum(weights)
    # weighted_var = tf.reduce_sum(weights * tf.square(ratios - weighted_mean)) / tf.reduce_sum(weights)
    #
    # return weighted_var

    return tf.math.reduce_std(tf.math.log(ratios))

def combined_loss(
    labels,
    predictions,
    representations,
    alpha=1
):
    regression_loss = compute_regression_loss(labels, predictions)
    representation_loss = compute_representation_loss(labels, representations)
    total_loss = regression_loss*0 + representation_loss * alpha
    return total_loss, regression_loss, representation_loss

# 5. Dummy data — two label tensors, one per output head
x_train = tf.clip_by_value(tf.random.normal((1000, 2)), clip_value_min=-2.5, clip_value_max=2.5)
y_reg_train = tf.reshape(tf.linalg.norm(x_train, axis=1), (-1, 1))

train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_reg_train)
).shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(1000)

model = build_sep_ec_model(
    layer_dimensions=[128, 128, 64, 64, 32, 32],
    representation_layer_dimension=2
)

print(model.summary())

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions, representations = model(x, training=True)

        total_loss, regression_loss, representation_loss = combined_loss(
            y,
            predictions,
            representations,
            alpha=ALPHA
        )

    gradients = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return total_loss, regression_loss, representation_loss


for epoch in range(EPOCHS):
    print(f"\nStart of epoch {epoch + 1}")

    for step, (x_batch, y_reg_batch) in enumerate(train_dataset):

        if MANUALLY_SORT_EVERY_BATCH:
            sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
            x_batch = tf.gather(x_batch, sort_indices)
            y_reg_batch = tf.gather(y_reg_batch, sort_indices)

        loss, reg_loss, rep_loss = train_step(x_batch, y_reg_batch)
        if step % 10 == 0:
            print(
                f"  Step {step:3d} | "
                f"Total: {float(loss):.4f} | "
                f"Regression: {float(reg_loss):.4f} | "
                f"Representation: {float(rep_loss):.4f}"
            )

sort_indices = tf.argsort(tf.squeeze(y_reg_train))
y_reg_train = tf.gather(y_reg_train, sort_indices)
x_train = tf.gather(x_train, sort_indices)

predictions, representations = model.predict(x_train)

labels_shifted = tf.roll(y_reg_train, shift=-1, axis=0)
representations_shifted = tf.roll(representations, shift=-1, axis=0)

label_distances = tf.linalg.norm(y_reg_train - labels_shifted, axis=1) + 1e-9
representation_distances = tf.linalg.norm(representations - representations_shifted, axis=1) + 1e-9

ratios = label_distances / representation_distances

print(ratios.numpy())
print(tf.math.reduce_std(ratios))
print(representations[:50])
print(model.layers)

x_train = x_train.numpy()
y_reg_train = y_reg_train.numpy()

plot_true_vs_predictions(
    y_reg_train,
    predictions
)

def plot_representation_space(dim_one_index, dim_two_index):
    min_rep_x = np.min(representations[:, dim_one_index])
    max_rep_x = np.max(representations[:, dim_one_index])
    min_rep_y = np.min(representations[:, dim_two_index])
    max_rep_y = np.max(representations[:, dim_two_index])
    distance_rep_x = max_rep_x - min_rep_x
    distance_rep_y = max_rep_y - min_rep_y
    if distance_rep_x > distance_rep_y:
        diff = distance_rep_x - distance_rep_y
        half_diff = diff / 2
        min_rep_y -= half_diff
        max_rep_y += half_diff
    else:
        diff = distance_rep_y - distance_rep_x
        half_diff = diff / 2
        min_rep_x -= half_diff
        max_rep_x += half_diff

    min_rep_x -= 1
    max_rep_x += 1
    min_rep_y -= 1
    max_rep_y += 1

    plt.scatter(
        representations[:, dim_one_index].reshape(-1),
        representations[:, dim_two_index].reshape(-1),
        c=y_reg_train,
        cmap='jet',
        alpha=0.5
    )
    plt.xlim(min_rep_x, max_rep_x)
    plt.ylim(min_rep_y, max_rep_y)
    plt.colorbar()
    plt.show()

plot_representation_space(0, 1)
# plot_representation_space(0, 2)
# plot_representation_space(0, 3)
# plot_representation_space(1, 2)
# plot_representation_space(1, 3)
# plot_representation_space(2, 3)

tsne_visualization(
    model=model,
    data=x_train,
    labels=y_reg_train,
    representation_layer_index=REPRESENTATION_LAYER_INDEX,
    gradient='jet'
)

from sklearn.linear_model import LinearRegression

lin_reg_model = LinearRegression()
lin_reg_model.fit(predictions.reshape(-1, 1), y_reg_train.reshape(-1))

adjusted_predictions = lin_reg_model.predict(predictions.reshape(-1, 1))


plot_true_vs_predictions(
    y_reg_train,
    adjusted_predictions
)