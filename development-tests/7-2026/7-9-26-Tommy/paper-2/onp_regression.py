"""
Import packages
"""
import imbal
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tools import FitType, load_onp_data, plot_representation_space, build_sep_ec_model, generate_weights

# Axes of Exploration
# - Representation loss used
# - regular/balanced
# - joint/freeze/tune
# - num of representation layers
# - constrain ratio? y/n

"""
Set script parameters
"""

# tf.config.run_functions_eagerly(True)

FIGURE_NAME = 'special_augmented_pcc_joint_5'
LEARNING_RATE = 2e-4
FIT = FitType.REGULAR
SINGLE_WEIGHT_ALPHA = 0
VALIDATION_DATA = True

FIT_MODE = 'joint'
MSE_LAMBDA = 1
REPRESENTATION_LAMBDA = 1
UNIT_REPRESENTATION = False
RATIO_CONSTRAIN = False
JUST_RATIO = False
EXTRA_REGRESSOR_LAYERS = True
PCC_LAMBDA = 0

MANUALLY_SORT_EVERY_BATCH = True
BATCH_SIZE = 4096
EARLY_STOPPING_PATIENCE = 100
EPOCHS = 50000
SECOND_STAGE_EPOCHS = 50000

DATA_PATH = 'onp_normalized_log_labels'

"""
Load data
"""

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_onp_data(
    f"cleaned-ONP-data/{DATA_PATH}",
)

# bandwidth = imbal.regression.fit_kde(y_train)
#
# imbal.regression.plot_kde_1d(
#     y_train,
#     bandwidth,
#     bin_count=64
# )

"""
Build model
"""

LAYER_DIMS = [128, 128, 128, 64, 64, 64, 32, 32, 2]

model = build_sep_ec_model(
    x_train.shape,
    LAYER_DIMS,
    unit=UNIT_REPRESENTATION,
    extra_regressor_layers=EXTRA_REGRESSOR_LAYERS,
)
model.summary()
"""
Generate sample densities
"""
(x_train, y_train, sample_weights), val_data = generate_weights(
    x_train,
    y_train,
    x_val,
    y_val,
    weight_alpha=SINGLE_WEIGHT_ALPHA,
    combine_validation=not VALIDATION_DATA,
)


x_val, y_val, w_val = val_data

sort_indices = tf.argsort(tf.squeeze(y_val))
x_val = tf.gather(x_val, sort_indices)
y_val = tf.gather(y_val, sort_indices)
w_val = tf.gather(w_val, sort_indices)

def safe_norm(x, axis):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + 1e-12)

def pcc(y_true, y_pred):
    y_true = tf.reshape(y_true, tf.shape(y_pred))

    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    return 1 - (
        tf.reduce_sum(y_true_centered * y_pred_centered) /
        (tf.norm(y_true_centered) * tf.norm(y_pred_centered))
    )

def compute_regression_loss(y_true, y_pred, sample_weight=None):
    pcc_value = pcc(y_true, y_pred)*PCC_LAMBDA

    if sample_weight is None:
        sample_weight = tf.ones_like(y_true)
    else:
        sample_weight = tf.reshape(sample_weight, y_true.shape)

    return tf.reduce_sum(tf.square(y_true - y_pred) * sample_weight) / tf.reduce_sum(sample_weight) + pcc_value

def compute_representation_loss(labels, representations, weight=None):
    EPSILON = 1e-4

    # # Attempt 1 - Minimize variance
    # distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    # distance_to_first_label = tf.abs(labels[1:] - labels[0])
    #
    # distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
    # distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)
    #
    # combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0) + EPSILON
    # combined_label_distances = tf.expand_dims(combined_label_distances, axis=-1)
    # combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_first_representation], axis=0) + EPSILON
    # combined_representation_distances = tf.reshape(combined_representation_distances, combined_label_distances.shape)
    #
    # ratios = combined_representation_distances / combined_label_distances
    #
    # std_dev = tf.math.reduce_std(tf.keras.ops.log10(ratios))
    # return std_dev

    # # Attempt 2
    # distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    # distance_to_last_label = tf.abs(labels[1:-1] - labels[-1])
    # distance_to_first_label = tf.abs(labels[1:] - labels[0])
    #
    # distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
    # distance_to_last_representation = safe_norm(representations[1:-1] - representations[-1], axis=-1)
    # distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)
    #
    # combined_label_distances = tf.concat([distance_to_next_label, distance_to_last_label, distance_to_first_label], axis=0) + EPSILON
    # combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_last_representation, distance_to_first_representation], axis=0) + EPSILON
    # combined_representation_distances = tf.reshape(combined_representation_distances, combined_label_distances.shape)
    #
    # return tf.reduce_mean(tf.square(combined_representation_distances - combined_label_distances))

    # # Attempt 3 - Augmented PCC of representations
    # labels_reshaped = tf.reshape(labels, (-1, 1))
    # extended_representations = tf.concat([representations, labels_reshaped], axis=1)
    #
    # return 1 - tf.reduce_mean(tf.math.abs(tfp.stats.correlation(extended_representations)))

    # Attempt 3.5 - Augmented PCC of each representation dimension
    labels = tf.reshape(labels, [-1])
    labels_reshaped = tf.tile(labels[:, tf.newaxis], [1, tf.shape(representations)[1]])

    pcc = tfp.stats.correlation(representations, labels_reshaped, sample_axis=0, event_axis=None)

    return 1.0 - tf.reduce_mean(tf.abs(pcc))

    # # Attempt 4 - PCC(label dist, rep dist)
    # distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    # distance_to_first_label = tf.abs(labels[1:] - labels[0])
    #
    # distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
    # distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)
    #
    # combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0) + EPSILON
    # combined_label_distances = tf.expand_dims(combined_label_distances, axis=-1)
    # combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_first_representation], axis=0) + EPSILON
    # combined_representation_distances = tf.reshape(combined_representation_distances, combined_label_distances.shape)
    #
    # return 1 - tfp.stats.correlation(combined_label_distances, combined_representation_distances)

    # # Attempt 5 - Maximize entropy
    # distance_to_next_label = tf.abs(labels[1:] - labels[:-1])
    # distance_to_first_label = tf.abs(labels[1:] - labels[0])
    # distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=-1)
    # distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)
    #
    # combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0) + EPSILON
    # combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_first_representation], axis=0) + EPSILON
    # combined_representation_distances = tf.reshape(combined_representation_distances, combined_label_distances.shape)
    #
    # ratios = combined_label_distances / combined_representation_distances
    # ratios = ratios / tf.reduce_sum(ratios)
    # return tf.reduce_sum(ratios * tf.math.log(ratios)) - tf.cast(tf.math.log(1 / tf.size(ratios)), dtype=tf.float32)

    # # Attempt 6 - Maximize entropy (unit representation)
    # distance_to_next_label = tf.linalg.norm(labels[1:] - labels[:-1], axis=1) + EPSILON
    # distance_to_next_representation = tf.linalg.norm(representations[1:] - representations[:-1], axis=1) + EPSILON
    #
    # ratios = distance_to_next_representation / distance_to_next_label
    # ratios = ratios / tf.reduce_sum(ratios)
    # return tf.reduce_sum(ratios * tf.math.log(ratios)) - tf.cast(tf.math.log(1 / tf.size(ratios)), dtype=tf.float32)

    # # Attempt 7 - PCC(label dist, rep dist) (unit representation)
    # distance_to_next_label = tf.expand_dims(tf.linalg.norm(labels[1:] - labels[:-1], axis=1) + EPSILON, axis=-1)
    # distance_to_next_representation = tf.expand_dims(tf.linalg.norm(representations[1:] - representations[:-1], axis=1) + EPSILON, axis=-1)
    #
    # return 1 - tfp.stats.correlation(distance_to_next_label, distance_to_next_representation)

    # return tf.constant(0.0)

def ratio_loss(labels, representations, unit=UNIT_REPRESENTATION):

    distance_to_next_label = labels[1:] - labels[:-1]
    distance_to_first_label = labels[1:] - labels[0]

    distance_to_next_representation = safe_norm(representations[1:] - representations[:-1], axis=1)
    distance_to_first_representation = safe_norm(representations[1:] - representations[0], axis=-1)

    combined_label_distances = tf.concat([distance_to_next_label, distance_to_first_label], axis=0)
    combined_representation_distances = tf.concat([distance_to_next_representation, distance_to_first_representation], axis=0)
    ratio = tf.reduce_sum(combined_representation_distances) / (tf.reduce_sum(combined_label_distances) + 1e-12)
    ratio = tf.clip_by_value(ratio, 1e-7, 1e7)

    loss_value = tf.keras.ops.log10(ratio)**2

    return loss_value * REPRESENTATION_LAMBDA

def combined_loss(
    labels,
    predictions,
    representations,
    sample_weight=None,
    alpha=1
):
    regression_loss = compute_regression_loss(labels, predictions, sample_weight=sample_weight)
    representation_loss = compute_representation_loss(labels, representations, weight=sample_weight)
    if RATIO_CONSTRAIN:
        if JUST_RATIO:
            representation_loss = ratio_loss(labels, representations)
        else:
            representation_loss += ratio_loss(labels, representations)
    total_loss = regression_loss * MSE_LAMBDA + representation_loss * alpha
    return total_loss, regression_loss, representation_loss

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(sample_weights))

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        x_train,
        y_train,
        tf.ones_like(y_train) if FIT == FitType.REGULAR else sample_weights,
     )
).shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
second_stage_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

@tf.function
def joint_step(x, y, w, train=True):
    if MANUALLY_SORT_EVERY_BATCH:
        sort_indices = tf.argsort(tf.squeeze(y))
        x = tf.gather(x, sort_indices)
        y = tf.gather(y, sort_indices)
        w = tf.gather(w, sort_indices)

    mae = None
    with tf.GradientTape() as tape:
        predictions, representations = model(x, training=train)

        tf.debugging.check_numerics(predictions, "outputs")
        tf.debugging.check_numerics(representations, "reps")

        total_loss, regression_loss, representation_loss = combined_loss(
            y,
            predictions,
            representations,
            alpha=REPRESENTATION_LAMBDA,
            sample_weight=w
        )

        w = tf.reshape(w, y.shape)
        mae = tf.reduce_sum(tf.abs(predictions - y) * w) / tf.reduce_sum(w)

    gradients = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    if train:
        return total_loss, regression_loss, representation_loss
    else:
        return total_loss, regression_loss, representation_loss, mae

@tf.function
def representation_step(x, y, w, train=True):
    if MANUALLY_SORT_EVERY_BATCH:
        sort_indices = tf.argsort(tf.squeeze(y))
        x = tf.gather(x, sort_indices)
        y = tf.gather(y, sort_indices)
        w = tf.gather(w, sort_indices)

    mae = None
    with tf.GradientTape() as tape:
        predictions, representations = model(x, training=train)

        total_loss, regression_loss, representation_loss = combined_loss(
            y,
            predictions,
            representations,
            alpha=REPRESENTATION_LAMBDA,
            sample_weight=w
        )

        w = tf.reshape(w, y.shape)
        mae = tf.reduce_sum(tf.abs(predictions - y) * w) / tf.reduce_sum(w)

    gradients = tape.gradient(representation_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    if train:
        return total_loss, regression_loss, representation_loss
    else:
        return total_loss, regression_loss, representation_loss, mae

@tf.function
def regression_step(x, y, w, train=True):
    if MANUALLY_SORT_EVERY_BATCH:
        sort_indices = tf.argsort(tf.squeeze(y))
        x = tf.gather(x, sort_indices)
        y = tf.gather(y, sort_indices)
        w = tf.gather(w, sort_indices)

    mae = None
    with tf.GradientTape() as tape:
        predictions, representations = model(x, training=train)

        total_loss, regression_loss, representation_loss = combined_loss(
            y,
            predictions,
            representations,
            alpha=REPRESENTATION_LAMBDA,
            sample_weight=w
        )

        w = tf.reshape(w, y.shape)
        mae = tf.reduce_sum(tf.abs(predictions - y) * w) / tf.reduce_sum(w)

    gradients = tape.gradient(regression_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    if train:
        return total_loss, regression_loss, representation_loss
    else:
        return total_loss, regression_loss, representation_loss, mae



DELTA = 1e-4
best_val_metric = None
timer = 0
best_weights = None

first_stage_epochs = None
second_stage_epochs = None

if FIT_MODE == 'joint':
    for epoch in range(EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch, weights) in enumerate(train_dataset):

            loss, reg_loss, rep_loss = joint_step(x_batch, y_reg_batch, weights)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )

        if VALIDATION_DATA:
            val_loss, val_reg_loss, val_rep_loss, mae = joint_step(x_val, y_val, w_val, train=False)
            print(
                f"  Validation for epoch {epoch + 1} | "
                f"Total: {float(val_loss):.4f} | "
                f"Regression: {float(val_reg_loss):.4f} | "
                f"Representation: {float(val_rep_loss):.4f} | "
                f"MAE: {float(mae):.4f}"
            )
            if best_val_metric is None or mae <= best_val_metric - DELTA:
                best_val_metric = mae
                timer = 0
                best_weights = model.get_weights()
            else:
                timer += 1
            if timer == EARLY_STOPPING_PATIENCE:
                model.set_weights(best_weights)
                first_stage_epochs = epoch+1 - EARLY_STOPPING_PATIENCE
                break

elif FIT_MODE == 'freeze':
    for epoch in range(EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch, weights) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = representation_step(x_batch, y_reg_batch, weights)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )

        if VALIDATION_DATA:
            val_loss, val_reg_loss, val_rep_loss, mae = representation_step(x_val, y_val, w_val, train=False)
            print(
                f"  Validation for epoch {epoch + 1} | "
                f"Total: {float(val_loss):.4f} | "
                f"Regression: {float(val_reg_loss):.4f} | "
                f"Representation: {float(val_rep_loss):.4f} | "
                f"MAE: {float(mae):.4f}"
            )
            if best_val_metric is None or val_rep_loss <= best_val_metric - DELTA:
                best_val_metric = val_rep_loss
                timer = 0
                best_weights = model.get_weights()
            else:
                timer += 1
            if timer == EARLY_STOPPING_PATIENCE:
                model.set_weights(best_weights)
                first_stage_epochs = epoch + 1 - EARLY_STOPPING_PATIENCE
                break


    for layer in model.layers:
        layer.trainable = False
        if layer.name == 'representation':
           break
    best_val_metric = None
    timer = 0
    best_weights = None

    optimizer = second_stage_optimizer

    for epoch in range(SECOND_STAGE_EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch, weights) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = regression_step(x_batch, y_reg_batch, weights)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )
        if VALIDATION_DATA:
            val_loss, val_reg_loss, val_rep_loss, mae = regression_step(x_val, y_val, w_val, train=False)
            print(
                f"  Validation for epoch {epoch + 1} | "
                f"Total: {float(val_loss):.4f} | "
                f"Regression: {float(val_reg_loss):.4f} | "
                f"Representation: {float(val_rep_loss):.4f} | "
                f"MAE: {float(mae):.4f}"
            )
            if best_val_metric is None or mae <= best_val_metric - DELTA:
                best_val_metric = mae
                timer = 0
                best_weights = model.get_weights()
            else:
                timer += 1
            if timer == EARLY_STOPPING_PATIENCE:
                model.set_weights(best_weights)
                second_stage_epochs = epoch + 1 - EARLY_STOPPING_PATIENCE
                break

elif FIT_MODE == 'tune':
    for epoch in range(EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch, weights) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = representation_step(x_batch, y_reg_batch, weights)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )

        if VALIDATION_DATA:
            val_loss, val_reg_loss, val_rep_loss, mae = representation_step(x_val, y_val, w_val, train=False)
            print(
                f"  Validation for epoch {epoch + 1} | "
                f"Total: {float(val_loss):.4f} | "
                f"Regression: {float(val_reg_loss):.4f} | "
                f"Representation: {float(val_rep_loss):.4f} | "
                f"MAE: {float(mae):.4f}"
            )
            if best_val_metric is None or val_rep_loss <= best_val_metric - DELTA:
                best_val_metric = val_rep_loss
                timer = 0
                best_weights = model.get_weights()
            else:
                timer += 1
            if timer == EARLY_STOPPING_PATIENCE:
                model.set_weights(best_weights)
                first_stage_epochs = epoch + 1 - EARLY_STOPPING_PATIENCE
                break

    optimizer = second_stage_optimizer
    best_val_metric = None
    timer = 0
    best_weights = None

    for epoch in range(SECOND_STAGE_EPOCHS):
        print(f"\nStart of epoch {epoch + 1}")

        for step, (x_batch, y_reg_batch, weights) in enumerate(train_dataset):

            if MANUALLY_SORT_EVERY_BATCH:
                sort_indices = tf.argsort(tf.squeeze(y_reg_batch))
                x_batch = tf.gather(x_batch, sort_indices)
                y_reg_batch = tf.gather(y_reg_batch, sort_indices)

            loss, reg_loss, rep_loss = regression_step(x_batch, y_reg_batch, weights)
            if step % 10 == 0:
                print(
                    f"  Step {step:3d} | "
                    f"Total: {float(loss):.4f} | "
                    f"Regression: {float(reg_loss):.4f} | "
                    f"Representation: {float(rep_loss):.4f}"
                )

        if VALIDATION_DATA:
            val_loss, val_reg_loss, val_rep_loss, mae = regression_step(x_val, y_val, w_val, train=False)
            print(
                f"  Validation for epoch {epoch + 1} | "
                f"Total: {float(val_loss):.4f} | "
                f"Regression: {float(val_reg_loss):.4f} | "
                f"Representation: {float(val_rep_loss):.4f} | "
                f"MAE: {float(mae):.4f}"
            )
            if best_val_metric is None or mae <= best_val_metric - DELTA:
                best_val_metric = mae
                timer = 0
                best_weights = model.get_weights()
            else:
                timer += 1

            if timer == EARLY_STOPPING_PATIENCE:
                model.set_weights(best_weights)
                second_stage_epochs = epoch + 1 - EARLY_STOPPING_PATIENCE
                break

def generate_plots(
    x,
    y,
    path_prefix
):
    predictions, representations = model.predict(x)
    predictions = predictions.reshape(-1)

    if isinstance(y, tf.Tensor):
        y = y.numpy()

    y = y.reshape(-1)

    common_sample_mask = (y > 6.25) & (y < 9)
    common_predictions = predictions[common_sample_mask]
    rare_predictions = predictions[~common_sample_mask]
    common_labels = y[common_sample_mask]
    rare_labels = y[~common_sample_mask]

    mae = np.mean(np.abs(predictions - y))
    common_mae = np.mean(np.abs(common_predictions - common_labels))
    rare_mae = np.mean(np.abs(rare_predictions - rare_labels))

    plot_representation_space(
        representations,
        y,
        0,
        1,
        vmin=0,
        vmax=13.5,
        save_figure=f'{path_prefix}/representation/{FIGURE_NAME}.png'
    )

    imbal.regression.plot_true_vs_predictions(
        y,
        predictions,
        title=f'Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae) / 2:.4f}',
        save_figure=f'{path_prefix}/true_vs_predicted/{FIGURE_NAME}.png'
    )

generate_plots(x_train, y_train, 'results_onp/training')
generate_plots(x_val, y_val, 'results_onp/validation')
generate_plots(x_test, y_test, 'results_onp/test')


# predictions, representations = model.predict(x_val)
# predictions = predictions.reshape(-1)
# y_val = y_val.numpy().reshape(-1)
#
# common_sample_mask = (y_val > -0.5) & (y_val < 0.5)
# common_predictions = predictions[common_sample_mask]
# rare_predictions = predictions[~common_sample_mask]
# common_labels = y_val[common_sample_mask]
# rare_labels = y_val[~common_sample_mask]
#
# mae = np.mean(np.abs(predictions - y_val))
# common_mae = np.mean(np.abs(common_predictions - common_labels))
# rare_mae = np.mean(np.abs(rare_predictions - rare_labels))
#
# plot_representation_space(
#     representations,
#     y_val,
#     0,
#     1,
#     save_figure=f'results/validation/representation/{FIGURE_NAME}.png'
# )
#
# imbal.regression.plot_true_vs_predictions(
#     y_val,
#     predictions,
#     title=f'Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}',
#     save_figure=f'results/validation/true_vs_predicted/{FIGURE_NAME}.png'
# )
#
# predictions, representations = model.predict(x_test)
# predictions = predictions.reshape(-1)
# y_test = y_test.reshape(-1)
#
# common_sample_mask = (y_test > -0.5) & (y_test < 0.5)
# common_predictions = predictions[common_sample_mask]
# rare_predictions = predictions[~common_sample_mask]
# common_labels = y_test[common_sample_mask]
# rare_labels = y_test[~common_sample_mask]
#
# mae = np.mean(np.abs(predictions - y_test))
# common_mae = np.mean(np.abs(common_predictions - common_labels))
# rare_mae = np.mean(np.abs(rare_predictions - rare_labels))
#
# plot_representation_space(
#     representations,
#     y_test,
#     0,
#     1,
#     save_figure=f'results/test/representation/{FIGURE_NAME}.png'
# )
#
# imbal.regression.plot_true_vs_predictions(
#     y_test,
#     predictions,
#     title=f'Common MAE: {common_mae:.4f}, Rare MAE: {rare_mae:.4f}, AORE: {(mae + rare_mae)/2:.4f}',
#     save_figure=f'results/test/true_vs_predicted/{FIGURE_NAME}.png'
# )

print(first_stage_epochs, second_stage_epochs)

