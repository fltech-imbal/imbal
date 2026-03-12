import tensorflow as tf
from .weight_to_dict import create_weight_tensor_fast
from tensorflow.keras import backend as K

def mse_wpcc(
    y_true: tf.Tensor, y_pred: tf.Tensor,
    lambda_factor,
    phase_manager,
    train_mse_weight_dict = None,
    val_mse_weight_dict = None,
    train_pcc_weight_dict = None,
    val_pcc_weight_dict = None,
    normalized_weights = False,
) -> tf.Tensor:
    """
    Custom loss function combining Mean Squared Error (MSE) and Pearson Correlation Coefficient (PCC)
    with re-weighting based on label values. The final loss is a combination of weighted MSE and
    weighted PCC with a scaling factor lambda_factor.

    Args:
    - y_true (tf.Tensor): Ground truth labels.
    - y_pred (tf.Tensor): Predicted labels.
    - lambda_factor (float): Scaling factor for the PCC portion of the loss.
    - phase_manager (TrainingPhaseManager): Manager that tracks whether we are in training or validation phase.
    - train_mse_weight_dict (dict, optional): Dictionary mapping label values to weights for training MSE samples.
    - val_mse_weight_dict (dict, optional): Dictionary mapping label values to weights for validation MSE samples.
    - train_pcc_weight_dict (dict, optional): Dictionary mapping label values to weights for training PCC samples.
    - val_pcc_weight_dict (dict, optional): Dictionary mapping label values to weights for validation PCC samples.
    - normalized_weights (bool, optional): If True, weights are already normalized and we use sum instead of mean.
    - asym_type (str, optional): Type of asymmetric weight to use ('silu' or 'sigmoid').
    - bias_penalty_factor (float, optional): Scaling factor for the bias penalty term. If None, no bias penalty is applied.

    Returns:
    - tf.Tensor: The calculated loss value as a single scalar.
    """
    # Select the appropriate weight dictionaries based on the mode
    mse_weight_dict = train_mse_weight_dict if phase_manager.is_training_phase() else val_mse_weight_dict

    # Generate the weight tensor for MSE using the optimized function
    mse_weights = create_weight_tensor_fast(y_true, mse_weight_dict)

    # Compute the Mean Squared Error (MSE) with asymmetric weights
    # If weights are normalized, use sum instead of mean
    if normalized_weights:
        mse = tf.reduce_sum(mse_weights * tf.square(y_pred - y_true))
    else:
        mse = tf.reduce_mean(mse_weights * tf.square(y_pred - y_true))

    # Early return if lambda_factor is zero (no PCC component)
    if lambda_factor == 0:
        # Add bias penalty if factor is provided
        return mse

    # If lambda_factor is not zero, compute the PCC component
    pcc_weight_dict = train_pcc_weight_dict if phase_manager.is_training_phase() else val_pcc_weight_dict
    pcc_weights = create_weight_tensor_fast(y_true, pcc_weight_dict)

    # Compute the correlation regularization term using coreg
    pcc_loss = coreg(y_true, y_pred, pcc_weights)

    # Combine the weighted MSE and weighted PCC with lambda_factor
    loss = mse + lambda_factor * pcc_loss

    # Return the final loss as a single scalar value
    return loss


def coreg(y_true: tf.Tensor, y_pred: tf.Tensor, pcc_weights = None) -> tf.Tensor:
    """
    Correlation based regularizer:
    Compute 1 minus the Pearson Correlation Coefficient (PCC) between two sets of predictions.

    PCC measures the linear correlation between two variables, providing a value
    between -1 (perfect negative correlation) and 1 (perfect positive correlation).
    A value of 0 indicates no linear correlation.

    Returns 1-PCC which can be used as a loss function, where 0 represents perfect
    positive correlation and 2 represents perfect negative correlation.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth values.
    y_pred : tf.Tensor
        Predicted values.
    pcc_weights : tf.Tensor, optional
        Weights for each observation. If None, uniform weights are used.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing 1-PCC (for use as a loss function)
    """
    if pcc_weights is None:
        pcc_weights = tf.ones_like(y_true)

    # Center the data by subtracting their means
    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)

    # Compute covariance
    cov = tf.reduce_sum(pcc_weights * y_true_centered * y_pred_centered)

    # Compute variances
    var_y_true = tf.reduce_sum(pcc_weights * tf.square(y_true_centered))
    var_y_pred = tf.reduce_sum(pcc_weights * tf.square(y_pred_centered))

    # Compute PCC using single sqrt
    pcc = cov / (tf.sqrt(var_y_true * var_y_pred) + K.epsilon())

    # Return 1-PCC
    return 1.0 - pcc