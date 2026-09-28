import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold

import imbal


TARGET_COLUMN = "ln_peak_intensity"
RAW_TARGET_COLUMN = "peak_intensity"
NON_FEATURE_COLUMNS = [
    "source_row",
    "SEP_onset_time",
    "CME_DONKI_time",
    "CME_CDAW_time",
]
RARE_THRESHOLD = np.log(10)
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIRECTORY, "sep_10mev_training_raw.csv")
TEST_DATA_PATH = os.path.join(SCRIPT_DIRECTORY, "sep_10mev_testing_raw.csv")
EXPECTED_INPUT_COUNT = 22
MAX_EPOCHS = 500
PATIENCE = 100
BATCH_SIZE = 32
MIN_KDE_BANDWIDTH = 0.05
DENSITY_EPSILON = 1e-7

LOAD_SAVED_MODEL = False
MODEL_RUN_NAME = "density_trust_sep_c_hierarchical_trust"
MODEL_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "saved_models", MODEL_RUN_NAME)
COMMON_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "common_expert.keras")
RARE_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "rare_expert.keras")
GATE_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "gate.keras")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIRECTORY, "ensemble.keras")


def set_global_determinism(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_sep_c_dataset(
    train_data_path=TRAIN_DATA_PATH,
    test_data_path=TEST_DATA_PATH,
    target_column=TARGET_COLUMN,
    rare_threshold=RARE_THRESHOLD,
    expected_input_count=EXPECTED_INPUT_COUNT,
):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    for split_name, data in (("training", train_data), ("testing", test_data)):
        required_columns = [target_column, RAW_TARGET_COLUMN, *NON_FEATURE_COLUMNS]
        missing_columns = [
            column for column in required_columns if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The {split_name} data is missing required columns: "
                f"{missing_columns}."
            )

    # Keep ln_peak_intensity as the regression target. Remove the original
    # peak intensity, row/time metadata, and target from the model inputs.
    columns_to_drop = [
        target_column,
        RAW_TARGET_COLUMN,
        *NON_FEATURE_COLUMNS,
    ]
    training_feature_data = train_data.drop(columns=columns_to_drop)
    testing_feature_data = test_data.drop(columns=columns_to_drop)

    if list(training_feature_data.columns) != list(testing_feature_data.columns):
        raise ValueError(
            "Training and testing feature columns must match in the same order."
        )

    raw_training_features = training_feature_data.to_numpy(dtype="float32")
    raw_testing_features = testing_feature_data.to_numpy(dtype="float32")
    training_features, testing_features = sign_preserving_min_max_scale(
        raw_training_features, raw_testing_features
    )
    training_targets = train_data[target_column].to_numpy(
        dtype="float32"
    ).reshape(-1, 1)
    testing_targets = test_data[target_column].to_numpy(
        dtype="float32"
    ).reshape(-1, 1)

    for split_name, features in (
        ("training", training_features),
        ("testing", testing_features),
    ):
        if features.shape[1] != expected_input_count:
            raise ValueError(
                f"Expected {expected_input_count} input features, but found "
                f"{features.shape[1]} in the {split_name} data."
            )

    training_region_labels = (training_targets >= rare_threshold).astype(
        "int32"
    )
    testing_region_labels = (testing_targets >= rare_threshold).astype("int32")

    all_features = np.concatenate([training_features, testing_features], axis=0)
    all_targets = np.concatenate([training_targets, testing_targets], axis=0)

    return (
        training_features,
        testing_features,
        training_targets,
        testing_targets,
        training_region_labels,
        testing_region_labels,
        all_features,
        all_targets,
    )


def sign_preserving_min_max_scale(training_features, testing_features):
    """Scale each input feature by sign using training-only bounds.

    Negative values are mapped to [-1, 0), positive values to (0, 1], and
    zeros remain exactly zero. Test values outside the training range are
    clipped so the transformed inputs remain within [-1, 1].
    """
    training_features = np.asarray(training_features, dtype="float32")
    testing_features = np.asarray(testing_features, dtype="float32")

    if not np.all(np.isfinite(training_features)):
        raise ValueError("Training features contain NaN or infinite values.")
    if not np.all(np.isfinite(testing_features)):
        raise ValueError("Testing features contain NaN or infinite values.")

    negative_scales = np.max(
        np.where(training_features < 0, -training_features, 0.0), axis=0
    )
    positive_scales = np.max(
        np.where(training_features > 0, training_features, 0.0), axis=0
    )

    def transform(features):
        scaled = np.zeros_like(features, dtype="float32")
        negative_mask = features < 0
        positive_mask = features > 0

        safe_negative_scales = np.where(negative_scales > 0, negative_scales, 1.0)
        safe_positive_scales = np.where(positive_scales > 0, positive_scales, 1.0)
        scaled = np.where(
            negative_mask,
            features / safe_negative_scales,
            scaled,
        )
        scaled = np.where(
            positive_mask,
            features / safe_positive_scales,
            scaled,
        )
        return np.clip(scaled, -1.0, 1.0).astype("float32")

    return transform(training_features), transform(testing_features)


def build_expert(expert_name, input_count=EXPECTED_INPUT_COUNT):
    inputs = keras.Input(shape=(input_count,), name=f"{expert_name}_input")
    x = layers.Dense(18, activation="relu", name=f"{expert_name}_dense_1")(inputs)
    x = layers.Dense(12, activation="relu", name=f"{expert_name}_dense_2")(x)
    x = layers.Dense(8, activation="relu", name=f"{expert_name}_dense_3")(x)
    rep = layers.Dense(6, activation="relu", name=f"{expert_name}_representation")(x)
    return keras.Model(inputs, layers.Dense(1, name=f"{expert_name}_prediction")(rep), name=expert_name)


def compile_expert(model):
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])


ALPHA_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
SUBBIN_BALANCE_TOLERANCE = 0.10


def build_gate(input_count=EXPECTED_INPUT_COUNT):
    """Output learned normalized expert trusts [g(C), g(R)]."""
    inputs = keras.Input(shape=(input_count,), name="gate_input")
    x = layers.Dense(18, activation="relu", name="gate_dense_1")(inputs)
    x = layers.Dense(12, activation="relu", name="gate_dense_2")(x)
    x = layers.Dense(8, activation="relu", name="gate_representation")(x)
    trust = layers.Dense(2, activation="softmax", name="gate_trust")(x)
    return keras.Model(inputs, trust, name="gate")


def reciprocal_weights(targets, alpha):
    y = np.asarray(targets).reshape(-1)
    kde = imbal.regression.fit_kde(y)
    density = imbal.regression.get_sample_densities(y, kde)
    return np.asarray(imbal.regression.reciprocal_importance(density, alpha=float(alpha)), dtype="float32").reshape(-1)


def expert_subbin_info(targets):
    y = np.asarray(targets).reshape(-1)
    midpoint = float(np.min(y) + (np.max(y) - np.min(y)) / 2.0)
    low, high = int(np.sum(y <= midpoint)), int(np.sum(y > midpoint))
    # "10% difference" implemented relative to the larger sub-bin.
    difference = abs(low - high) / max(low, high, 1)
    return midpoint, low, high, difference, difference >= SUBBIN_BALANCE_TOLERANCE


def subbin_average_error(targets, predictions, midpoint):
    y, pred = np.asarray(targets).reshape(-1), np.asarray(predictions).reshape(-1)
    low, high = y <= midpoint, y > midpoint
    if not np.any(low) or not np.any(high):
        return np.inf
    return (mean_absolute_error(y[low], pred[low]) + mean_absolute_error(y[high], pred[high])) / 2.0


class ExpertSAEEarlyStopping(keras.callbacks.Callback):
    def __init__(self, x_val, y_val, midpoint, patience=PATIENCE):
        super().__init__(); self.x_val=x_val; self.y_val=y_val; self.midpoint=midpoint; self.patience=patience
        self.best_sae=np.inf; self.best_epoch=1; self.best_weights=None; self.wait=0
    def on_epoch_end(self, epoch, logs=None):
        sae = subbin_average_error(self.y_val, self.model(self.x_val, training=False), self.midpoint)
        if sae < self.best_sae:
            self.best_sae=sae; self.best_epoch=epoch+1; self.best_weights=self.model.get_weights(); self.wait=0
        else: self.wait += 1
        if self.wait >= self.patience: self.model.stop_training=True
    def on_train_end(self, logs=None):
        if self.best_weights is not None: self.model.set_weights(self.best_weights)


def evaluate_expert_alpha_with_kfold(features, targets, expert_name, alpha, midpoint, weighted, fold_count=5, seed=42):
    sublabels=(np.asarray(targets).reshape(-1)>midpoint).astype("int32")
    counts=np.bincount(sublabels, minlength=2)
    if np.min(counts) < fold_count:
        raise ValueError(f"{expert_name}: each sub-bin needs at least {fold_count} samples for SAE k-fold validation.")
    splitter=StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    saes=[]; epochs=[]
    for fold,(tr,va) in enumerate(splitter.split(features,sublabels),1):
        tf.keras.backend.clear_session(); set_global_determinism(seed+fold)
        model=build_expert(expert_name, features.shape[1]); compile_expert(model)
        cb=ExpertSAEEarlyStopping(features[va], targets[va], midpoint)
        sw=reciprocal_weights(targets[tr], alpha) if weighted else None
        model.fit(features[tr],targets[tr],sample_weight=sw,epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,callbacks=[cb],shuffle=False,verbose=0)
        saes.append(cb.best_sae); epochs.append(cb.best_epoch)
    return float(np.mean(saes)), max(1,int(np.round(np.mean(epochs))))


def select_and_train_expert(features, targets, expert_name, seed):
    midpoint,n1,n2,diff,imbalanced=expert_subbin_info(targets)
    print(f"\n{expert_name} sub-bins: midpoint={midpoint:.6f}, counts={n1}/{n2}, difference={100*diff:.2f}%")
    candidates=ALPHA_CANDIDATES if imbalanced else [None]
    best=(np.inf,None,None)
    for alpha in candidates:
        sae,epochs=evaluate_expert_alpha_with_kfold(features,targets,expert_name,0.0 if alpha is None else alpha,midpoint,imbalanced,seed=seed)
        print(f"  {'regular MSE' if alpha is None else f'alpha={alpha}'}: SAE={sae:.4f}, epochs={epochs}")
        if sae < best[0]: best=(sae,alpha,epochs)
    _,alpha,epochs=best
    tf.keras.backend.clear_session(); set_global_determinism(seed+1000)
    model=build_expert(expert_name,features.shape[1]); compile_expert(model)
    sw=reciprocal_weights(targets,alpha) if imbalanced else None
    model.fit(features,targets,sample_weight=sw,epochs=epochs,batch_size=BATCH_SIZE,shuffle=False,verbose=0)
    print(f"  selected alpha: {alpha if imbalanced else 'N/A (balanced)'}; selected epochs: {epochs}; SAE={best[0]:.4f}")
    return model,alpha,epochs

def select_kde_bandwidth(targets, minimum_bandwidth=MIN_KDE_BANDWIDTH):
    values = np.asarray(targets, dtype="float64").reshape(-1)
    if len(values) < 2:
        return float(minimum_bandwidth)
    estimated = 1.06 * np.std(values, ddof=1) * (len(values) ** (-1.0 / 5.0))
    if not np.isfinite(estimated):
        estimated = 0.0
    return float(max(estimated, minimum_bandwidth))


def gaussian_kde_density(query_values, training_targets, bandwidth):
    queries = np.asarray(query_values, dtype="float64").reshape(-1)
    samples = np.asarray(training_targets, dtype="float64").reshape(-1)
    if len(samples) == 0:
        raise ValueError("A density profile cannot be built without training targets.")
    normalizer = len(samples) * bandwidth * np.sqrt(2.0 * np.pi)
    densities = np.empty(len(queries), dtype="float64")
    for start in range(0, len(queries), 1024):
        stop = min(start + 1024, len(queries))
        differences = (queries[start:stop, None] - samples[None, :]) / bandwidth
        densities[start:stop] = np.sum(np.exp(-0.5 * differences ** 2), axis=1) / normalizer
    return densities


def build_density_profile(expert_targets):
    targets = np.asarray(expert_targets, dtype="float32").reshape(-1)
    bandwidth = select_kde_bandwidth(targets)
    training_densities = gaussian_kde_density(targets, targets, bandwidth)
    maximum_density = float(np.max(training_densities))
    if not np.isfinite(maximum_density) or maximum_density <= 0.0:
        raise ValueError("The KDE maximum density must be positive and finite.")
    return {"training_targets": targets, "bandwidth": bandwidth, "maximum_density": maximum_density}


def evaluate_normalized_density(predictions, profile):
    raw_density = gaussian_kde_density(predictions, profile["training_targets"], profile["bandwidth"])
    normalized = raw_density / profile["maximum_density"]
    return np.clip(normalized, DENSITY_EPSILON, 1.0).astype("float32")


@keras.utils.register_keras_serializable(package="SEPDensityTrust")
class NormalizedGaussianKDE(layers.Layer):
    def __init__(self, training_targets, bandwidth, maximum_density, **kwargs):
        super().__init__(**kwargs)
        self.training_targets_list = np.asarray(training_targets, dtype="float32").reshape(-1).tolist()
        self.bandwidth = float(bandwidth)
        self.maximum_density = float(maximum_density)

    def call(self, predictions):
        samples = tf.constant(self.training_targets_list, dtype=predictions.dtype)
        differences = (tf.reshape(predictions, [-1, 1]) - tf.reshape(samples, [1, -1])) / tf.cast(self.bandwidth, predictions.dtype)
        kernel_sum = tf.reduce_sum(tf.exp(-0.5 * tf.square(differences)), axis=1)
        normalizer = tf.cast(len(self.training_targets_list) * self.bandwidth * np.sqrt(2.0 * np.pi), predictions.dtype)
        raw_density = kernel_sum / normalizer
        normalized = raw_density / tf.cast(self.maximum_density, predictions.dtype)
        return tf.reshape(tf.clip_by_value(normalized, DENSITY_EPSILON, 1.0), [-1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"training_targets": self.training_targets_list, "bandwidth": self.bandwidth, "maximum_density": self.maximum_density})
        return config


@keras.utils.register_keras_serializable(package="SEPDensityTrust")
class NormalizeExpertWeights(layers.Layer):
    def call(self, values):
        denominator = tf.maximum(tf.reduce_sum(values, axis=1, keepdims=True), DENSITY_EPSILON)
        return values / denominator


def build_trainable_gate_ensemble(common_expert, rare_expert, gate, common_density_profile, rare_density_profile):
    common_expert.trainable = False
    rare_expert.trainable = False
    gate.trainable = True
    inputs = keras.Input(shape=common_expert.input_shape[1:], name="features")
    cp = common_expert(inputs, training=False)
    rp = rare_expert(inputs, training=False)
    trust = gate(inputs)
    cq = NormalizedGaussianKDE(**common_density_profile, name="common_expert_density")(cp)
    rq = NormalizedGaussianKDE(**rare_density_profile, name="rare_expert_density")(rp)
    quality = layers.Concatenate(name="expert_densities")([cq, rq])
    trust_quality = layers.Multiply(name="density_weighted_gate")([trust, quality])
    weights = NormalizeExpertWeights(name="normalized_expert_weights")(trust_quality)
    preds = layers.Concatenate(name="expert_predictions")([cp, rp])
    out = layers.Dot(axes=1, name="gated_prediction")([preds, weights])
    model = keras.Model(inputs, out, name="trainable_density_trust_ensemble")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


class GateEnsembleAOREEarlyStopping(keras.callbacks.Callback):
    def __init__(self,x_val,y_val,region_labels,gate,patience=PATIENCE):
        super().__init__(); self.x_val=x_val; self.y_val=np.asarray(y_val).reshape(-1)
        self.rare=np.asarray(region_labels).reshape(-1).astype(bool); self.gate=gate; self.patience=patience
        self.best_aore=np.inf; self.best_epoch=1; self.best_gate_weights=None; self.wait=0
    def on_epoch_end(self,epoch,logs=None):
        pred=np.asarray(self.model(self.x_val,training=False)).reshape(-1)
        overall=mean_absolute_error(self.y_val,pred); rare=mean_absolute_error(self.y_val[self.rare],pred[self.rare])
        aore=(overall+rare)/2.0
        if aore < self.best_aore:
            self.best_aore=aore; self.best_epoch=epoch+1; self.best_gate_weights=self.gate.get_weights(); self.wait=0
        else: self.wait+=1
        if self.wait>=self.patience: self.model.stop_training=True
    def on_train_end(self,logs=None):
        if self.best_gate_weights is not None: self.gate.set_weights(self.best_gate_weights)


def evaluate_gate_alpha_with_kfold(features,targets,region_labels,common_expert,rare_expert,common_density_profile,rare_density_profile,alpha,fold_count=5,seed=42):
    labels=np.asarray(region_labels).reshape(-1).astype("int32")
    splitter=StratifiedKFold(n_splits=fold_count,shuffle=True,random_state=seed)
    aores=[]; epochs=[]
    for fold,(tr,va) in enumerate(splitter.split(features,labels),1):
        tf.keras.backend.clear_session(); set_global_determinism(seed+100+fold)
        gate=build_gate(features.shape[1]); trainer=build_trainable_gate_ensemble(common_expert,rare_expert,gate,common_density_profile,rare_density_profile)
        cb=GateEnsembleAOREEarlyStopping(features[va],targets[va],labels[va],gate)
        sw=reciprocal_weights(targets[tr],alpha)
        trainer.fit(features[tr],targets[tr],sample_weight=sw,epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,callbacks=[cb],shuffle=False,verbose=0)
        aores.append(cb.best_aore); epochs.append(cb.best_epoch)
    return float(np.mean(aores)),max(1,int(np.round(np.mean(epochs))))


def train_gate(features,targets,common_expert,rare_expert,common_density_profile,rare_density_profile,alpha,epoch_count,seed):
    tf.keras.backend.clear_session(); set_global_determinism(seed)
    gate=build_gate(features.shape[1]); trainer=build_trainable_gate_ensemble(common_expert,rare_expert,gate,common_density_profile,rare_density_profile)
    trainer.fit(features,targets,sample_weight=reciprocal_weights(targets,alpha),epochs=epoch_count,batch_size=BATCH_SIZE,shuffle=False,verbose=0)
    return gate


def build_gated_ensemble(common_expert, rare_expert, gate, common_density_profile, rare_density_profile):
    common_expert.trainable = False
    rare_expert.trainable = False
    gate.trainable = False
    inputs = keras.Input(shape=common_expert.input_shape[1:], name="features")
    cp = common_expert(inputs, training=False)
    rp = rare_expert(inputs, training=False)
    trust = gate(inputs, training=False)
    cq = NormalizedGaussianKDE(**common_density_profile, name="common_expert_density")(cp)
    rq = NormalizedGaussianKDE(**rare_density_profile, name="rare_expert_density")(rp)
    quality = layers.Concatenate(name="expert_densities")([cq, rq])
    trust_quality = layers.Multiply(name="density_weighted_gate")([trust, quality])
    weights = NormalizeExpertWeights(name="normalized_expert_weights")(trust_quality)
    preds = layers.Concatenate(name="expert_predictions")([cp, rp])
    out = layers.Dot(axes=1, name="gated_prediction")([preds, weights])
    return keras.Model(inputs, out, name="two_expert_density_trust_ensemble")


def evaluate_regressor_with_aore_components(
    model,
    evaluation_features,
    evaluation_targets,
    evaluation_region_labels,
):
    predictions = np.asarray(model(evaluation_features, training=False)).reshape(-1)
    targets = np.asarray(evaluation_targets).reshape(-1)
    rare_mask = np.asarray(evaluation_region_labels).reshape(-1).astype(bool)

    if not np.any(rare_mask):
        raise ValueError("No rare samples were found when computing rare MAE.")

    overall_mae = mean_absolute_error(targets, predictions)
    rare_mae = mean_absolute_error(targets[rare_mask], predictions[rare_mask])
    return overall_mae, rare_mae, (overall_mae + rare_mae) / 2.0


def print_gate_summary(gate, features, region_labels, split_name):
    trust = np.asarray(gate(features, training=False))
    labels = np.asarray(region_labels).reshape(-1).astype("int32")
    common, rare = labels == 0, labels == 1
    print(f"{split_name} mean g(C) for true common samples: {np.mean(trust[common,0]):.4f}")
    print(f"{split_name} mean g(R) for true common samples: {np.mean(trust[common,1]):.4f}")
    print(f"{split_name} mean g(C) for true rare samples: {np.mean(trust[rare,0]):.4f}")
    print(f"{split_name} mean g(R) for true rare samples: {np.mean(trust[rare,1]):.4f}")


def print_elevated_rare_tables(common_expert, rare_expert, gate, features, targets, common_profile, rare_profile, split_name):
    true_targets = np.asarray(targets).reshape(-1)
    cp = np.asarray(common_expert(features, training=False)).reshape(-1)
    rp = np.asarray(rare_expert(features, training=False)).reshape(-1)
    g = np.asarray(gate(features, training=False))
    qc = evaluate_normalized_density(cp, common_profile)
    qr = evaluate_normalized_density(rp, rare_profile)
    q = np.column_stack([qc, qr])
    raw = g * q
    w = raw / np.maximum(np.sum(raw, axis=1, keepdims=True), DENSITY_EPSILON)
    final = w[:,0] * cp + w[:,1] * rp
    regions = [("ELEVATED SAMPLES", (true_targets >= 0.0) & (true_targets < RARE_THRESHOLD)),
               ("RARE SAMPLES", true_targets >= RARE_THRESHOLD)]
    for name, mask in regions:
        indices = np.flatnonzero(mask)
        print(f"\n{split_name} {name}")
        header = (f"{'Idx':>5} {'True y':>10} {'CommonPred':>12} {'RarePred':>12} "
                  f"{'g(C)':>9} {'g(R)':>9} {'q(C)':>9} {'q(R)':>9} "
                  f"{'w(C)':>9} {'w(R)':>9} {'FinalPred':>12} {'Delta':>12}")
        print(header); print("-" * len(header))
        for i in indices:
            print(f"{i:5d} {true_targets[i]:10.4f} {cp[i]:12.4f} {rp[i]:12.4f} "
                  f"{g[i,0]:9.4f} {g[i,1]:9.4f} {qc[i]:9.4f} {qr[i]:9.4f} "
                  f"{w[i,0]:9.4f} {w[i,1]:9.4f} {final[i]:12.4f} {(final[i]-true_targets[i]):12.4f}")



def run_two_expert_gated_ensemble(seed=42):
    set_global_determinism(seed)
    (training_features, testing_features, training_targets, testing_targets,
     training_region_labels, testing_region_labels, all_features, all_targets) = load_sep_c_dataset()

    common_mask = training_region_labels.reshape(-1) == 0
    rare_mask = training_region_labels.reshape(-1) == 1
    common_features, common_targets = training_features[common_mask], training_targets[common_mask]
    rare_features, rare_targets = training_features[rare_mask], training_targets[rare_mask]

    print(f"Training samples: {len(training_features)}")
    print(f"Common training samples: {len(common_features)}")
    print(f"Rare training samples: {len(rare_features)}")
    print(f"Testing samples: {len(testing_features)}")

    common_density_profile = build_density_profile(common_targets)
    rare_density_profile = build_density_profile(rare_targets)
    print("\nDensity Trust Profiles")
    print(f"Common KDE bandwidth: {common_density_profile['bandwidth']:.6f}")
    print(f"Rare KDE bandwidth: {rare_density_profile['bandwidth']:.6f}")

    if LOAD_SAVED_MODEL:
        print(f"\nLoading saved models from: {MODEL_DIRECTORY}")
        common_expert=keras.models.load_model(COMMON_MODEL_PATH)
        rare_expert=keras.models.load_model(RARE_MODEL_PATH)
        gate=keras.models.load_model(GATE_MODEL_PATH)
    else:
        common_expert, common_alpha, common_epochs = select_and_train_expert(
            common_features, common_targets, "common_expert", seed+1000)
        rare_expert, rare_alpha, rare_epochs = select_and_train_expert(
            rare_features, rare_targets, "rare_expert", seed+2000)

        print("\nGate reciprocal-alpha search (trust gate trained through ensemble MSE)")
        best=(np.inf,None,None)
        for alpha in ALPHA_CANDIDATES:
            aore,epochs=evaluate_gate_alpha_with_kfold(
                training_features,training_targets,training_region_labels,
                common_expert,rare_expert,common_density_profile,rare_density_profile,alpha,fold_count=5,seed=seed)
            print(f"  alpha={alpha}: mean validation AORE={aore:.4f}, mean best epochs={epochs}")
            if aore < best[0]: best=(aore,alpha,epochs)
        best_aore,best_gate_alpha,best_gate_epochs=best
        print(f"\nBest gate alpha: {best_gate_alpha}")
        print(f"Gate epochs: {best_gate_epochs}")
        print(f"Mean k-fold validation AORE: {best_aore:.4f}")

        gate=train_gate(training_features,training_targets,common_expert,rare_expert,
                        common_density_profile,rare_density_profile,best_gate_alpha,best_gate_epochs,seed+3000)
        os.makedirs(MODEL_DIRECTORY,exist_ok=True)
        common_expert.save(COMMON_MODEL_PATH); rare_expert.save(RARE_MODEL_PATH); gate.save(GATE_MODEL_PATH)

    ensemble=build_gated_ensemble(common_expert,rare_expert,gate,common_density_profile,rare_density_profile)
    if not LOAD_SAVED_MODEL:
        ensemble.save(ENSEMBLE_MODEL_PATH)
        print(f"Saved models to: {MODEL_DIRECTORY}")

    print("\nGate Trust Results")
    print_gate_summary(gate,training_features,training_region_labels,"Training")
    print_gate_summary(gate,testing_features,testing_region_labels,"Test")
    print_elevated_rare_tables(common_expert,rare_expert,gate,testing_features,testing_targets,common_density_profile,rare_density_profile,"Test")

    training_metrics=evaluate_regressor_with_aore_components(ensemble,training_features,training_targets,training_region_labels)
    testing_metrics=evaluate_regressor_with_aore_components(ensemble,testing_features,testing_targets,testing_region_labels)
    print("\nTraining Ensemble Results")
    print(f"Training overall MAE: {training_metrics[0]:.4f}")
    print(f"Training rare MAE: {training_metrics[1]:.4f}")
    print(f"Training AORE: {training_metrics[2]:.4f}")
    print("\nFinal Test Set Results")
    print(f"Test overall MAE: {testing_metrics[0]:.4f}")
    print(f"Test rare MAE: {testing_metrics[1]:.4f}")
    print(f"Test AORE: {testing_metrics[2]:.4f}")

    common_representation=keras.Model(common_expert.input,common_expert.get_layer("common_expert_representation").output)
    rare_representation=keras.Model(rare_expert.input,rare_expert.get_layer("rare_expert_representation").output)
    gate_representation=keras.Model(gate.input,gate.get_layer("gate_representation").output)
    imbal.regression.tsne_visualization(common_representation,all_features,all_targets)
    imbal.regression.tsne_visualization(rare_representation,all_features,all_targets)
    imbal.regression.tsne_visualization(gate_representation,all_features,all_targets)
    test_predictions=ensemble.predict(testing_features)
    imbal.regression.plot_true_vs_predictions(testing_targets,test_predictions)
    return ensemble,common_expert,rare_expert,gate


if __name__ == "__main__":
    final_ensemble, common_model, rare_model, gating_model = (
        run_two_expert_gated_ensemble(seed=42)
    )
