"""Load data, run one SEP-C method, and report results without repeated KDE work."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from sep_c_models import (
    ExpertMSEPCCLoss,
    NormalizedGaussianKDE,
    NormalizeExpertWeights,
    WeightedExpertMSEPCCLoss,
    build_density_profile,
    build_ensemble,
    predict_in_batches,
)
from sep_c_training import (
    aore_components,
    create_prediction_cache,
    mix_cached_predictions,
    select_and_train_gate,
    set_global_determinism,
    train_experts,
)


TARGET_COLUMN = "ln_peak_intensity"
RARE_THRESHOLD = np.log(10)
EXPECTED_INPUT_COUNT = 22


def load_sep_c_dataset(data_directory):
    """Read the existing pseudolabeled splits without fitting any preprocessing."""

    split_arrays = {}
    expected_columns = None

    for split_name in ("training", "testing"):
        data_path = Path(data_directory) / f"sep_10mev_{split_name}_pseudo_labeled.csv"

        # Retain pandas' original parsing, followed by conversion to float32.
        # Release each DataFrame before reading the next split to avoid overlap.
        dataframe = pd.read_csv(data_path)
        columns = list(dataframe.columns)

        if TARGET_COLUMN not in columns:
            raise ValueError(f"{data_path} is missing {TARGET_COLUMN}.")

        if expected_columns is not None and columns != expected_columns:
            raise ValueError(f"Column order differs in {data_path}.")

        expected_columns = columns
        features = dataframe.drop(columns=[TARGET_COLUMN]).to_numpy(dtype="float32")
        targets = dataframe[TARGET_COLUMN].to_numpy(dtype="float32").reshape(-1, 1)
        del dataframe

        if features.shape[1] != EXPECTED_INPUT_COUNT:
            raise ValueError(f"Expected {EXPECTED_INPUT_COUNT} input features in {data_path}.")

        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError(f"{data_path} contains nonfinite inputs or targets.")

        split_arrays[split_name] = (features, targets)

    # These files already contain the normalized inputs and logarithmic target.
    # Preserve the October split exactly; testing rows never join training here.
    training_features, training_targets = split_arrays["training"]
    testing_features, testing_targets = split_arrays["testing"]

    return training_features, training_targets, testing_features, testing_targets


def load_saved_ensemble(settings):
    """Load inference state, including saved density profiles, without refitting."""

    ensemble_path = settings.model_directory / "ensemble.keras"

    if not ensemble_path.is_file():
        raise FileNotFoundError(
            f"Saved ensemble not found: {ensemble_path}. Train this variant first, "
            "or set SAVED_MODEL_DIRECTORY to a matching SEP-C model folder."
        )

    legacy_loss = (
        WeightedExpertMSEPCCLoss if settings.regression_gate else ExpertMSEPCCLoss
    )

    # October archives use the SEPDensityTrust registry names. Map their density
    # layer to the bounded implementation while retaining every saved reference.
    custom_objects = {
        "SEPDensityTrust>NormalizedGaussianKDE": NormalizedGaussianKDE,
        "SEPDensityTrust>NormalizeExpertWeights": NormalizeExpertWeights,
        "SEPDensityTrust>ExpertMSEPCCLoss": legacy_loss,
        "NormalizedGaussianKDE": NormalizedGaussianKDE,
        "NormalizeExpertWeights": NormalizeExpertWeights,
        "ExpertMSEPCCLoss": legacy_loss,
    }

    # Keras merges its global registry again when loading nested models. If an
    # October script was already imported, a scope alone can select its old layer.
    # Override these aliases only for this load, then restore the caller's registry.
    registry = keras.utils.get_custom_objects()
    missing_entry = object()
    previous_entries = {
        name: registry.get(name, missing_entry) for name in custom_objects
    }

    try:
        registry.update(custom_objects)

        with keras.utils.custom_object_scope(custom_objects):
            ensemble = keras.models.load_model(ensemble_path, compile=False)
    finally:
        for name, previous_value in previous_entries.items():
            if previous_value is missing_entry:
                registry.pop(name, None)
            else:
                registry[name] = previous_value

    # A SEP-E archive can have matching layer names but incompatible inputs.
    # Check the dataset width before accepting any saved model for this runner.
    if ensemble.input_shape[-1] != EXPECTED_INPUT_COUNT:
        raise ValueError(f"The saved SEP-C ensemble must have {EXPECTED_INPUT_COUNT} inputs.")

    common_expert = ensemble.get_layer("common_expert")
    rare_expert = ensemble.get_layer("rare_expert")
    gate = ensemble.get_layer("gate")

    # Fail clearly if a different experiment's archive was selected. This avoids
    # silently evaluating a gate-only model as a density ensemble or vice versa.
    density_layers = [
        layer for layer in ensemble.layers if isinstance(layer, NormalizedGaussianKDE)
    ]

    if bool(density_layers) != settings.density_at_inference:
        raise ValueError("The saved ensemble's density setting does not match this script.")

    sigmoid_output = gate.layers[-1].get_config().get("activation") == "sigmoid"

    if sigmoid_output != settings.sigmoid_gate:
        raise ValueError("The saved gate's output activation does not match this script.")

    profiles = None

    if settings.density_at_inference:
        profiles = []

        for layer_name in ("common_expert_density", "rare_expert_density"):
            density_layer = ensemble.get_layer(layer_name)
            density_layer.query_chunk_size = settings.density_query_chunk_size
            density_layer.reference_chunk_size = settings.density_reference_chunk_size

            profiles.append({
                "training_targets": np.asarray(
                    density_layer.training_targets_list, dtype="float32"
                ),
                "bandwidth": density_layer.bandwidth,
                "maximum_density": density_layer.maximum_density,
            })

    print(f"Loaded {ensemble_path}; skipping searches, training, and density-profile fitting.")
    return ensemble, common_expert, rare_expert, gate, profiles


def save_experiment(ensemble, common_expert, rare_expert, gate, settings):
    model_directory = settings.model_directory
    model_directory.mkdir(parents=True, exist_ok=True)

    # Keep the separate networks for inspection, and save the complete ensemble
    # so a later inference run can recover its original density normalization.
    common_expert.save(model_directory / "common_expert.keras")
    rare_expert.save(model_directory / "rare_expert.keras")
    gate.save(model_directory / "gate.keras")
    ensemble.save(model_directory / "ensemble.keras")

    print(f"Saved networks and bounded inference ensemble to {model_directory}")


def print_instance_table(
    targets, cache, trust, mixture_weights, predictions, region_name, mask
):
    """Show per-instance mixing only when the entry script explicitly enables it."""

    target_values = np.asarray(targets).reshape(-1)
    print(f"\n{region_name} samples")
    print(
        f"{'Index':>6} {'Target':>10} {'Common':>10} {'Rare':>10} "
        f"{'Gate C':>9} {'Gate R':>9} {'Quality C':>10} {'Quality R':>10} "
        f"{'Weight C':>10} {'Weight R':>10} {'Prediction':>11} {'Error':>10}"
    )

    for sample_index in np.flatnonzero(mask):
        print(
            f"{sample_index:6d} {target_values[sample_index]:10.4f} "
            f"{cache.predictions[sample_index, 0]:10.4f} "
            f"{cache.predictions[sample_index, 1]:10.4f} "
            f"{trust[sample_index, 0]:9.4f} {trust[sample_index, 1]:9.4f} "
            f"{cache.density_quality[sample_index, 0]:10.4f} "
            f"{cache.density_quality[sample_index, 1]:10.4f} "
            f"{mixture_weights[sample_index, 0]:10.4f} "
            f"{mixture_weights[sample_index, 1]:10.4f} "
            f"{predictions[sample_index]:11.4f} "
            f"{predictions[sample_index] - target_values[sample_index]:10.4f}"
        )


def report_split(split_name, gate, features, targets, cache, settings):
    """Share one set of gate predictions between metrics and trust diagnostics."""

    trust = predict_in_batches(gate, features, settings.prediction_batch_size)
    predictions, mixture_weights = mix_cached_predictions(
        trust, cache, use_density=settings.density_at_inference,
        normalize_trust=settings.sigmoid_gate,
    )

    target_values = np.asarray(targets).reshape(-1)
    rare_mask = target_values >= RARE_THRESHOLD
    overall_error, rare_error, aore = aore_components(targets, predictions, rare_mask)

    print(f"\n{split_name} ensemble results ({len(target_values)} samples)")
    print(f"Overall MAE: {overall_error:.6f}")
    print(f"Rare MAE:    {rare_error:.6f}")
    print(f"AORE:        {aore:.6f}")

    # The diagnostic compares which expert is favored in each true region. It
    # reuses the same predictions/qualities instead of evaluating the KDE again.
    for region_name, region_mask, expected_expert in (
        ("Common", ~rare_mask, 0),
        ("Rare", rare_mask, 1),
    ):
        if not np.any(region_mask):
            continue

        other_expert = 1 - expected_expert
        region_trust = trust[region_mask]
        region_quality = cache.density_quality[region_mask]

        gate_favors_region = region_trust[:, expected_expert] > region_trust[:, other_expert]
        quality_favors_region = (
            region_quality[:, expected_expert] > region_quality[:, other_expert]
        )

        print(
            f"{region_name}: mean gate C/R={region_trust.mean(axis=0)}, "
            f"mean final weights C/R={mixture_weights[region_mask].mean(axis=0)}"
        )
        print(f"  Gate favors the region's expert: {100 * gate_favors_region.mean():.2f}%")

        if settings.density_at_inference:
            print(
                f"  Quality favors that expert: {100 * quality_favors_region.mean():.2f}%; "
                f"both favor it: {100 * (gate_favors_region & quality_favors_region).mean():.2f}%"
            )

    if settings.print_instance_tables and split_name == "Test":
        elevated_mask = (target_values >= 0) & ~rare_mask

        for region_name, mask in (("Elevated", elevated_mask), ("Rare", rare_mask)):
            print_instance_table(
                targets, cache, trust, mixture_weights, predictions, region_name, mask
            )

    metrics = {"overall_mae": overall_error, "rare_mae": rare_error, "aore": aore}
    return metrics, predictions


def show_visualizations(
    common_expert,
    rare_expert,
    gate,
    training_features,
    training_targets,
    testing_features,
    testing_targets,
    testing_predictions,
    settings,
):
    """Run optional exploratory plots without changing any evaluated sample set."""

    import imbal

    features = np.concatenate([training_features, testing_features], axis=0)
    targets = np.concatenate([training_targets, testing_targets], axis=0)
    sample_limit = settings.visualization_sample_limit

    if sample_limit is not None and len(features) > sample_limit:
        if sample_limit <= 30:
            raise ValueError("The visualization sample limit must exceed t-SNE perplexity 30.")

        # Choose evenly spaced label ranks so rare target ranges remain visible.
        # This sampling affects only plots; training and metrics use the full data.
        label_order = np.argsort(targets.reshape(-1))
        selected_ranks = np.linspace(0, len(label_order) - 1, sample_limit, dtype=int)
        selected_indices = label_order[selected_ranks]
        features = features[selected_indices]
        targets = targets[selected_indices]

    print(f"Running three optional t-SNE plots with {len(features)} samples each.")

    for model, representation_name in (
        (common_expert, "common_expert_representation"),
        (rare_expert, "rare_expert_representation"),
        (gate, "gate_representation"),
    ):
        representation = keras.Model(model.input, model.get_layer(representation_name).output)

        # The model already ends at the requested representation. Explicitly
        # select its last layer rather than silently visualizing the prior layer.
        imbal.regression.tsne_visualization(
            representation, features, targets, representation_layer_index=-1
        )

    imbal.regression.plot_true_vs_predictions(testing_targets, testing_predictions)


def run_experiment(settings):
    """Execute one original experiment with bounded and reused calculations."""

    set_global_determinism(settings.seed)
    training_features, training_targets, testing_features, testing_targets = (
        load_sep_c_dataset(settings.data_directory)
    )

    training_regions = (training_targets.reshape(-1) >= RARE_THRESHOLD).astype("int32")
    print(f"Experiment: {settings.run_name}")
    print(f"Training: {len(training_features)}; test: {len(testing_features)}")
    print(f"Common / rare training counts: {np.bincount(training_regions, minlength=2)}")
    print(f"KDE workspace blocks: {settings.density_query_chunk_size} queries by "
          f"{settings.density_reference_chunk_size} references")

    if settings.load_saved_model:
        ensemble, common_expert, rare_expert, gate, profiles = load_saved_ensemble(settings)
        training_cache = create_prediction_cache(
            common_expert, rare_expert, training_features, profiles, settings
        )
    else:
        profiles = None

        if settings.density_at_inference:
            print("Building common and rare density profiles once.")
            profiles = [
                build_density_profile(training_targets[training_regions == region], settings)
                for region in (0, 1)
            ]

        common_expert, rare_expert = train_experts(
            training_features, training_targets, training_regions, settings
        )

        print("Caching frozen-expert predictions and qualities for all gate candidates.")
        training_cache = create_prediction_cache(
            common_expert, rare_expert, training_features, profiles, settings
        )

        gate = select_and_train_gate(
            training_features, training_targets, training_regions, training_cache, settings
        )

        ensemble = build_ensemble(common_expert, rare_expert, gate, profiles, settings)
        save_experiment(ensemble, common_expert, rare_expert, gate, settings)

    # Metrics and diagnostic tables share these small caches. Inference still
    # works directly through ensemble.keras, whose own KDE layers are bounded.
    testing_cache = create_prediction_cache(
        common_expert, rare_expert, testing_features, profiles, settings
    )

    training_metrics, _ = report_split(
        "Training", gate, training_features, training_targets, training_cache, settings
    )
    testing_metrics, testing_predictions = report_split(
        "Test", gate, testing_features, testing_targets, testing_cache, settings
    )

    # Store each new experiment's measurements under November, even when the
    # model source is an explicitly selected, read-only October archive folder.
    from sep_c_settings import EXPERIMENT_DIRECTORY

    results_directory = EXPERIMENT_DIRECTORY / "saved_results"
    results_directory.mkdir(parents=True, exist_ok=True)
    result_path = results_directory / f"{settings.run_name}.json"

    result_path.write_text(json.dumps({
        "run_name": settings.run_name,
        "loaded_saved_model": settings.load_saved_model,
        "model_directory": str(settings.model_directory),
        "training": training_metrics,
        "test": testing_metrics,
    }, indent=2) + "\n", encoding="utf-8")

    if settings.show_visualizations:
        show_visualizations(
            common_expert, rare_expert, gate,
            training_features, training_targets, testing_features, testing_targets,
            testing_predictions, settings,
        )

    return ensemble, common_expert, rare_expert, gate
