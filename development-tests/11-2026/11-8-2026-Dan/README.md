# November 8 · SEP-C sequential expert regression

Run `sequential_ensemble_runner_sep_c.py` in PyCharm with the project interpreter. Edit
`create_settings()` in the runner to change the experiment. The default
reads October 7's pseudolabeled training and testing CSVs, using the 22 normalized
input columns and the logarithmic target `ln_peak_intensity`.

## Training

The settings use seeds 42 through 46, five shuffled folds per seed, and a cap of
five experts. Each fold trains expert 1 with MSE. Each later expert sees every
fold-training row, weighted by the previous prefix mean's squared error. Weights
average one. A perfect fit falls back to uniform weights.

For each candidate, the runner scores its prefix mean on every held-out row once.
A drop of at least 1% in unweighted validation MSE accepts the candidate. The
first rejection ends the search. The runner then fits the accepted sequence on
all training rows using each expert's mean best epoch across folds. It computes
fresh weights after each full-data fit.

Router epoch selection uses the same folds and fold-specific expert predictions.
The final routers train on predictions from the frozen full-data experts. A
positive label means at least one longer prefix improves the individual row's
squared error. At inference, a probability of 0.5 or higher continues to the next
expert. The trust input is always zero.

## Results and reloading

Before accepting or rejecting Expert 2, the runner reports held-out metrics for
Expert 1, the two-expert mean, and the better prediction on each row using the
known label. The last row is an oracle benchmark for selective routing. The
rare-row counts and median absolute-error reduction show how widely the second
expert helps. The acceptance rule still uses unweighted validation MSE.

At the end, the console prints the held-out comparison averaged across seeds,
then one line per mean test metric with routed and full-mean columns. The JSON
stores the same averages and each seed's validation history.

The dated folder's `saved_results/sep_c_sequential_experts.json` records each
seed's attempted and accepted expert counts, routed and full-mean metrics, and
the five-seed mean test metrics. Rare rows have target values at or above ln(10).
AORE averages the overall and rare MAE.

Each seed has a ZIP archive in `saved_models/sep_c_sequential_experts/`.
`median.zip` stores the run ranked third by routed test AORE. Set
`load_saved_model=True` to evaluate that archive directly. Set
`show_visualizations=True` for the existing true-versus-predicted plot.

For predictions on new rows, `SequentialEnsemble.load(path)` in the local
`sep_c_models.py` module returns the ensemble and its metadata. Supply
normalized features in the saved `preprocessing.feature_columns` order to
`ensemble.predict(features)` or `ensemble.predict_full(features)`.

Read `sep_c_training.py` for fold
selection and refitting, `sep_c_models.py` for the networks and routing, and
`sep_c_experiment.py` for the evaluation and median-model workflow.
