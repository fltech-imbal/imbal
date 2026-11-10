# November 9 · SEP-E sequential expert regression

Run `sequential_ensemble_runner_sep_e.py` in PyCharm with the project interpreter. Edit
`create_settings()` in the runner for this dataset's run.

As in the November 2 experiment, the runner pools October 16's normalized SEP-E
training and validation CSVs before forming five shuffled folds. The supplied
test CSV provides the final evaluation split. The target is `p16.4_tplus6`,
the input has 156 features, and rare rows have targets at or above ln(10).

SEP-E uses the same expert weighting, validation acceptance, binary routing, and
five-seed evaluation procedure as SEP-C. In this folder, `sep_e_training.py`
handles fitting, `sep_e_models.py` defines the networks and routing, and
`sep_e_experiment.py` loads the data and evaluates the models.

Before accepting or rejecting Expert 2, the runner reports held-out metrics for
Expert 1, the two-expert mean, and the better prediction on each row using the
known label. The last row is an oracle benchmark for selective routing. The
rare-row counts and median absolute-error reduction show how widely the second
expert helps. The acceptance rule still uses unweighted validation MSE.

At the end, the console prints the held-out comparison averaged across seeds,
then one line per mean test metric with routed and full-mean columns. The JSON
stores the same averages and each seed's validation history.

Results are written to `saved_results/sep_e_sequential_experts.json`. The seed
archives and `median.zip` are in `saved_models/sep_e_sequential_experts/`.
Set `load_saved_model=True` to evaluate the median archive and
`show_visualizations=True` to plot its routed test predictions.
