# November 2 Dan — bounded SEP-E ensembles

This folder contains the seven optimized October 16 SEP-E experiments and two
additional quality-free variants for the professor's table. All nine use readable
shared modules with bounded density workspaces and reusable frozen-expert
calculations. The October files and shared `imbal` package are unchanged.

## Scripts for the professor's four rows

Quality means the density-based multiplier on the expert mixture. Every row below
uses MSE + PCC on both experts. The two regression rows use sigmoid gates whose
outputs are normalized before mixing, whether quality is enabled or disabled.

| Professor's row | Script | Quality in gate validation AORE | Quality at final testing |
| --- | --- | --- | --- |
| Classification gate, with quality at testing | [Classification + PCC + quality](gated-ensemble-with-density-trust-sep-e-both-experts-pcc.py) | Yes | Yes |
| Classification gate, without quality at testing | [Classification + PCC, no quality](gated-ensemble-without-density-trust-sep-e-both-experts-pcc.py) | No | No |
| Sigmoid regression gate, quality during training and testing | [Sigmoid regression + training/testing quality](gated-ensemble-regression-gate-sep-e-both-experts-pcc-density-trust-during-training-sigmoid.py) | Yes | Yes |
| Sigmoid regression gate, no quality during training or testing | [Sigmoid regression, no training/testing quality](gated-ensemble-regression-gate-sep-e-both-experts-pcc-no-quality-during-training-or-testing-sigmoid.py) | No | No |

Use the **Test** metrics printed by each run (or the `test` object in its result
JSON) to fill Overall MAE, Rare MAE, and AORE. AORE is their arithmetic mean.

The classification rows are independently selected experiments: their validation
AORE uses the same quality setting as final testing. They are not evaluations of
one identical gate checkpoint with a test-time switch. Classification training
still uses region cross-entropy in both cases. The regression rows respectively
include or exclude quality throughout gate training, validation, and testing.

Removing quality does not remove PCC or the regression experiments' reciprocal
importance weights. Those are separate parts of the original method. The new
quality-free sigmoid model normalizes `g(C)` and `g(R)` by their sum, so its direct
predictions, cached metrics, training mixture, and saved-model inference agree.

## Run an experiment

From the repository root, for example:

```powershell
.\.venv\Scripts\python.exe development-tests/11-2026/11-2-2026-Dan/gated-ensemble-regression-gate-sep-e-both-experts-pcc-density-trust-during-training.py
```

Each entry file has descriptive settings near the top. The default remains
`LOAD_SAVED_MODEL = False`, which performs the original parameter searches and
training. These are substantial searches, not a single network fit.

- CSVs are read from `development-tests/10-2026/10-16-2026-Dan` by default. Set
  `DATA_DIRECTORY` in an entry file to use another location. No CSV copies are needed.
- New networks are written to this folder's `saved_models/<run name>/`.
- Metrics are written to this folder's `saved_results/<run name>.json`.
- Set `LOAD_SAVED_MODEL = True` to evaluate a saved ensemble without retraining.
- For one of the original seven variants, to evaluate its October model, also set
  `SAVED_MODEL_DIRECTORY = OCTOBER_DIRECTORY / "saved_models" / MODEL_RUN_NAME`.
  The source archives are read without modification. Their density layers are
  mapped to the bounded implementation during loading.
- The two new variants have unique run names and no corresponding October
  archives. Train them once before setting `LOAD_SAVED_MODEL = True`.

Loading uses `ensemble.keras`, which already includes the trained experts, gate,
reference targets, bandwidths, and density normalization. It does not rebuild
density profiles from scratch. Keep the data splits aligned with the model being
evaluated; loading is not a request to refit density profiles to different data.

## All nine methods

The original seven entry filenames still match their October counterparts. The
additional filenames explicitly identify the two missing no-quality conditions.

| Entry filename | Gate objective | Expert objective | Density during gate training | Density at final inference | Gate output |
| --- | --- | --- | --- | --- | --- |
| `gated-ensemble-sep-e.py` | Region cross-entropy | MSE | No | No | Softmax |
| `gated-ensemble-with-density-trust-sep-e.py` | Region cross-entropy | MSE | No; used for validation AORE | Yes | Softmax |
| `gated-ensemble-with-density-trust-sep-e-both-experts-pcc.py` | Region cross-entropy | MSE + PCC | No; used for validation AORE | Yes | Softmax |
| `gated-ensemble-regression-gate-sep-e-both-experts-pcc-density-trust-during-training.py` | Weighted ensemble MSE | Weighted MSE + PCC | Yes | Yes | Softmax |
| `gated-ensemble-regression-gate-sep-e-both-experts-pcc-density-trust-during-training-sigmoid.py` | Weighted ensemble MSE | Weighted MSE + PCC | Yes | Yes | Sigmoid, then normalized |
| `gated-ensemble-regression-gate-sep-e-both-experts-pcc-no-quality-during-training.py` | Weighted ensemble MSE | Weighted MSE + PCC | No | Yes | Softmax |
| `gated-ensemble-regression-gate-sep-e-both-experts-pcc-no-quality-during-training-sigmoid.py` | Weighted ensemble MSE | Weighted MSE + PCC | No | Yes | Sigmoid, then normalized |
| `gated-ensemble-without-density-trust-sep-e-both-experts-pcc.py` | Region cross-entropy | MSE + PCC | No | No | Softmax |
| `gated-ensemble-regression-gate-sep-e-both-experts-pcc-no-quality-during-training-or-testing-sigmoid.py` | Weighted ensemble MSE | Weighted MSE + PCC | No | No | Sigmoid, then normalized |

The core experimental choices are retained:

- Separate common and rare experts, divided at target `ln(10)`.
- The same input features, no additional input scaling, and the same layer sizes.
- The same MSE/PCC definitions, correlation coefficient, reciprocal weighting,
  alpha candidates, class-weight candidates, seeds, folds, and training batch size.
- Original training and validation splits combined for selection and final fitting;
  the test split stays separate.
- Expert sub-bin selection for regression-gate experiments and the original
  validation-loss selection for classification-gate experiments.
- Gate selection by `(overall MAE + rare MAE) / 2`.
- In the two original `no-quality-during-training` regression variants, training
  and early-stopping AORE exclude density; final inference includes density.
  The new `no-quality-during-training-or-testing` variant excludes it at all three stages.

The expert models remain fixed across gate folds, as in October. This optimization
does not introduce out-of-fold expert retraining or change that validation design.
The frozen-expert cache is supplied to the mixture calculation, not as extra input
features to the gate network.

## What is optimized

`NormalizedGaussianKDE` evaluates at most 128 queries against 4,096 reference
targets per block by default. It still sums over every reference target. There is
no subsampling, interpolation, or bandwidth change. A float32 kernel block is at
most 2 MiB; other tensors and the TensorFlow runtime also consume memory. This
bound covers forward evaluation, including direct `ensemble(all_rows)` calls.

Prediction helpers separately batch network evaluation. NumPy density-profile
construction also chunks both axes and reuses its working buffer.

Frozen expert predictions and density scores are computed once for each split.
Gate training reuses them across batches, epochs, folds, and parameter candidates.
The existing fold-specific weighting-density cache is retained. Metrics and
diagnostic tables share predictions instead of invoking the same KDE repeatedly.

KDE reference chunking changes the order of floating-point additions. Small
numerical differences are expected; the estimator and all sample memberships are
unchanged. Such differences can affect a nearly tied early-stopping decision, so
bit-for-bit reproduction of an entire retraining campaign is not promised.

## Optional diagnostics

`SHOW_VISUALIZATIONS = False` and `PRINT_INSTANCE_TABLES = False` keep normal runs
focused on training and evaluation. Enable them to inspect representation plots
or per-instance trust, quality, mixture weight, prediction, and error.

Visualizations default to at most 5,000 label-ranked samples. Set
`VISUALIZATION_SAMPLE_LIMIT = None` for all samples. Only plots are sampled;
training, KDE reference sets, and reported metrics always use their complete sets.
The plots explicitly select each named representation layer.

## Code map

- The nine entry scripts expose the method and run settings.
- `sep_e_settings.py` defines shared defaults and paths.
- `sep_e_models.py` contains networks, losses, bounded KDEs, and cached gate trainers.
- `sep_e_training.py` contains expert selection, gate selection, and cache reuse.
- `sep_e_experiment.py` contains data loading, archive handling, metrics, and plots.
- `test_sep_e_optimizations.py` checks the optimized mathematics against October.

Run verification from the repository root:

```powershell
.\.venv\Scripts\python.exe development-tests/11-2026/11-2-2026-Dan/test_sep_e_optimizations.py
```

The checks compare KDE values, both expert losses and prediction gradients, all
four October regression-gate forward calculations, gate gradients, and an actual
Adam update. Additional checks enforce the four professor-table configurations
and the quality-free sigmoid mixture. Short training and save/load checks cover
all nine entry points.
The short smoke tests stub weighting-density estimation; they do not replace a
complete scientific retraining comparison.

After adding the two professor-table variants, all seven test cases passed,
including short training and save/load coverage for all nine scripts. The sigmoid
regression check explicitly compares training, validation, cached reporting, and
exported inference against a known normalized mixture. Full searches for the two
new variants have not been run; they still need their first trained checkpoints.

## Original seven-variant verification measurements

Checked locally with TensorFlow 2.20.0 on CPU, using the existing October saved
networks and SEP-E CSVs. These are bounded inference/training-step checks, not a
new full hyperparameter-search campaign.

| Check | Observed result |
| --- | --- |
| Five focused test cases, including seven-variant short training/save/load coverage | Passed |
| All seven October archives, original versus bounded inference on 257 rows | Maximum absolute prediction difference at most `2.4e-7` |
| Full 39,667-row direct call through the bounded regression density ensemble | 5.60 seconds; peak resident memory 448 MiB |
| Cached versus direct predictions over those 39,667 rows | Maximum absolute difference `1.9e-6` |
| Complete saved-model load, CSV read, train/test metrics and summaries | 11.04 seconds; peak resident memory 424 MiB |
| 50 warmed 32-row gate-training steps, original density graph | 0.112 seconds |
| Same 50 steps with cached frozen-expert values | 0.052 seconds; approximately 2.2 times faster |

The gate-step comparison excludes compilation and initial cache construction.
It used the saved softmax density-during-training experts and identical initial
gate weights; the two final prediction arrays matched in that check. It does not
establish a 2.2-times speedup for the entire experiment.

The end-to-end memory check used the softmax density-during-training archive and
all 39,667 development rows plus 8,260 test rows. Windows private committed memory
was about 1.79 GiB, which includes TensorFlow runtime allocation and differs from
resident memory. That run reported training AORE `0.340248` and test AORE `0.507176`.
The original unbatched 20 GB path was not deliberately rerun.
