# November 1 Dan — SEP-C pseudolabeled experiments

These four entry scripts reproduce the professor's four-row comparison using
the recent SEP-C pseudolabeled dataset. Both experts use MSE + PCC in every row.

| Row | Gate and density quality | Script |
| --- | --- | --- |
| 1 | Classification; quality at testing | `gated-ensemble-with-density-trust-sep-c-pseudolabeled-both-experts-pcc.py` |
| 2 | Classification; no quality at testing | `gated-ensemble-without-density-trust-sep-c-pseudolabeled-both-experts-pcc.py` |
| 3 | Sigmoid regression; quality during training and testing | `gated-ensemble-regression-gate-sep-c-pseudolabeled-both-experts-pcc-density-trust-during-training-sigmoid.py` |
| 4 | Sigmoid regression; no quality during training or testing | `gated-ensemble-regression-gate-sep-c-pseudolabeled-both-experts-pcc-no-quality-during-training-or-testing-sigmoid.py` |

Run each entry file normally in PyCharm using the project's Python environment.
Keep the four `sep_c_*.py` helper modules alongside the entry files.
The helpers are independent of the November 2 SEP-E folder.

## Additional experiment: MSE-only experts

These two additional files repeat the regression rows with simpler expert
training. They are separate from the four original PCC variants above.

| Gate and density quality | Script |
| --- | --- |
| Sigmoid regression; quality during training and testing | `gated-ensemble-regression-gate-sep-c-pseudolabeled-mse-experts-density-trust-during-training-sigmoid.py` |
| Sigmoid regression; no quality during training or testing | `gated-ensemble-regression-gate-sep-c-pseudolabeled-mse-experts-no-quality-during-training-or-testing-sigmoid.py` |

Each common/rare expert uses **plain, unweighted MSE** and one five-fold pass.
Validation MSE chooses the best epoch in each fold; the rounded mean of those
epoch counts determines the final refit on all training rows in that expert's
region. This means five fold fits plus one final fit per expert, with no PCC,
expert alpha search, or dual-bin selection.

The gate still searches all nine `GATE_ALPHA_CANDIDATES`, evaluates each with
five folds using ensemble AORE, and refits using the best mean AORE candidate.
Both variants retain normalized sigmoid mixing. The quality-free variant still
uses sample importance weights for gate training; it removes density quality
from the mixture during training, validation, and testing.

Both new variants use the same pseudolabeled dataset, bounded calculations,
save/load controls, and separate model/result directories as the original four.

## Dataset and outputs

The default data directory is `development-tests/10-2026/10-7-2026-Dan`:

- `sep_10mev_training_pseudo_labeled.csv`: 1,531 training rows.
- `sep_10mev_testing_pseudo_labeled.csv`: 766 held-out testing rows.

Each file contains 22 input columns and the target `ln_peak_intensity`. The files
are read directly without additional scaling or pseudolabel generation. Their
existing train/test split is preserved. Rare means target >= ln(10).

Every script starts with `LOAD_SAVED_MODEL = False` and has a unique run name.
Models are saved under this folder's `saved_models/<run_name>/`; metrics go to
`saved_results/<run_name>.json`. After training, set `LOAD_SAVED_MODEL = True`
to reload that variant. A matching explicit model directory can be supplied
through `SAVED_MODEL_DIRECTORY`.

Use the JSON **test** values `overall_mae`, `rare_mae`, and `aore` for the table.
AORE is `(overall_mae + rare_mae) / 2`. Errors are measured in the supplied
logarithmic target units, including the supplied pseudolabels.

## Preserved method and memory controls

The two common/rare experts, PCC losses, original search grids, five folds,
500-epoch limit, patience of 100, and regression expert dual-bin selection are
retained from the November implementation. The classification rows select
their gates independently; validation AORE uses each row's testing mixture.
The regression rows use their respective mixture during training, validation,
and testing. This dataset adaptation does not redesign the validation procedure.

“No quality” removes the KDE multiplier from expert mixing. It does not remove
the regression method's density-based sample importance weights. Sigmoid outputs
are normalized to mixture weights in both regression rows, including row 4.

The shared implementation retains bounded KDE query/reference chunks, batched
prediction, cached frozen-expert predictions and qualities, and reuse of fold
densities across alpha candidates. Saved ensembles contain the bounded density
layer, and loading skips model searches and density-profile fitting. Plots and
per-instance tables remain optional. These bounds apply to individual KDE
workspaces; they are not a hard limit on total Python/TensorFlow process memory.

## Checks

Run `test_sep_c_optimizations.py` in the project environment. It checks the
original SEP-C loss and KDE mathematics, exact dataset loading, the four variant
settings, sigmoid normalization, density reuse, and short training/save/load
runs. The smoke runs use small synthetic data and reduced searches, with the
importance-density estimator stubbed. They do not produce research table scores
or establish a measured performance improvement on a full SEP-C run.

## Shared training and prediction mixing

`sep_c_models.py::combine_expert_outputs` is the single mixing definition
used by the gate trainer, exported ensemble, and cached evaluation. It multiplies
gate trust by optional quality, normalizes when needed, and takes the dot product
with the two expert predictions. The layers add no learned parameters.

The callers prepare inputs and choose the experiment's quality setting:

| Caller | Expert predictions | Quality source |
| --- | --- | --- |
| `build_cached_gate_trainer` | Cached frozen-expert predictions | Cached quality when training quality is enabled |
| `build_ensemble` | Expert forward passes on new features | Bounded KDE layers when inference quality is enabled |
| `mix_cached_predictions` | Cached expert predictions | Cached quality selected by the evaluation policy |

The regression rows in the professor's table use the same quality policy during
training and testing, so their mixing graphs calculate the same function.
Any older inference-only-quality variant deliberately supplies quality only at
inference. Classification gates still train with cross-entropy; their ensemble
validation and testing both use the shared mixing function.

Cached evaluation runs the shared layers eagerly and converts their outputs to
NumPy for metrics. The saved graph retains the existing layer names and custom
normalization class. Existing archives can still be loaded. Checks compare the
formula, gate-trust gradients, trainer predictions, cached predictions, and
serialized predictions; KDE is still calculated in bounded chunks.
