## Last Meeting's Tasks
- Refactoring / rewrite of `generate_decoder_branch`
- Decoupled with AE, use AE model for first stage, original model for second stage $\checkmark$
	- `StopGradient`?
- Validation sets having sample weights (handle all possible combinations?) $\checkmark$
- Add check for $n' \neq n$ (sum of provided weights does not equal number of samples, in fit functions) $\checkmark$
- Fixed bug with metrics being duplicated (this was an UNTRACABLE bug. Thanks Python.) $\checkmark$
# 2/2/26
## Tasks:
- `EarlyStopping` tables (see below) $\checkmark$
- For validation, only necessary to support the same data types as TF $\checkmark$
- Bug with Metrics causing crash for decoupled fit $\checkmark$
- Documentation overhaul (WIP)
- **Later on:** Refactoring / rewrite of `generate_decoder_branch`
## Notes:

---
Stratify sampling is on for all tables below. Validation split = 0.2. Used `EarlyStopping` with a patience of 20. Learning rate 4e-4.
## Regression on SEP-C

| Method               | Representation Layer | Epochs   | Time (s) | Frequent MSE | Rare MSE |
| -------------------- | -------------------- | -------- | -------- | ------------ | -------- |
| ==Regular w/o AE==   | ==N/A==              | 1220     | 78.29    | 0.3806       | 14.7851  |
| ==Balanced w/o AE==  | ==N/A==              | 95       | 6.82     | 8.0030       | 2.1647   |
| ==Decoupled w/o AE== | ==-2==               | 1161/30  | 75.29    | 2.7675       | 5.4321   |
| ==Decoupled w/o AE== | ==-3==               | 1401/35  | 89.16    | 4.6619       | 2.8369   |
| Regular w/ AE        | -2                   | 394      | 28.84    | 0.2706       | 12.1286  |
| Balanced w/ AE       | -2                   | 121      | 9.15     | 7.8233       | 2.2785   |
| Decoupled w/ AE      | -2                   | 391/283  | 46.99    | 6.4371       | 2.8791   |
| Regular w/ AE        | -3                   | 515      | 37.39    | 0.2871       | 11.1098  |
| Balanced w/ AE       | -3                   | 163      | 12.91    | 7.5920       | 2.1879   |
| Decoupled w/ AE      | -3                   | 2065/109 | 166.61   | 5.6786       | 2.2497   |
## Classification on SEP-C

| Method               | Representation Layer | Epochs | Time (s) | Rare F1 | Rare AUROC |
| -------------------- | -------------------- | ------ | -------- | ------- | ---------- |
| ==Regular w/o AE==   | ==N/A==              |        |          |         |            |
| ==Balanced w/o AE==  | ==N/A==              |        |          |         |            |
| ==Decoupled w/o AE== | ==-2==               |        |          |         |            |
| ==Decoupled w/o AE== | ==-3==               |        |          |         |            |
| Regular w/ AE        | -2                   |        |          |         |            |
| Balanced w/ AE       | -2                   |        |          |         |            |
| Decoupled w/ AE      | -2                   |        |          |         |            |
| Regular w/ AE        | -3                   |        |          |         |            |
| Balanced w/ AE       | -3                   |        |          |         |            |
| Decoupled w/ AE      | -3                   |        |          |         |            |
## Regression on `AgeDB`

| Method           | Representation Layer | Epochs | Time (s) | Frequent MSE | Rare MSE |
| ---------------- | -------------------- | ------ | -------- | ------------ | -------- |
| Regular w/o AE   | N/A                  |        |          |              |          |
| Balanced w/o AE  | N/A                  |        |          |              |          |
| Decoupled w/o AE | -2                   |        |          |              |          |
| Decoupled w/o AE | -3                   |        |          |              |          |
| Regular w/ AE    | -2                   |        |          |              |          |
| Balanced w/ AE   | -2                   |        |          |              |          |
| Decoupled w/ AE  | -2                   |        |          |              |          |
| Regular w/ AE    | -3                   |        |          |              |          |
| Balanced w/ AE   | -3                   |        |          |              |          |
| Decoupled w/ AE  | -3                   |        |          |              |          |
## Classification on `AgeDB`

| Method           | Representation Layer | Epochs | Time (s) | Rare F1 | Rare AUROC |
| ---------------- | -------------------- | ------ | -------- | ------- | ---------- |
| Regular w/o AE   | N/A                  |        |          |         |            |
| Balanced w/o AE  | N/A                  |        |          |         |            |
| Decoupled w/o AE | -2                   |        |          |         |            |
| Decoupled w/o AE | -3                   |        |          |         |            |
| Regular w/ AE    | -2                   |        |          |         |            |
| Balanced w/ AE   | -2                   |        |          |         |            |
| Decoupled w/ AE  | -2                   |        |          |         |            |
| Regular w/ AE    | -3                   |        |          |         |            |
| Balanced w/ AE   | -3                   |        |          |         |            |
| Decoupled w/ AE  | -3                   |        |          |         |            |

---

## Notes:
- Although issue with re-compilation was fixed (sort of band-aid fix), should recompilation be allowed at all?
	- Do we expect the user to change the optimizer, loss, or metrics?
	- Disallowing this would prevent the need for a "band-aid" fix
- For second stage of decoupled fit
	- Callbacks are stateful, and don't reset at the start of a model fit
	- Therefore, passing a `EarlyStopping` from stage one to stage two will almost always result in the second stage ending after `patience` epochs have passed
	- Current fix: Passed callbacks are only used during the first stage by default. Callbacks for second stage will have to be explicitly provided by the user.
		- Behavior of decoupled fit is no longer second stage epochs = half of provided first stage epochs, now second stage epochs = half of length of history of first stage
			- Adapts to `EarlyStopping` better
	- Thoughts?
- Should `stratify_batches` be a parameter to `fit` instead of compile?
	- Unlike `generate_decoder_branch`, it is not something that affects the model itself, just the data. Hence, I feel now that `stratify_batches` is a better fit as a parameter to `fit`
# 2/5/26
## Tasks:
 - `AgeDB` on table
	- Documentation overhaul (WIP)
- **Later on:** Refactoring / rewrite of `generate_decoder_branch`
- For SEP-C regression plots, adjust $\checkmark$
	- Vertical line for ln(10), both ln(10) lines should be gray $\checkmark$
	- Change green dots $\rightarrow$ red $\checkmark$
- Get rid of `override_second_stage_compile_parameters` $\checkmark$
- Can Callbacks be 'deep copied' before fit is performed? If so, do that
- First and second stage of decoupled, *same number of epochs now* $\checkmark$
- `stratify_batches` should be a parameter for the fit functions

## Notes
- Logic errors found in `decoupled_fit`
	- Stage one sample weighting was not instance based when validation data was provided

## Regression on SEP-C

| Method               | Representation Layer | Epochs  | Time (s) | Frequent MSE | Rare MSE |
| -------------------- | -------------------- | ------- | -------- | ------------ | -------- |
| ==Regular w/o AE==   | ==N/A==              | 725     | 40.50    | 0.2802       | 12.3094  |
| ==Balanced w/o AE==  | ==N/A==              | 359     | 20.43    | 4.2391       | 2.0572   |
| ==Decoupled w/o AE== | ==-2==               | 126/55  | 11.53    | 3.63110      | 3.15091  |
| ==Decoupled w/o AE== | ==-3==               | 584/180 | 47.40    | 3.37504      | 2.93389  |
| Regular w/ AE        | -2                   | 1206    | 86.22    | 0.26889      | 11.17252 |
| Balanced w/ AE       | -2                   | 258     | 21.77    | 4.79717      | 2.43208  |
| Decoupled w/ AE      | -2                   | 487/827 | 78.37    | 3.38522      | 3.35404  |
| Regular w/ AE        | -3                   | 970     | 65.78    | 0.27937      | 12.48287 |
| Balanced w/ AE       | -3                   | 532     | 37.42    | 3.62584      | 2.55779  |
| Decoupled w/ AE      | -3                   | 257/95  | 23.05    | 6.14561      | 3.24729  |
## Classification on SEP-C

| Method               | Representation Layer | Epochs             | Time (s) | Rare F1 | Rare AUROC |
| -------------------- | -------------------- | ------------------ | -------- | ------- | ---------- |
| ==Regular w/o AE==   | ==N/A==              | 926                | 51.27    | 0.5517  | 0.9386     |
| ==Balanced w/o AE==  | ==N/A==              | 305                | 16.48    | 0.2926  | 0.9511     |
| ==Decoupled w/o AE== | ==-2==               | 210/102            | 19.94    | 0.2522  | 0.9366     |
| ==Decoupled w/o AE== | ==-3==               | 516/767            | 73.89    | 0.3880  | 0.9577     |
| Regular w/ AE        | -2                   | 1151               | 73.08    | 0.1249  | 0.9224     |
| Balanced w/ AE       | -2                   | 324                | 20.13    | 0.3823  | 0.9377     |
| Decoupled w/ AE      | -2                   | 523/77             | 38.07    | 0.3513  | 0.9328     |
| Regular w/ AE        | -3                   | 1522               | 94.10    | 0.6153  | 0.9402     |
| Balanced w/ AE       | -3                   | 201                | 12.80    | 0.3116  | 0.9096     |
| Decoupled w/ AE      | -3                   | 900/4126 **(???)** | 282.97   | 0.3846  | 0.9113     |
