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

| Method               | Representation Layer | Epochs  | Time (s) | Frequent MSE | Rare MSE | Average |
| -------------------- | -------------------- | ------- | -------- | ------------ | -------- | ------- |
| ==Regular w/o AE==   | ==N/A==              | 1124    | 71.13    | 0.4685       | 14.4414  | 7.4550  |
| ==Balanced w/o AE==  | ==N/A==              | 95      | 7.27     | 8.0013       | 2.1646   | 5.0830  |
| ==Decoupled w/o AE== | ==-2==               | 1523/24 | 102.64   | 2.7540       | 5.5742   | 4.1641  |
| ==Decoupled w/o AE== | ==-3==               | 1162/32 | 74.67    | 4.5330       | 3.2080   | 3.8705  |
| Regular w/ AE        | -2                   | 391     | 28.66    | 0.2714       | 12.2392  | 6.2553  |
| Balanced w/ AE       | -2                   | 121     | 9.51     | 7.8233       | 2.2784   | 5.0509  |
| Decoupled w/ AE      | -2                   | 379/337 | 48.67    | 6.4606       | 2.8140   | 4.6372  |
| Regular w/ AE        | -3                   | 1478    | 120.95   | 0.3815       | 17.1761  | 8.7788  |
| Balanced w/ AE       | -3                   | 163     | 12.71    | 7.5923       | 2.1908   | 4.8916  |
| Decoupled w/ AE      | -3                   | 1625/98 | 127.37   | 5.5401       | 2.2220   | 3.8811  |
## Classification on SEP-C

| Method               | Representation Layer | Epochs   | Time (s) | Rare F1 | Rare AUROC |
| -------------------- | -------------------- | -------- | -------- | ------- | ---------- |
| ==Regular w/o AE==   | ==N/A==              | 388      | 26.10    | 0.2222  | 0.9586     |
| ==Balanced w/o AE==  | ==N/A==              | 135      | 8.92     | 0.1891  | 0.9311     |
| ==Decoupled w/o AE== | ==-2==               | 373/53   | 26.20    | 0.4285  | 0.9746     |
| ==Decoupled w/o AE== | ==-3==               | 317/35   | 21.80    | 0.0809  | 0.9468     |
| Regular w/ AE        | -2                   | 1145     | 90.42    | 0.5185  | 0.9460     |
| Balanced w/ AE       | -2                   | 334      | 23.93    | 0.4642  | 0.9144     |
| Decoupled w/ AE      | -2                   | 1342/306 | 114.49   | 0.4799  | 0.9137     |
| Regular w/ AE        | -3                   | 912      | 62.22    | 0.4999  | 0.8964     |
| Balanced w/ AE       | -3                   | 401      | 27.98    | 0.4905  | 0.9313     |
| Decoupled w/ AE      | -3                   | 1195/140 | 91.19    | 0.4727  | 0.9789     |
## Regression on `AgeDB`

| Method           | Representation Layer | Epochs | Time (s) | Frequent MSE | Rare MSE | Average |
| ---------------- | -------------------- | ------ | -------- | ------------ | -------- | ------- |
| Regular w/o AE   | N/A                  |        |          |              |          |         |
| Balanced w/o AE  | N/A                  |        |          |              |          |         |
| Decoupled w/o AE | -2                   |        |          |              |          |         |
| Decoupled w/o AE | -3                   |        |          |              |          |         |
| Regular w/ AE    | -2                   |        |          |              |          |         |
| Balanced w/ AE   | -2                   |        |          |              |          |         |
| Decoupled w/ AE  | -2                   |        |          |              |          |         |
| Regular w/ AE    | -3                   |        |          |              |          |         |
| Balanced w/ AE   | -3                   |        |          |              |          |         |
| Decoupled w/ AE  | -3                   |        |          |              |          |         |
## Classification on MNIST

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
 - `AgeDB` on table $\checkmark$
	- Documentation overhaul $\checkmark$
		- Just missing `AgeDB` classfication $\times$

- For SEP-C regression plots, adjust $\checkmark$
	- Vertical line for ln(10), both ln(10) lines should be gray $\checkmark$
	- Change green dots $\rightarrow$ red $\checkmark$
- Get rid of `override_second_stage_compile_parameters` $\checkmark$
- First and second stage of decoupled, *same number of epochs now* $\checkmark$
- `stratify_batches` should be a parameter for the fit functions $\checkmark$
## Notes
- Logic errors found in `decoupled_fit`
	- Stage one sample weighting was not instance based when validation data was provided
- I'm skeptical that `AgeDB` is a good dataset for the purposes of what we are trying to demonstrate in the documentation.
	- Should these remain "bare bones" examples?
	- Should I normalize output?
	- Should I normalize input?
# 2/10/26
## Tasks
- For balanced/decoupled fit, ensure validation data is *always* weighted in some manner when weights are not supplied $\checkmark$
	- Both classification and regression $\checkmark$
- Red and green need to be swapped for regression $\checkmark$
	- `AgeDB` needs this too (!!!)
		- Done for SEP-C regression $\checkmark$
- cRT $\rightarrow$ rRT in documentation tables $\checkmark$
- For regression tables, one more column, average of rare and frequent MSE $\checkmark$
- MNIST for image classification
- Use [Resnet-like](https://arxiv.org/pdf/1512.03385) architecture for `AgeDB` model (regression)
- Documentation overhaul
- **Later on:** Refactoring / rewrite of `generate_decoder_branch` $\times$
- Can Callbacks be 'deep copied' before fit is performed? If so, do that $\times$