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

| Method               | Representation Layer | Epochs   | Time (s) | Rare F1 | Rare AUROC |
| -------------------- | -------------------- | -------- | -------- | ------- | ---------- |
| ==Regular w/o AE==   | ==N/A==              | 388      | 26.10    | 0.2222  | 0.9586     |
| ==Balanced w/o AE==  | ==N/A==              | 135      | 8.92     | 0.1891  | 0.9311     |
| ==Decoupled w/o AE== | ==-2==               | 373/53   | 26.20    | 0.4285  | 0.9746     |
| ==Decoupled w/o AE== | ==-3==               | 317/35   | 21.80    | 0.0809  | 0.9468     |
| Regular w/ AE        | -2                   | 1299     | 92.42    | 0.4999  | 0.8802     |
| Balanced w/ AE       | -2                   | 334      | 23.93    | 0.4642  | 0.9144     |
| Decoupled w/ AE      | -2                   | 1342/306 | 114.49   | 0.4799  | 0.9137     |
| Regular w/ AE        | -3                   | 912      | 62.22    | 0.4999  | 0.8964     |
| Balanced w/ AE       | -3                   | 401      | 27.98    | 0.4905  | 0.9313     |
| Decoupled w/ AE      | -3                   | 1195/140 | 91.19    | 0.4727  | 0.9789     |
## Regression on `AgeDB`

| Method           | Representation Layer | Epochs  | Time (s) | Frequent MSE | Rare MSE |
| ---------------- | -------------------- | ------- | -------- | ------------ | -------- |
| Regular w/o AE   | N/A                  | 38      | 20.12    | 177.8285     | 545.1328 |
| Balanced w/o AE  | N/A                  | 43      | 23.08    | 245.7015     | 501.7611 |
| Decoupled w/o AE | -2                   | 46/41   | 38.40    | 228.0658     | 504.6126 |
| Decoupled w/o AE | -3                   | 39/31   | 32.91    | 269.7757     | 442.1957 |
| Regular w/ AE    | -2                   | 98      | 82.33    | 186.0832     | 539.1149 |
| Balanced w/ AE   | -2                   | 187     | 135.20   | 245.0360     | 470.3938 |
| Decoupled w/ AE  | -2                   | 120/187 | 156.80   | 269.3215     | 573.4518 |
| Regular w/ AE    | -3                   | 96      | 75.74    | 181.8075     | 558.9628 |
| Balanced w/ AE   | -3                   | 228     | 160.28   | 275.1886     | 528.5124 |
| Decoupled w/ AE  | -3                   | 199/208 | 211.02   | 237.7709     | 468.4099 |
## Classification on `AgeDB`

| Method           | Representation Layer | Epochs  | Time (s) | Rare F1 | Rare AUROC |
| ---------------- | -------------------- | ------- | -------- | ------- | ---------- |
| Regular w/o AE   | N/A                  | 29      | 18.18    | 0.0159  | 0.5440     |
| Balanced w/o AE  | N/A                  | 27      | 18.15    | 0.1251  | 0.6079     |
| Decoupled w/o AE | -2                   | 26/21   | 26.16    | 0.1005  | 0.5282     |
| Decoupled w/o AE | -3                   | 27/21   | 24.07    | 0.0991  | 0.5622     |
| Regular w/ AE    | -2                   | 580     | 414.57   | 0.1019  | 0.5365     |
| Balanced w/ AE   | -2                   | 350     | 254.89   | 0.1240  | 0.6309     |
| Decoupled w/ AE  | -2                   | 587/125 | 469.49   | 0.1056  | 0.5978     |
| Regular w/ AE    | -3                   | 858     | 609.09   | 0.0     | 0.5846     |
| Balanced w/ AE   | -3                   |         |          |         |            |
| Decoupled w/ AE  | -3                   |         |          |         |            |

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
	- Documentation overhaul (WIP) $\times$
- **Later on:** Refactoring / rewrite of `generate_decoder_branch` $\times$
- For SEP-C regression plots, adjust $\checkmark$
	- Vertical line for ln(10), both ln(10) lines should be gray $\checkmark$
	- Change green dots $\rightarrow$ red $\checkmark$
- Get rid of `override_second_stage_compile_parameters` $\checkmark$
- Can Callbacks be 'deep copied' before fit is performed? If so, do that $\times$
- First and second stage of decoupled, *same number of epochs now* $\checkmark$
- `stratify_batches` should be a parameter for the fit functions $\checkmark$
## Notes
- Logic errors found in `decoupled_fit`
	- Stage one sample weighting was not instance based when validation data was provided