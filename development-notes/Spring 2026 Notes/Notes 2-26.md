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

| Method           | Representation Layer | Epochs  | Time (s) | Frequent MSE | Rare MSE | Average  |
| ---------------- | -------------------- | ------- | -------- | ------------ | -------- | -------- |
| Regular w/o AE   | N/A                  | 815     | 4549.04  | 75.90783     | 285.7970 | 180.85   |
| Balanced w/o AE  | N/A                  | 203     | 1186.85  | 113.2086     | 265.0980 | 189.1533 |
| Decoupled w/o AE | -2                   | 83/45   | 703.14   | 143.9198     | 215.1342 | 179.527  |
| Decoupled w/o AE | -3                   | 1717/43 | 8943.71  | 92.8854      | 245.0631 | 168.9742 |
| Regular w/ AE    | -2                   |         |          |              |          |          |
| Balanced w/ AE   | -2                   |         |          |              |          |          |
| Decoupled w/ AE  | -2                   |         |          |              |          |          |
| Regular w/ AE    | -3                   |         |          |              |          |          |
| Balanced w/ AE   | -3                   |         |          |              |          |          |
| Decoupled w/ AE  | -3                   |         |          |              |          |          |
## Classification on MNIST
lr = 1e-4, patience=50

| Method           | Representation Layer | Epochs  | Time (s) | Rare F1 | Rare AUROC |
| ---------------- | -------------------- | ------- | -------- | ------- | ---------- |
| Regular w/o AE   | N/A                  | 212     | 45.86    | 0.0     | 0.9291     |
| Balanced w/o AE  | N/A                  | 228     | 50.17    | 0.2399  | 0.9531     |
| Decoupled w/o AE | -2                   | 311/54  | 80.53    | 0.2702  | 0.9463     |
| Decoupled w/o AE | -3                   | 268/89  | 78.88    | 0.4210  | 0.9591     |
| Regular w/ AE    | -2                   | 2948    | 1124.91  | 0.3749  | 0.8735     |
| Balanced w/ AE   | -2                   | 1671    | 697.92   | 0.3414  | 0.9559     |
| Decoupled w/ AE  | -2                   | 695/355 | 373.81   | 0.1791  | 0.9599     |
| Regular w/ AE    | -3                   | 1782    | 1500.41  | 0.0     | 0.8294     |
| Balanced w/ AE   | -3                   | 911     | 555.05   | 0.0     | 0.7240     |
| Decoupled w/ AE  | -3                   | 1665/97 | 1113.20  | 0.0     | 0.6427     |

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
	- `AgeDB` needs this too (!!!) $\checkmark$
		- Done for SEP-C regression $\checkmark$
		- Done for `AgeDB` $\checkmark$
- cRT $\rightarrow$ rRT in documentation tables $\checkmark$
- For regression tables, one more column, average of rare and frequent MSE $\checkmark$
- MNIST for image classification
- Use [Resnet-like](https://arxiv.org/pdf/1512.03385) architecture for `AgeDB` model (regression) $\checkmark$
- Documentation overhaul
- **Later on:** Refactoring / rewrite of `generate_decoder_branch` $\times$
- Can Callbacks be 'deep copied' before fit is performed? If so, do that $\times$
## Notes:
- Discussed an issue with Daniel
- Even after implementation of `ResNet` structure, I still have concerns about `AgeDB` (training loss decreases much faster than validation loss).

![[Pasted image 20260212111201.png|600]]
# 2/12/25
## Tasks:
- Lower learning rate for `AgeDB` $\checkmark$
	- Alternatively, learning rate schedule $\checkmark$
	- Alternatively, a pre-trained model $\checkmark$
	- Alternatively, SDO dataset $\checkmark$
- Check `AgeDB` regression paper $\checkmark$
	- Hyperparameters and such $\checkmark$
	- What were there results? $\checkmark$
		- **Didn't find a paper with a simple model implementation for `AgeDB`.**
- Just w/o AE comparisons for `AgeDB` $\checkmark$
- MNIST for image classification $\checkmark$
- Documentation overhaul $\checkmark$
- **Later on:** Refactoring / rewrite of `generate_decoder_branch`
# 2/17/26
## Tasks:
- Handle first
	- Update code examples in `Model` documentation $\checkmark$
	- Properly link all comparison pages (8 total) to the `imbal.classification.Model` and `imbal.regression.Model` pages (4 each) $\checkmark$
	- Emphasize for documentation for `generate_decoder_branch` that it only works on sequential model (models without skip connections / branches) $\checkmark$
		- Add check and fatal error $\checkmark$
		- Copy over explanation of rare vs. frequent from comparison of fit images to comparison of ae images $\checkmark$
	- Mute `sample_density`/`validation_densities` for classification `Model` object, and `class_weight` for regression `Model` object $\checkmark$
	- Override `fit`/`balanced_fit`/`decoupled_fit` without "unnecessary parameters". $\checkmark$
	- `decoupled_fit` is private, only `cRT` and `rRT` are accessible. $\checkmark$
	- Is there a way to specify order that functions appear in documentation $\checkmark$
		- Yes! And done $\checkmark$
- 70%
	- SEP-EC
		- Two subsets
			- all columns
			- electron and proton data only
		- Run through all `imbal` model function
			- Assume stratified sampling
			- w and w/o AE
			- Comparison table as before
			- Representation layer $-2$/$-3$
			- Regression only
	- **Later on:** MDI and wPCC
- 30%
	- `imbal` implementations
		- Function with six parameters: `predictions`, `targets`, `metric`, `maximize`,`step_size` (default $0.1$), `range` (default $(0,1)$). Finds the threshold from $(0,1)$ that produces the best results for the provided `metric`. $\checkmark$
			- Done, implemented under classification at the moment (because thresholding creates binary classes). $\checkmark$
		- Function with three parameters: `start_weights`, `end_weights`, `steps`. Returns linearly interpolated class weights for the labels. $\checkmark$
			- Done, implemented under classification $\checkmark$
			- For regression $\frac{1}{d^\alpha}$ and vary, but mostly same as above.
				- Actually, do this first and apply to SEC-EC described above
		- An extension for `balanced_fit` that takes in possible class weights and returns the class weights that minimize the validation loss.
	- Documentation and code examples for above.
	- Then, refactoring / rewrite of `generate_decoder_branch`.
# 2/17/26
## Tasks:
- Documentation for `interpolate_class_weights`, `optimize_metric`