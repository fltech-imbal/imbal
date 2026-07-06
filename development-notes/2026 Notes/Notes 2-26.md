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
			- For regression $\frac{1}{d^\alpha}$ and vary, but mostly same as above. $\checkmark$
				- Actually, do this first and apply to SEC-EC described above $\checkmark$

	- Documentation and code examples for above.
	- Then, refactoring / rewrite of `generate_decoder_branch`.
# 2/17/26
## Tasks:
- Documentation for `interpolate_class_weights`, `optimize_metric`, `reciprocal_importance_function` $\checkmark$
	- Also, refactor `reciprocal_importance_function` to work with a single alpha or a range of alphas $\checkmark$
- An extension for `balanced_fit` that takes in possible class weights and returns the class weights that minimize the validation loss. $\checkmark$
- Highlight and label "Limitations" is `generate_decoder_branch` $\checkmark$

- SEP-EC - regression problem $\checkmark$
	- Normalize feature values from $[-1, 1]$ (divide by largest magnitude) $\checkmark$
	- Borrow k-fold cross validation from Daniel $\checkmark$
		- Received from Daniel, but have not added to my code yet.
	- Two subsets $\checkmark$
		- all columns $\checkmark$
		- electron and proton data only $\checkmark$
	- Run through all `imbal` model function $\checkmark$
		- Assume stratified sampling $\checkmark$
		- w and w/o AE $\checkmark$
		- Comparison table as before $\checkmark$
		- Representation layer $-2$/$-3$ $\checkmark$
		- Regression only $\checkmark$

---
### Regression on SEP-EC w/ CME (2/24/26)
For all runs below:
- Stratified batching is enabled
	- Early stopping used with patience of 20 epochs
- Same seed used for consistency

| Method                        | LR     | Epochs   | Time (s) | MSE     |
| ----------------------------- | ------ | -------- | -------- | ------- |
| Regular w/o AE                | $1e-3$ | 558      | 92.30    | 0.04970 |
| Regular w/ AE (rep = $-2$)    | $1e-3$ | 699      | 153.25   | 0.06012 |
| Regular w/ AE (rep = $-3$)    | $1e-3$ | 643      | 136.90   | 0.05789 |
| Balanced w/o AE               | $2e-4$ | 484      | 80.55    | 0.12949 |
| Balanced w/ AE (rep = $-2$)   | $2e-4$ | 479      | 104.72   | 0.10312 |
| Balanced w/ AE (rep = $-3$)   | $2e-4$ | 591      | 130.91   | 0.13388 |
| Decoupled w/o AE (rep = $-2$) | $2e-4$ | 1139/248 | 238.93   | 0.06862 |
| Decoupled w/o AE (rep = $-3$) | $2e-4$ | 1209/308 | 244.05   | 0.07713 |
| Decoupled w/ AE (rep = $-2$)  | $2e-4$ | 1376/382 | 425.51   | 0.06475 |
| Decoupled w/ AE (rep = $-3$)  | $2e-4$ | 1183/641 | 507.16   | 0.06671 |
### Regression on SEP-EC w/o CME (2/24/26)
For all runs below:
- Stratified batching is enabled
	- Early stopping used with patience of 20 epochs
- Same seed used for consistency

| Method                        | LR     | Epochs   | Time (s) | MSE     |
| ----------------------------- | ------ | -------- | -------- | ------- |
| Regular w/o AE                | $1e-3$ | 394      | 86.59    | 0.03484 |
| Regular w/ AE (rep = $-2$)    | $1e-3$ | 443      | 132.61   | 0.03428 |
| Regular w/ AE (rep = $-3$)    | $1e-3$ | 350      | 101.04   | 0.03280 |
| Balanced w/o AE               | $2e-4$ | 423      | 96.56    | 0.08015 |
| Balanced w/ AE (rep = $-2$)   | $2e-4$ | 376      | 114.53   | 0.09639 |
| Balanced w/ AE (rep = $-3$)   | $2e-4$ | 340      | 106.54   | 0.09303 |
| Decoupled w/o AE (rep = $-2$) | $2e-4$ | 1036/118 | 256.14   | 0.03716 |
| Decoupled w/o AE (rep = $-3$) | $2e-4$ | 1105/228 | 297.78   | 0.03695 |
| Decoupled w/ AE (rep = $-2$)  | $2e-4$ | 1047/633 | 437.11   | 0.03510 |
| Decoupled w/ AE (rep = $-3$)  | $2e-4$ | 1169/287 | 387.84   | 0.03669 |

---
## Tasks:
- Bug with decoupled fits related to how balanced and decoupled fit are not split between classification and regression classes. Need to do same for standard fit for consistency and to fix some issues. $\checkmark$
- `interpolate_class_weights` code example should do steps=5 (or whatever gives \[.25, .25, .25, .25]) $\checkmark$
- If a list of alphas is is provided to `reciprocal_inportance`, use those $\checkmark$
- Use $|x| < 1$  for rare instances $\checkmark$

Add these same metrics from [this paper](https://cs.fit.edu/~pkc/papers/icmla25.pdf) for tables $\checkmark$
- Also, look at model structure from this paper
![[Pasted image 20260224163319.png|500]]

- **Later on:** MDI, wPCC, and DenseLoss
- **30%:** Refactoring / rewrite of `generate_decoder_branch`.

## Notes:
- From paper, in regards to model structure:
	- Not much about specific structure (though I am using MLP)
	- Can't use residuals with auto-decoder
> All models are implemented in TensorFlow using a residual MLP architecture. Dataset-specific hyperparameters are determined via four-fold stratified cross-validation on the training data.

---
### Regression on SEP-EC w/ CME (2/27/26)
For all runs below:
- Stratified batching is enabled
	- Early stopping used with patience of 20 epochs
- Same seed used for consistency

| Method                        | LR     | Epochs   | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----------------------------- | ------ | -------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Regular w/o AE                | $1e-3$ | 520      | 130.78   | 0.077           | 0.347             | 0.212            | 0.613         | 0.933           | 0.773          |
| Regular w/ AE (rep = $-2$)    | $1e-3$ | 548      | 174.58   | 0.075           | 0.469             | 0.272            | 0.762         | 0.950           | 0.856          |
| Regular w/ AE (rep = $-3$)    | $1e-3$ | 539      | 156.76   | 0.074           | 0.473             | 0.273            | 0.694         | 0.930           | 0.812          |
| Balanced w/o AE               | $2e-4$ | 579      | 132.97   | 0.267           | 0.575             | 0.421            | 0.276         | 0.894           | 0.585          |
| Balanced w/ AE (rep = $-2$)   | $2e-4$ | 483      | 141.42   | 0.230           | 0.579             | 0.405            | 0.292         | 0.891           | 0.591          |
| Balanced w/ AE (rep = $-3$)   | $2e-4$ | 641      | 212.27   | 0.258           | 0.498             | 0.378            | 0.304         | 0.923           | 0.613          |
| Decoupled w/o AE (rep = $-2$) | $2e-4$ | 1720/323 | 457.11   | 0.092           | 0.382             | 0.237            | 0.620         | 0.953           | 0.787          |
| Decoupled w/o AE (rep = $-3$) | $2e-4$ | 1899/322 | 508.84   | 0.170           | 0.507             | 0.338            | 0.429         | 0.896           | 0.662          |
| Decoupled w/ AE (rep = $-2$)  | $2e-4$ | 1565/537 | 496.43   | 0.096           | 0.582             | 0.339            | 0.535         | 0.855           | 0.695          |
| Decoupled w/ AE (rep = $-3$)  | $2e-4$ | 1571/887 | 1307.48  | 0.092           | 0.263             | 0.178            | 0.700         | 0.988           | 0.844          |
### Regression on SEP-EC w/o CME (2/27/26)
For all runs below:
- Stratified batching is enabled
	- Early stopping used with patience of 20 epochs
- Same seed used for consistency

| Method                        | LR     | Epochs   | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----------------------------- | ------ | -------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Regular w/o AE                | $1e-3$ | 498      | 90.33    | 0.018           | 0.236             | **0.127**        | **0.952**     | **0.985**       | **0.968**      |
| Regular w/ AE (rep = $-2$)    | $1e-3$ | 498      | 118.96   | **0.017**       | 0.370             | 0.194            | 0.946         | 0.922           | 0.934          |
| Regular w/ AE (rep = $-3$)    | $1e-3$ | 637      | 144.23   | 0.022           | 0.299             | 0.161            | 0.948         | 0.956           | 0.952          |
| Balanced w/o AE               | $2e-4$ | 566      | 105.53   | 0.091           | 0.441             | 0.265            | 0.666         | 0.919           | 0.793          |
| Balanced w/ AE (rep = $-2$)   | $2e-4$ | 423      | 99.38    | 0.211           | 0.383             | 0.297            | 0.417         | 0.968           | 0.693          |
| Balanced w/ AE (rep = $-3$)   | $2e-4$ | 391      | 90.27    | 0.164           | 0.364             | 0.264            | 0.493         | 0.971           | 0.732          |
| Decoupled w/o AE (rep = $-2$) | $2e-4$ | 1012/212 | 299.24   | 0.052           | **0.205**         | 0.128            | 0.892         | 0.987           | 0.940          |
| Decoupled w/o AE (rep = $-3$) | $2e-4$ | 935/251  | 339.22   | 0.066           | 0.256             | 0.161            | 0.851         | 0.971           | 0.911          |
| Decoupled w/ AE (rep = $-2$)  | $2e-4$ | 789/427  | 398.76   | 0.024           | 0.288             | 0.156            | 0.928         | 0.971           | 0.949          |
| Decoupled w/ AE (rep = $-3$)  | $2e-4$ | 979/291  | 462.82   | 0.080           | 0.426             | 0.253            | 0.694         | 0.933           | 0.813          |

---
# 2/27/26
## Tasks:
- RankSim
	- Two caveats
		- Non-differentiable
		- Impartial to the scaling/relative distance between features, regardless of the distances in the label space
		- Computational efficiency (is there a solution better than $O(n^2)$?)
			- $O(n\log n)$ seems feasible
		- Can we reduce redundant features? (t-SNE)
	- How to fix?
		- *Current idea:* Don't rank, normalize label/feature similarities

- Daniel k-fold addition $\checkmark$
- Updated model structure according to paper $\checkmark$
- Testing across $\alpha$ values (reciprocal) $\checkmark$
	- Picking methods from table above that are already performing well, vary alpha to see how much improvement can be achieved $\checkmark$
- DenseLoss with multiple $\alpha$ from $[0.1, 2]$, $0.1$ increment $\checkmark$
- **From Dr. Chan:**
	- for varying alpha in DenseWeight and reciprocal, pick the best row for regular, balance, and decoupled.  That is, try to optimize the 3 basic methods. $\checkmark$
		- First stage decoupled is not affected by sample weights. $\checkmark$
	- For the SEP-EC data, the rare thresholds are +/- 0.5 $\checkmark$
	- Outside `imbal`,  add variation of having an extra trained layers for decoupled fit $\times$
	- Inside `imbal`:
		- update balanced/decoupled fit to incorporate finding the "best" class/sample weights based on the validation set
			1.  Classification, iterate on different lists of class weights
			2.  Regression, iterate on different lists of sample weights via alpha for reciprocal importance
	- Fit functions will run much slower, I suggest printing status messages. I suggest an int parameter to indicate message levels, this will also help debugging, for example: 
		 0. no messages (except those from `keras`/`tensorflow`)
		 1. main steps, found epoch number based on validation, class weights (alpha in reciprocal importance) based on validation..., training on the entire training set
		 2. Different class weights, alphas, ... being evaluated
- **Later on:** MDI, wPCC
- **30%:** Refactoring / rewrite of `generate_decoder_branch`


---
### Regression on SEP-EC w/ CME (3/3/26)
For all runs below:
- Stratified batching is enabled
- Same seed used for consistency
-  Ideal weights to use based on `AORE` on 5-fold validation

| Method                        | LR     | Epochs | Weights | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----------------------------- | ------ | ------ | ------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Regular w/o AE                | $2e-4$ | 224    | ---     | 41.29    | 0.041           | 0.396             | 0.219            | 0.814         | 0.927           | 0.871          |
| Regular w/ AE (rep = $-2$)    | $2e-4$ | 224    | ---     | 58.68    | 0.038           | 0.329             | 0.184            | 0.823         | 0.925           | 0.874          |
| Regular w/ AE (rep = $-3$)    | $2e-4$ | 224    | ---     | 57.52    | 0.040           | 0.338             | 0.189            | 0.813         | 0.930           | 0.871          |
| Balanced w/o AE               | $5e-5$ |        |         |          |                 |                   |                  |               |                 |                |
| Balanced w/ AE (rep = $-2$)   | $5e-5$ |        |         |          |                 |                   |                  |               |                 |                |
| Balanced w/ AE (rep = $-3$)   | $5e-5$ |        |         |          |                 |                   |                  |               |                 |                |
| Decoupled w/o AE (rep = $-2$) | $5e-5$ |        |         |          |                 |                   |                  |               |                 |                |
| Decoupled w/o AE (rep = $-3$) | $5e-5$ |        |         |          |                 |                   |                  |               |                 |                |
| Decoupled w/ AE (rep = $-2$)  | $5e-5$ |        |         |          |                 |                   |                  |               |                 |                |
| Decoupled w/ AE (rep = $-3$)  | $5e-5$ |        |         |          |                 |                   |                  |               |                 |                |
### Regression on SEP-EC w/o CME (3/3/26)
For all runs below:
- Stratified batching is enabled
- Same seed used for consistency
- Ideal weights to use based on `val_loss` on 5-fold validation

| Method                        | LR     | Epochs | Weights                 | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----------------------------- | ------ | ------ | ----------------------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Regular w/o AE                | $2e-4$ | 118    | ---                     | 53.90    | 0.023           | 0.131             | 0.077            | 0.961         | 0.980           | 0.971          |
| Regular w/ AE (rep = $-2$)    | $2e-4$ | 118    | ---                     | 66.72    | 0.021           | 0.121             | 0.071            | 0.959         | 0.982           | 0.971          |
| Regular w/ AE (rep = $-3$)    | $2e-4$ | 118    | ---                     | 67.65    | 0.024           | 0.101             | 0.063            | 0.962         | 0.984           | 0.973          |
| Balanced w/o AE               | $5e-5$ | 305    | DenseLoss, $\alpha=0.1$ | 151.55   | 0.024           | 0.129             | 0.077            | 0.962         | 0.976           | 0.969          |
| Balanced w/ AE (rep = $-2$)   | $5e-5$ | 305    | DenseLoss, $\alpha=0.1$ | 190.96   | 0.017           | 0.109             | 0.063            | 0.965         | 0.979           | 0.972          |
| Balanced w/ AE (rep = $-3$)   | $5e-5$ | 305    | DenseLoss, $\alpha=0.1$ | 165.68   | 0.022           | 0.118             | 0.070            | 0.957         | 0.975           | 0.966          |
| Decoupled w/o AE (rep = $-2$) | $5e-5$ | 118/80 | DenseLoss, $\alpha=0.1$ | 75.33    | 0.022           | 0.115             | 0.068            | 0.952         | 0.977           | 0.964          |
| Decoupled w/o AE (rep = $-3$) | $5e-5$ | 118/80 | DenseLoss, $\alpha=0.1$ | 40.48    | 0.021           | 0.120             | 0.070            | 0.952         | 0.975           | 0.964          |
| Decoupled w/ AE (rep = $-2$)  | $5e-5$ | 118/80 | DenseLoss, $\alpha=0.1$ | 54.17    | 0.020           | 0.118             | 0.069            | 0.953         | 0.976           | 0.965          |
| Decoupled w/ AE (rep = $-3$)  | $5e-5$ | 118/80 | DenseLoss, $\alpha=0.1$ | 51.97    | 0.018           | 0.119             | 0.069            | 0.957         | 0.978           | 0.967          |

---