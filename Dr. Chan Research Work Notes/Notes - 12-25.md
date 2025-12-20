# 12/3/25
## Tasks:
- See 11/28 email, warning if sample weights and densities are provided for `regression.balanced_fit` $\checkmark$
	- *Note: Have done the same for `class_weights` and `sample_weights` in classification case*
- Make sure ordering for parameters in documentation matches function $\checkmark$
	- class weights before sample weights $\checkmark$
	- sample weights before densities $\checkmark$
- `generate_weights` $\rightarrow$ `generate_sample_weights` $\checkmark$
- `get_densities` $\rightarrow$ `get_sample_densities` $\checkmark$
- make sure docstrings have `class_weights` and `sample_densities`, etc... $\checkmark$
- Row/column issue with MSE... rerun all three and update tables and figures $\checkmark$
- "Frequent" instead of "common" in documentation $\checkmark$
- F1score $\checkmark$
	- Split predictions and labels into one-hot vectors $\checkmark$
		- Predictions sum to 1, 1 - prediction to get "confidence" for 0 class $\checkmark$
		- *Note: This was, in fact, how it was already being done. HOWEVER, while I don't think it is explicitly stated anywhere, the metric output that TensorFlow provides seems to be an average of the per-class F1 scores, which was part of why we were receiving unexpected results*
	- Get F1score for each class, report rare and frequent F1
		- [See documentation](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/F1Score)
- AUC
	- Should be done correctly already
		- 1 is rare, 0 is frequent
	- Use toy data if struggling
	- Just confirm
- Begin looking at autoencoder in regression case
	- Look at model structure and build out decoder
	- Use autoencoder for stage 1, then freeze and swap in class

![[Pasted image 20251203122413.png]]

2 branches, loss for each branch (MSE for autoencoder branch)
- Don't worry about interface for now. Functional prototype first, interface/wrapper later.
- Decoupled/balanced fit takes priority
# 12/5/25
## Notes:
- cifar10 imbalanced $1:24$
	- regular
		- F1- 0.9795417, 0.0
		- AUC (rare): 1.00
			- 0.073361754
			-  Class 1 confidences - \[0.28495795 0.28495795 0.28495795 0.28495795 0.28495795 0.28495795
			 0.28495795 0.28495795 0.28495795 0.28495795 0.28495795 0.28495795
			 0.28495795 0.28495795 0.28495795 0.28495795 0.28495795 0.28495795
			 0.28495795 0.28495795]
		- Adjusted F1-  0.9795417 1.0
	- balanced
		- F1 - 0.9838709 0.7246376
		- AUC (rare): 1.00
			- 0.3845506
			- \[0.9588368 0.9588368 0.9588368 0.9588368 0.9588368 0.9588368 0.9588368
			 0.9588368 0.9588368 0.9588368 0.9588368 0.9588368 0.9588368 0.9588368
			 0.9588368 0.9588368 0.9588368 0.9588368 0.9588368 0.9588368]
		- Adjusted F1 - 0.9023384 1.0
	- decoupled
		- F1 - 0.9795417 0.0
		- AUC (rare): 1.00
			- 0.07008728 
			- \[0.5536695 0.5536695 0.5536695 0.5536695 0.5536695 0.5536695 0.5536695
			 0.5536695 0.5536695 0.5536695 0.5536695 0.5536695 0.5536695 0.5536695
			 0.5536695 0.5536695 0.5536695 0.5536695 0.5536695 0.5536695]
		- Adjusted F1 - 0.9795417 1.
- cifar10 imbalanced $1:120$
	- regular
		- F1 - 0.99584025 0.
		- AUC - 0.35004178
			- 0.022277392
			- \[0.00488797 0.00488797 0.00488797 0.00488797 0.00488797 0.00488797
				 0.00488797 0.00488797 0.00488797 0.00488797]
		- Adjusted F1 - 0.99584025 0.0
	- balanced
		- F1 - 0.9786689  0.28571427
		- AUC - 0.99999994
			- 0.7337203
			- \[0.7706794 0.7706794 0.7706794 0.7706794 0.7706794 0.7706794 0.7706794
			 0.7706794 0.7706794 0.7706794]
		- Adjusted F1 - 0.72659576 1.
	- decoupled
		- F1 Score - 0.99584025 0.0
		- AUC - 0.7648287
			- 0.027821876
			- \[0.01749455 0.01749455 0.01749455 0.01749455 0.01749455 0.01749455
			 0.01749455 0.01749455 0.01749455 0.01749455]
		- Adjusted F1 - 0.99584025 0.0
## Tasks:
- Show both data balances $1:24$ and $1:120$
	-  F1-score (threshold 0.5) for rare and AUC for rare for the tables
	- Confusion matrix, TSNE, and AUC for plots
- $1:24$
	- regular
		- Rare F1 - 0.0
		- AUC - 1.0
	- balanced
		- Rare F1 - 0.758
		- AUC - 1.0
	- decoupled
		- Rare F1 - 0.0
		- AUC - 0.995
	- table (algorithm, time, rare F1, rare AUC)
- $1:120$
	- regular
		- Rare F1 - 0.0
		- AUC - .548
	- balanced
		- Rare F1 - 0.513
		- AUC - 1.0
	- decoupled
		- Rare F1 - 0.0
		- AUC - 0.427
	- table (algorithm, time, rare F1, rare AUC)

# 12/8/25
- Explanation of discrepancy of F1, AUC
	- Delete "Notably, some examples have a high AUC, but low F1 score, because the threshold...." in documentation $\checkmark$
	- Mention that it is possible for F1 to be 0 while AUC remains high $\checkmark$
	- Explain F1 being 0 $\checkmark$
	- Explain AUC still being close to 1 $\checkmark$
	- Place the explanation after each table, where the values that might raise confusion are. $\checkmark$

# 12/10/25
## Notes:
- low imbalance
	- regular
		- F1 - 0.0
		- AUC - 0.865
	- balanced
		- F1 - 0.240
		- AUC - 0.864
	- decoupled
		- F1 - 0.0
		- AUC - 0.775
- high imbalance
	- regular
		- F1 - 0.0
		- AUC - 0.844
	- balanced
		- F1 - 0.092
		- AUC - 0.855
	- decoupled
		- F1 - 0.0
		- AUC - 0.703
- Questions
	- Does AE always use original sample distribution?
		- If not, might require a lot of reworking
	- Mirror versus "by block"
		- Block!
## Tasks:
- For decoupled/balanced use third to last trainable layer instead of second to last $\checkmark$
	- Could be applied to regression model as well, in which case, update that documentation as well $\checkmark$ (already third to last)
- Confusion matrix $\checkmark$
	- X axis is actual $\checkmark$
	- Y axis is predicted $\checkmark$
- Custom loss function as alternative to MSE? $\checkmark$
	- Get sample weighting working $\checkmark$
	- Also maybe alternative through former student $\checkmark$
- Implement block-based AE mirroring (if there is time)


## Notes
- Updated decouple
	- low imbalance
		- F1 - 0.211
		- AUC - 0.848
	- high imbalance
		- F1 - 0.079
		- AUC - 0.836
- CISIR (regression)
	- regular
		- common MSE - 0.01548
		- rare MSE - 1.13322
	- balanced
		- common MSE - 0.08732
		- rare MSE - 0.23823
	- decoupled
		- common MSE - 0.02212
		- rare MSE - 0.84895


# 12/12/25
## Tasks:
- Experiment with dropout layer positions in model (before representation layer is probably better than after) $\checkmark$
- Generalize MSE to work on any data shape (issue with`axis=[1,2,3]`) $\checkmark$
- Have comparison of regular/fit/balanced with and without AE $\checkmark$
	- All 6 combinations $\checkmark$
	- Table with 6 rows $\checkmark$
		- regular without AE $\checkmark$
		- regular with AE $\checkmark$
		- balanced without AE $\checkmark$
		- balanced with AE $\checkmark$
		- decoupled without AE $\checkmark$
		- decoupled with AE $\checkmark$
	- AUC and confusion matrix for each scenario $\checkmark$
## Notes:
### Low Imbalance (1:24)

| Method           | Epochs  | Time (s) | Rare F1 | Rare AUC |
| ---------------- | ------- | -------- | ------- | -------- |
| Regular w/o AE   | 30      | 10.6     | 0.0     | 0.737    |
| Regular w/ AE    | 600     | 75.5     | 0.463   | 0.945    |
| Balanced w/o AE  | 30      | 13.1     | 0.202   | 0.805    |
| Balanced w/ AE   | 600     | 164.2    | 0.667   | 0.957    |
| Decoupled w/o AE | 30/15   | 18.6     | 0.215   | 0.829    |
| Decoupled w/ AE  | 600/300 | 161.1    | 0.713   | 0.969    |

| Regular w/o AE                               |                                       |
| -------------------------------------------- | ------------------------------------- |
| ![[confusion-matrix--low.png]]               | ![[roc-curve--low.png]]               |
| Regular w/ AE                                |                                       |
| ![[confusion-matrix-regular-low-ae.png]]     | ![[roc-curve-regular-low-ae.png]]     |
| Balanced w/o AE                              |                                       |
| ![[confusion-matrix-balanced-low.png]]       | ![[roc-curve-balanced-low.png]]       |
| Balanced w/ AE                               |                                       |
| ![[confusion-matrix-balanced-low-ae 1.png]]  | ![[roc-curve-balanced-low-ae.png]]    |
| Decoupled w/o AE                             |                                       |
| ![[confusion-matrix-decoupled-low.png]]      | ![[roc-curve-decoupled-low.png]]      |
| Decoupled w/ AE                              |                                       |
| ![[confusion-matrix-decoupled-low-ae 1.png]] | ![[roc-curve-decoupled-low-ae 1.png]] |

### High Imbalance (1:120)

| Method           | Epochs  | Time (s) | Rare F1 | Rare AUC |
| ---------------- | ------- | -------- | ------- | -------- |
| Regular w/o AE   | 30      | 10.9     | 0.0     | 0.631    |
| Regular w/ AE    | 600     | 74.7     | 0.0     | 0.838    |
| Balanced w/o AE  | 30      | 12.3     | 0.016   | 0.691    |
| Balanced w/ AE   | 600     | 159.3    | 0.133   | 0.853    |
| Decoupled w/o AE | 30/15   | 17.9     | 0.0     | 0.747    |
| Decoupled w/ AE  | 600/300 | 159.2    | 0.143   | 0.823    |

| Regular w/o AE                                |                                        |
| --------------------------------------------- | -------------------------------------- |
| ![[confusion-matrix--high.png]]               | ![[roc-curve--high.png]]               |
| Regular w/ AE                                 |                                        |
| ![[confusion-matrix--high-ae 1.png]]          | ![[roc-curve--high-ae 1.png]]          |
| Balanced w/o AE                               |                                        |
| ![[confusion-matrix-balanced-high.png]]       | ![[roc-curve-balanced-high.png]]       |
| Balanced w/ AE                                |                                        |
| ![[confusion-matrix-balanced-high-ae 1.png]]  | ![[roc-curve-balanced-high-ae 1.png]]  |
| Decoupled w/o AE                              |                                        |
| ![[confusion-matrix-decoupled-high.png]]      | ![[roc-curve-decoupled-high.png]]      |
| Decoupled w/ AE                               |                                        |
| ![[confusion-matrix-decoupled-high-ae 1.png]] | ![[roc-curve-decoupled-high-ae 1.png]] |
 
## Tasks:
- `DatasetWithBatching` convert to work with NumPy arrays instead of tensors $\checkmark$
	- Also, make `DatasetWithBatching` work for models with multiple inputs and outputs $\checkmark$
		- Parameter for multi-input and multi-output, default to false $\checkmark$
		- Parameter for index of class/label output, default to 0 $\checkmark$
		- Multi-input/output passed as list $\checkmark$
- Balanced/decoupled fit also need multi-input/output, and index for class/label $\checkmark$
- Add AE functionality to wrapper $\checkmark$ (nearly)

- Finish regression with autoencoder wrapper

# 12/17/25
## Tasks:
- Table w/ 6 rows $\checkmark$
	- Frequent and rare MSE across regular/balanced/decoupled, with and without AE $\checkmark$
- Move all charts to documentation pages $\checkmark$
- `generate_ae_branch` should be `False` by default in all wrappers $\checkmark$
- Include documentation for how to identify representation layer index $\checkmark$
- Change default `representation_layer_index` from `-2` to `-3` for all functions $\checkmark$
- Document... document... document... $\checkmark$
	- Intro/explain $\checkmark$
	- Parameters $\checkmark$
	- Examples $\checkmark$
	- For AE, why do we provide it $\checkmark$
	- Why we pick -3 for default representation layer (especially for decoupled, linear/non-linear) $\checkmark$
	- Recommend AE, but not it is experimental and may not always work (or work well) $\checkmark$

- Rename `generate_ae_branch` to `generate_decoder_branch` $\checkmark$
- Move `generate_decoder_branch` from `util` to `util.backend` $\checkmark$
- **BUG:** balanced/decoupled fit should update output when generating AE branch $\checkmark$
- *Low priority:* AE branch should be removed/disabled after training $\checkmark$ (Actually fixed naturally when finalizing current balanced/decoupled fit)
## For later...
- *Low priority:* `imbal.classification.fit` and `imbal.regression.fit` (wrapper of standard model fit with option for AE generation)

## SEC-EC

| Method           | Epochs   | Time (s) | Frequent MSE | Rare MSE |
| ---------------- | -------- | -------- | ------------ | -------- |
| Regular w/o AE   | 1000     | 69.2     | 0.00519      | 0.17974  |
| Regular w/ AE    | 1000     | 92.2     | 0.00808      | 0.19683  |
| Balanced w/o AE  | 1000     | 100.1    | 0.01700      | 0.05807  |
| Balanced w/ AE   | 1000     | 143.6    | 0.02119      | 0.04558  |
| Decoupled w/o AE | 1000/500 | 117.2    | 0.00765      | 0.07888  |
| Decoupled w/ AE  | 1000/500 | 120.0    | 0.01244      | 0.15369  |

| Regular w/o AE                              |                                                 |
| ------------------------------------------- | ----------------------------------------------- |
| ![[fit-comparison--ae-False 1.png]]         | ![[tsne_visualization--ae-False 1.png]]         |
| Regular w/ AE                               |                                                 |
| ![[fit-comparison--ae-True 1.png]]          | ![[tsne_visualization--ae-True 1.png]]          |
| Balanced w/o AE                             |                                                 |
| ![[fit-comparison-balanced-ae-False 1.png]] | ![[tsne_visualization-balanced-ae-False 1.png]] |
| Balanced w/ AE                              |                                                 |
| ![[fit-comparison-balanced-ae-True.png]]    | ![[tsne_visualization-balanced-ae-True.png]]    |
| Decoupled w/o AE                            |                                                 |
| ![[Pasted image 20251219124107.png]]        | ![[tsne_visualization-decoupled-ae-False.png]]  |
| Decoupled w/ AE                             |                                                 |
| ![[fit-comparison-decoupled-ae-True.png]]   | ![[tsne_visualization-decoupled-ae-True.png]]   |
# 12/19/25
## Tasks:
- SEP-C classification should have its own page, similar to the comparison already done
	- See email from before
	- Classification decoupled/balanced fit should link to both
- Comparison of methods should be split between autoencoder on/off
	- For classification, on/off and image/tabular
- Is there a reason why for regression, decoupled w/ AE does worse than decoupled w/o
	- Are weights being preserved between extended model and original model?
	- Are layers being frozen properly?
	- Structure of extension (is it correct?)
- `output_label_index` $\rightarrow$ `imbalanced_output_label_index`
- Get rid of `multi_input` and `multi_output` from `decoupled_fit` and `balanced_fit`
	- For decoupled fit, a separate function needs to be created to handle the second stage balanced fit for AE
	- `DatasetWithBatching` should also not have these, but internally have a `DatasetWithBatching` that supports it for use during AE training
- For `util.generate_ae_branch`, explain branch generation algorithm $\checkmark$
	- Code example for decoder
	- Before and after of model structure (use `tf.keras.utils.plot_model` for documentation?)
	- Add developer parameter for `generate_decoder_branch` to output before and after plots
## For later...
- *Medium priority:* Refactoring decoupled/balanced fit from functions to wrapping around TF model object

## Notes:
- A chance to go back over/clean up/optimize `DatasetWithBatching` code would be nice
- Lots of refactoring will be necessary in the future