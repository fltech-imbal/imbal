# 9/1/26

## Thesis
Axes of exploration:
- Latent space unit hypersphere: y/n
- Loss function, constant vs non-constant distance ratio
- Use more than 2 (6) dimensions in representation space
- (!!!) Inclusion/excluison of weighting samples inversely with respect to the distance in the label space 
  (futher labels need not have similar representations)
- joint, freeze, fine-tuning (fine-tuning will probably work best)

We expect non-unit space and constant ratio to work well, and unit hypersphere and non-constant ratio to work well

**Order of tasks:**
- **Round 1:** Constant + linear (no hypersphere) w/ 6+ dimensions in the representation space (using 32 for SEP-E) w/ fine tuning $\checkmark$
	- **Round 2:** Then, constant + no hypersphere + "decorrelation among features" loss (we expect this to help) (with joint as well if first experiments shows it to be any better) $\checkmark$
	- Keep results of the "winner" of above 2 experiments, try it with hypersphere to see if there is any improvement $\checkmark$
- **Round 3:** Later, unit hypersphere with non-constant speed, cosine similarity-based representation loss
	- We expect this to be better than winner of first 2 experiments w/ hypersphere
- **Round 4:** Hypersphere, constant distance ratio, *non-linear regressor*, trying best from rounds 2 and 3 to see which does better... or, is round 2 sufficient when using a non-linear regressor? (multiple layer regressor with activation functions)
- Might be worth trying to weight samples by `t+6 - t` in the future

**Other thoughts...**
- Something for gradient conflicts
	- Think not only of direction, but length
	- More common samples means larger gradient vector
	- Considering magnitude, scaling such that the magnitude of the vectors are of the same length
## Paper
- SHAP will be used for explanations in section `4 SEP Forecasting tasks
- Crop time series plots further to highlight areas of interested (right before and right after rising edge)
	- Assume each picture will be 6.5 inches, across the page. Is is visible?
	- Include dates in each tick mark, ticks can be 6-12 hours apart
	- y label can simply be `ln(flux)`Loss
## NASA

**Using toy dataset...**

|         | MSE Loss     | Representation Loss | Fine Tuning/Joint | MAE | Rare MAE | AORE |
| ------- | ------------ | ------------------- | ----------------- | --- | -------- | ---- |
| Round 1 | $\checkmark$ | $\times$            | N/A               |     |          |      |
| Round 2 | $\times$     | $\checkmark$        | N/A               | N/A | N/A      | N/A  |
| Round 3 | $\checkmark$ | $\checkmark$        | joint             |     |          |      |
| Round 4 | $\checkmark$ | $\checkmark$        | fine tuning       |     |          |      |

- Representation learning, add `representation_lambda` parameter for `Model.compile`
	- For reconstruction branch, default behavior is determining lambda ourselves. If they specify a lambda, use theirs instead.
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate
- `validation_split` overrides previous behavior in some cases
	- Can this be split into `k_folds` and `validation_split`
---
# 9/8/26

## Thesis
- **Round 1**
	- Entropy, Cauchy-Schwartz, Distance PCC, Variance $\checkmark$
		- Ignore joint, ignore regular, always $\alpha=1$ $\checkmark$
	- Generate TSNE and true vs. predicted $\checkmark$
	- **Round 1b** - Pick top 2 performing from previous, vary joint/tuning, vary $\alpha$ $\checkmark$
		- **Rerun all round 1 with $\alpha$ varying 0.1, 0.3, 0.5, 0.7 $\checkmark$**
		- Original distribution for first stage, only vary alphas in second stage $\checkmark$
- **Round 2**
	- Pick top performing for each loss from round 1b, add decorrelation loss
	- **JUST** enable decorrelation, see if it performs better 
	- Generate TSNE and true vs. predicted 
	- **Round 2b** best performing 2 from round 2, enable/disable hypersphere, vary $\alpha$
		- Generate TSNE and true vs. predicted
		- **No need for decorrelation** $\times$
- **Round 3**
	- Hypersphere with cosine loss, FT/joint, vary $\alpha$
	- Generate TSNE and true vs. predicted
	- **With time, try enable weighting based on label distance**
		- Will have to be calculated "on-the-fly", in the loss
		- Keep RI for MSE, two sets of weights
- **Round 4**
	- Pick best two from round 2, use a non-linear regressor and constant speed (hence going back to round 2 loss functions)
	- Generate TSNE and true vs. predicted
- **Round 5**
	- Best 2-3 from Rounds 3-4
	- Add learnable $R$ parameter (described in 9/7/26)

**SEP-E and SEP-C**

Axes of exploration:
- Latent space unit hypersphere: y/n
- Loss function, constant vs non-constant distance ratio
- Use more than 2 (6) dimensions in representation space
- (!!!) Inclusion/excluison of weighting samples inversely with respect to the distance in the label space 
  (futher labels need not have similar representations)
- joint, freeze, fine-tuning (fine-tuning will probably work best)

We expect non-unit space and constant ratio to work well, and unit hypersphere and non-constant ratio to work well

**Other thoughts...**
- Might be worth trying to weight samples by `t+6 - t` in the future
- Something for gradient conflicts
	- Think not only of direction, but length
	- More common samples means larger gradient vector
	- Considering magnitude, scaling such that the magnitude of the vectors are of the same length
- 9/7/26 email
	- Learnable $R$ value using a model with no input, one output, loss update is $(R-r)^2$, where $R$ is the learned ratio and $r$ is the current ratio
		- Use sigmoid scaled from $[\frac{1}{a}, a]$ as activation for output logit
	- Can be added to primary representation loss functions
	- Using custom Keras layer
		- No input, no weights, just bias, which is passed to ratio loss
	- Potentially using one model, one input, one "pseudo-input", concatenated into a single output. Then use custom loss to separate back out and compute separate components
		- TensorFlow Concatenate layer
## Paper
- SHAP will be used for explanations in section `4 SEP Forecasting tasks
## NASA

**Using toy dataset...**

|         | MSE Loss     | Representation Loss | Fine Tuning/Joint | MAE | Rare MAE | AORE |
| ------- | ------------ | ------------------- | ----------------- | --- | -------- | ---- |
| Round 1 | $\checkmark$ | $\times$            | N/A               |     |          |      |
| Round 2 | $\times$     | $\checkmark$        | N/A               | N/A | N/A      | N/A  |
| Round 3 | $\checkmark$ | $\checkmark$        | joint             |     |          |      |
| Round 4 | $\checkmark$ | $\checkmark$        | fine tuning       |     |          |      |
- For shuffling
	- When shuffle is `False`, our shuffling and TF shuffling is off
	- When shuffle is `True`, our shuffling is ON, but *TF remains OFF*
- Is `decoupled_fit` compatible with representation loss? Are there issues at the moment?
- Representation learning, add `representation_lambda` parameter for `Model.compile`
	- For reconstruction branch, default behavior is determining lambda ourselves. If they specify a lambda, use theirs instead.
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate
- `validation_split` overrides previous behavior in some cases
	- Can this be split into `k_folds` and `validation_split`
- Perhaps add $(\|a\|-\|b\|)^2$ to cauchy-schwartz loss (makes distance ratio 1)  
- Alternative: $\|a-\frac{1}{\alpha} b\|$ as loss function ($\alpha$) sets the distance ratio (for thesis)  
- Switch to SEP-C **tutorial** dataset and rerun experiments in 4 tables above (small real dataset)
---

$$
\begin{array}{l}
(\|rep\|-\beta\|lab\|)^2=0
\\\\
\|rep\|-\beta\|lab\|=0
\\\\
-\beta\|lab\|=-\|rep\|
\\\\
\beta = \frac{\|rep\|}{\|lab\|}
\\\\\\\\

\|rep - \beta\cdot label\|=0
\\\\
rep-\beta\cdot label = \mathbf{0}
\\\\
-\beta \cdot label = -rep

\end{array}
$$