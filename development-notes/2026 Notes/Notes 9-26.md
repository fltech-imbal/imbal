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
	- Keep results of the "winner" of above 2 experiments, try it with hypersphere to see if there is any improvement
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