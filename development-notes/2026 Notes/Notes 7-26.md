# 7/1/26
- FIRST, update QR code in presentation to link to documentation page, which will now act as the "homepage" $\checkmark$
	- On documentation home, add pdf download for "presentation of the `imbal` tool on may 6th 2026" $\checkmark$

---
# 7/8/26

- Swap SEP-C with SEP-EC sections in paper (to avoid forward reference)
	- Relabel tables and figures to match new order
- Using [Dynamic Time Warping](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw_path.html#tslearn.metrics.dtw_path) on proton intensity column of each time series:
	- Find all pairwise distances
	- Rank pairs and find N sized groups with closest pairs
	- Rank sum distance of groups
	- Remove pairs that contained a member of the formed group
	- Repeat
![[Pasted image 20260708171213.png]]
- Rerun all tests for report


---
# 7/10/26
- Send plots of cluster to Dr. Chan $\checkmark$
- Does peak intensity vs DTW generate better clusters? $\checkmark$
	- In my opinion, inconclusive
## Class
- Why did "maximize entropy" have faster training time and better AORE
	- Is the loss minimized quickly? or is the loss not going down at all?

---
# 7/13/26
## Research
- Try to do explicit representation learning by adding representation layer as separate output with a representation loss
	- Does AE still work? Maybe not
	- Is `imbal` alone flexible enough to allow for this?
	- Compatible with June paper loss functions? If not, what can we do?

```
rep_loss = RepresentationLoss()

def loss_fn(y_true, y_pred, weights=None):
	...

model = Model(input=Input(), output=[prediction_layer, representation_layer])

model.compile(
    loss=['mse', rep_loss]
)

model.train(x, [true_prediction, true_prediction])
```
## Class
- For semicircle with non-constant change, check coefficients and bias of regressor $\checkmark$
	- Plot them as a plane. In the same plot, plot the data points in the test set (with label as Z) $\checkmark$
- Use `UnitNormalization` as representation layer
	- Using anchor to encourage constant Euclidean distance ratio globally

---
# 7/14/26

## Research
#### High priority:
- See 7/14 email (2:30pm)
	- (!!!) Make sure to use same label epsilons to have same true values for CME and non CME intensity datasets $\checkmark$
	- Plot actual vs. predicted after post processing for models that predict delta and compare with models that predict intensity. Calculate MAE, AORE with same ln(10) rare threshold $\checkmark$
#### Medium Priority
- Try to do explicit representation learning by adding representation layer as separate output with a representation loss
	- Does AE still work? Maybe not
	- Is `imbal` alone flexible enough to allow for this?
	- Compatible with June paper loss functions? If not, what can we do?

```
rep_loss = RepresentationLoss()

def loss_fn(y_true, y_pred, weights=None):
	...

model = Model(input=Input(), output=[prediction_layer, representation_layer])

model.compile(
    loss=['mse', rep_loss]
)

model.train(x, [true_prediction, true_prediction])
```
#### Low priority:
- See 7/14 email (1:00pm)
## Class
- Make "non-constant speed" more extreme for non-constant speed examples
	- Try spacing all samples equally on the representation semicircle and line (common samples will take up the large majority of the representation space)
- Specify only semicircle shape, does resulting distribution match one of the examples? Something else? $\checkmark$
- Use `UnitNormalization` as representation layer
	- Using anchor (largest, or smallest) to encourage constant Euclidean distance ratio globally
		- Try freezing, and then fine tuning. Does one do better than the other?

---

# 7/17/26
## Research
#### High Priority
- Rerun delta prediction runs with correct epsilon (code should be correct now, but double check by confirming) $\checkmark$
	- Using same epsilons might be easier (pick 1e-5 or 1e-9) $\checkmark$
		- Used 1e-5 for labels and 1e-9 for features (Proton Intensity always uses label epsilon and `p_t` always uses feature epsilon)
	- If delta seems to be consistently performing worse, just use predicting ln(intensity) (w/ vs w/o CME) $\checkmark$
#### Medium Priority
- Try to do explicit representation learning by adding representation layer as separate output with a representation loss
	- Does AE still work? Maybe not
	- Is `imbal` alone flexible enough to allow for this?
	- Compatible with June paper loss functions? If not, what can we do?

```
rep_loss = RepresentationLoss()

def loss_fn(y_true, y_pred, weights=None):
	...

model = Model(input=Input(), output=[prediction_layer, representation_layer])

model.compile(
    loss=['mse', rep_loss]
)

model.train(x, [true_prediction, true_prediction])
```
#### Low priority:
- See 7/14 email (1:00pm)
## Class
- Make "non-constant speed" more extreme for non-constant speed examples
	- Try spacing all samples equally on the representation semicircle and line (common samples will take up the large majority of the representation space)
- Use `UnitNormalization` as representation layer
	- Try pairwise distance
	- Using anchor distance (largest, or smallest) to encourage constant Euclidean distance ratio globally
		- Try freezing, and then fine tuning. Does one do better than the other?
	- Don't worry about $y > 0$
- Look at 4 from paper again

----

# 7/20/26
## Research
#### High Priority
- Double check rare MAE for `sep_e_log_normalized` (3rd row, 2nd and 3rd columns) $\checkmark$
	- The plot that looked more accurate despite having a higher Rare MAE was leftover from prior runs with the "inconsistent" data. Regenerated plot with updated figures. $\checkmark$
- Also, plot true vs predicted with different colors for each time series, see if "arcs" in plots are of the same time series or different ones $\checkmark$
- Create callback deep copies before use in first decoupled stage / multi fit stages $\checkmark$
- **BUG FOUND**: custom Model object attributes such as best weights and decision threshold were not being saved properly. Working on a fix. 


```
model.fit(
	callbacks=[EarlyStopping(...)]
)
```
#### Medium Priority
- Try to do explicit representation learning by adding representation layer as separate output with a representation loss
	- Does AE still work? Maybe not
	- Is `imbal` alone flexible enough to allow for this?
	- Compatible with June paper loss functions? If not, what can we do?

```
rep_loss = RepresentationLoss()

def loss_fn(y_true, y_pred, weights=None):
	...

model = Model(input=Input(), output=[prediction_layer, representation_layer])

model.compile(
    loss=['mse', rep_loss]
)

model.train(x, [true_prediction, true_prediction])
```
#### Low priority:
- See 7/14 email (1:00pm)
## Class
- Make "non-constant speed" more extreme for non-constant speed examples
	- Try spacing all samples equally on the representation semicircle and line (common samples will take up the large majority of the representation space)
- Use `UnitNormalization` as representation layer $\checkmark$
	- Try pairwise distance $\checkmark$
	- Using anchor distance (largest, or smallest) to encourage constant Euclidean distance ratio globally $\checkmark$
		- Try freezing, and then fine tuning. Does one do better than the other? $\checkmark$
	- Don't worry about $y > 0$ $\checkmark$
- Look at 4 from paper again
- Why did "maximize entropy" have faster training time and better AORE
	- Is the loss minimized quickly? or is the loss not going down at all?
#### Class notes
- Using `UnitNormalization`
	- With freezing (1e-4 influence of regression loss during representation learning), ...
		- Pairwise is prone to "zig-zagging", since only pairs being enforced means you can "double back" without penalty for non-neighbors being on top of each other
		- Anchor distance performed quite well! Not able to generate 100% perfect predictions, but near perfect
	- With fine tuning (1e-4 influence of regression loss during representation learning, vice versa during regression learning), ...
		- Pairwise does quite well, thought representation loss increases during the second stage (aka non-constant pairwise distances are created in second stage)
		- Anchor distance did well!
	- With 1:1 joint, ...
		- Pairwise is still prone to error due to the lack of an ordered constraint on the representation space, but does okay
		- Anchor distance does very well
- On cases where 1e-4 influence is used, results had artifacts from learned representation when 1e-4 influence was excluded

---
# 7/23/26
## Research
- (!!!) Push changes to GitHub so Daniel can have fixed Model class (with fixed decoupled fit) $\checkmark$
- Update tutorials for decoupled fit. Code should be accurate now, but needs to be rerun since the output will now be different (not using `override_second_stage_fit_parameters`) $\checkmark$
	- Focus on image classification/regression $\checkmark$
	- Ask Daniel to update tabular classification classification/regression (decoupled fit only) $\checkmark$
	- We each have six to do (classification/regression, standard/validation/AE) $\checkmark$
- Check if t-SNE/LIME/SHAP/GradCam. Those may need to be updated too. **They do not use decoupled models** $\checkmark$
- **BUG FOUND**: custom Model object attributes such as best weights and decision threshold were not being saved properly. Fixed! $\checkmark$
## Class
- Runs with both pairwise and anchor

- Stage 1
	1. Last 2 layers have no activation (linear functions)
	2. Goal is to encourage a straight-line representation in the second to last layer (right before the output)
	3. Third to last layer is unrestricted
	4. Constant speed representation loss on second to last layer, with adjacent pairs and reference point pairs to encourage a straight line representation
		1. Potentially, encourage diversity of weights leading into second to last layer
		2. Still, third to last is an arbitrary shape
- Stage 2
	1. Throw away last 2 layers, replace it with a single output unit
	2. Freeze feature extractor up to the previous third to last layer (now second to last)
	3. "Compress" the last two layers, or "compress" multi-dimensional straight-line representation into a singular output

Other idea:
- `UnitNormalization`
	- Stage one
		- Some constant-speed representation loss (adjacent and anchor pairs)
	- Stage two
		- Fine tuning to allow for non-constant speeds using MSE loss as objective

Third idea:
- Same as above but with a non-linear regressor, freezing instead of fine-tuning in the second stage.

Baseline:
- No semicircle/`UnitNormalization`, constant speed representation loss (linear shape) with a linear regressor afterwards.

- Use SEP-C instead of ONP

Cauchy-Schwartz:
- $\|a\|\cdot\|b\| - \langle a, b\rangle$, maybe squared. Less computationally expensive than PCC and related things (few operations). No logarithm (faster than entropy). Parallelizable.
- Can also have a ratio loss (nudge near $1$)

---
# 7/24/26

## Research
#### Medium Priority
- Try to do explicit representation learning by adding representation layer as separate output with a representation loss
	- Does AE still work? Maybe not
	- Is `imbal` alone flexible enough to allow for this?
	- Compatible with June paper loss functions? If not, what can we do?

```
rep_loss = RepresentationLoss()

def loss_fn(y_true, y_pred, weights=None):
	...

model = Model(input=Input(), output=[prediction_layer, representation_layer])

model.compile(
    loss=['mse', rep_loss]
)

model.train(x, [true_prediction, true_prediction])
```
#### Low priority:
- See 7/14 email (1:00pm)
## Class
- Stage 1
	1. Last 2 layers have no activation (linear functions)
	2. Goal is to encourage a straight-line representation in the second to last layer (right before the output)
	3. Third to last layer is unrestricted
	4. Constant speed representation loss on second to last layer, with adjacent pairs and reference point pairs to encourage a straight line representation
		1. Potentially, encourage diversity of weights leading into second to last layer
		2. Still, third to last is an arbitrary shape
- Stage 2
	1. Throw away last 2 layers, replace it with a single output unit
	2. Freeze feature extractor up to the previous third to last layer (now second to last)
	3. "Compress" the last two layers, or "compress" multi-dimensional straight-line representation into a singular output

Other idea: $\checkmark$
- `UnitNormalization` $\checkmark$
	- Stage one $\checkmark$
		- Some constant-speed representation loss (adjacent and anchor pairs) $\checkmark$
	- Stage two $\checkmark$
		- Fine tuning to allow for non-constant speeds using MSE loss as objective $\checkmark$

Third idea:
- Same as above but with a non-linear regressor, freezing instead of fine-tuning in the second stage.

Baseline: $\checkmark$
- No semicircle/`UnitNormalization`, constant speed representation loss (linear shape) with a linear regressor afterwards. $\checkmark$

- Use SEP-C instead of ONP

Cauchy-Schwartz: $\checkmark$
- $\|a\|\cdot\|b\| - \langle a, b\rangle$, maybe squared. Less computationally expensive than PCC and related things (few operations). No logarithm (faster than entropy). Parallelizable. $\checkmark$
- Can also have a ratio loss (nudge near $1$)

---
## Research
- See paper notes
## Class
- SEP-EC and SEP-C
- Loss ratio for unit hypersphere $\checkmark$
- Stage 1
	1. Last 2 layers have no activation (linear functions)
	2. Goal is to encourage a straight-line representation in the second to last layer (right before the output)
	3. Third to last layer is unrestricted
	4. Constant speed representation loss on second to last layer, with adjacent pairs and reference point pairs to encourage a straight line representation
		1. Potentially, encourage diversity of weights leading into second to last layer
		2. Still, third to last is an arbitrary shape
- Stage 2
	1. Throw away last 2 layers, replace it with a single output unit
	2. Freeze feature extractor up to the previous third to last layer (now second to last)
	3. "Compress" the last two layers, or "compress" multi-dimensional straight-line representation into a singular output
- **Why?** Second to last layer learns many redundant features, so we throw it out and replace it with a single output unit.

**Dimensions**
	1. Unit hypersphere yes/no $\checkmark$
		- If not, should have some ratio loss (close to 1) $\checkmark$
	2. Representation loss function used $\checkmark$
		1. Distance PCC $\checkmark$
		2. Cauchy-Schwartz $\checkmark$
		3. Entropy $\checkmark$
	3. Freezing, tuning, or joint? $\checkmark$
	4. Also, throw away second to last and freeze third to last $\checkmark$
		1. Add in Augmented PCC $\checkmark$
	5. SEP-EC / SEP-C $\checkmark$

---
# 7/29/26

## Research
- Focus on section 3 next for paper $\checkmark$
	- Add table with overview in section 3 $\checkmark$
	- Add an example for SHAP in explanation of predictions
	- Add number of events in training set vs test set (not instances)
		- Also add common/rare percentages for training and test sets
		- extend discussion from NASA report
- Section 4
	- Plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
		- Plot series that are highly over/underestimating (based on scatter plot) (2)
		- Also plot two that are quite accurate

---
# 7/31/26

## Research
- Change `\cite` to `\citeyear` where appropriate in paper $\checkmark$
- Remove subjective wording from related work $\checkmark$
- Instead of "work", referring to a paper/study, we can be more direct: $\checkmark$
	- Richardson et al. `\citeyear{richardson2018}` used CME data to derive $\checkmark$
	- In general, eliminate the word "work" wherever possible $\checkmark$
- Add an example for SHAP in explanation of predictions
- Add number of events in training set vs test set (not instances) $\checkmark$
	- Also add common/rare percentages for training and test sets $\checkmark$
	- extend discussion from NASA report, and remove "overexplained" parts of the NASA report $\checkmark$
- Section 4
	- Plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
		- Plot series that are highly over/underestimating (based on scatter plot) (2)
		- Also plot two that are quite accurate

