# 5/5/26
## Tasks
#### High priority
- Update installation guide with note to follow what TensorFlow's implementation says
- [Keras GradCam](https://keras.io/examples/vision/grad_cam/) how easily can we wrap this and implement in imbal?
	- Make sure it works for binary classification with single output unit (convert to 2 class)
	- Make sure to mention code is taken from that link in documentation
	- Can we make it work for regression
- Explain what "weight candidate index" means in output for documentation, help the user interpret what that means (refer to section where alphas are specified)
- Documentation for multi-weight fit should include that best weights/class weights and index are saved in the model object in particular fields.
- Incorporate finding decision thresholds in validation set (sweep thresholds across validation set)
	- Sweeping class weights
	- For each model, sweep decision threshold and find best thresholds for that class weight
	- For `imbal`, just one "fold" with early stopping. Sweeping is based on validation set
	- Separate "validation metric" passed during compile time
		- For single weight candidate fit, default to `val_loss`
		- For multi weight, default to first metric in `metrics`, or `F1` if none
		- Also allow for min or max parameter ([more here](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping))
	- Verbosity 1, print ex. `class weights: [0.1, 0.9] ... Best decision treshold: 0.3`
	- Verbosity 2, print ex. `class weights: [0.1, 0.9], testing DT 0.1 using metric [metric name] ... DT 0.2 ... DT 0.3 ... Best decision threshold: 0.6`
	- See email from May 7th for more details
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

---
# 5/8/26

#### High priority
- Incorporate finding decision thresholds in validation set (sweep thresholds across validation set)
	- Sweeping class weights $\checkmark$
	- For each model, sweep decision threshold and find best thresholds for that class weight $\checkmark$
	- For `imbal`, just one "fold" with early stopping. Sweeping is based on validation set $\checkmark$
		- What to do if there is no validation set? currently just using training set, but this is probably not the best idea. ${\huge ???}$
	- Separate "validation metric" passed during compile time $\checkmark$
		- For single weight candidate fit, default to `val_loss` $\checkmark$
		- For multi weight, default to first metric in `metrics`, or `F1` if none $\checkmark$
		- Also allow for min or max parameter ([more here](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)) $\checkmark$
	- Verbosity 1, print ex. `class weights: [0.1, 0.9] ... Best decision threshold: 0.3` $\checkmark$
	- Verbosity 2, print ex. `class weights: [0.1, 0.9], testing DT 0.1 using metric [metric name] ... DT 0.2 ... DT 0.3 ... Best decision threshold: 0.6` $\checkmark$
	- See email from May 7th for more details $\checkmark$
- Fixed some inconsistencies found in the documentation $\checkmark$
	- Fixed some images not showing in the documentation $\checkmark$
- documentation home page: t-SNE Model Explanation -> t-SNE Visualization of Latent/Representation Space $\checkmark$
- Update installation guide with note to follow what TensorFlow's implementation says $\checkmark$
- Explain what "weight candidate index" means in output for documentation, help the user interpret what that means (refer to section where alphas are specified)
- Documentation for multi-weight fit should include that best weights/class weights and index are saved in the model object in particular fields.
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

---
# 5/11/26

## Notes
- Validation with threshold sweeping currently in a working state, but several questions)
- For multi-weight regression, are we just taking lowest validation loss?
	- **Use first metric passed by user as well, default to MAE** $\checkmark$
- What to do if there is no validation set for multi-fit? currently just using training set, but this is probably not the best idea.
	- **Multiple weight candidates without any validation data is a fatal error**  $\checkmark$
- Should we allow a minimize/maximize parameter for the multi-fit validation metrics? Currently using the auto-decision used by Keras `EarlyStopping`
	- **Add `self._direction` to our metrics, to make them work with "auto" for `EarlyStopping` and threshold sweep**  $\checkmark$
- Should we allow for user to specify thresholds? (hold off until later)
- Documentation still needed $\$
- Double check: best class/sample weights saved alongside best metric threshold $\checkmark$
## Tasks
#### High priority
- Explain what "weight candidate index" means in output for documentation, help the user interpret what that means (refer to section where alphas are specified)
- Documentation for multi-weight fit should include that best weights/class weights and index are saved in the model object in particular fields. 
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

---
# 5/13/26

## Notes
- I made muti-fit throw an fatal error if it is called without validation data being specified. This now makes some of our tutorials "supplements" throw a fatal error (ex. balanced fit with multiple weight candidates) Should we get rid of those supplements? $\checkmark$

Ensure table below $\checkmark$

| multiple weight candidates | No validation set                                                            | With validation set                                                            |
| -------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| classification             | metric (ex. F1) against training set, vary decision threshold after training | metric (ex. F1) against validaiton set, vary decision threshold after training |
| regression                 | metric (ex. MAE) against training set                                        | metric (ex. MAE) against validaiton set                                        |
## Tasks
#### High priority
- Documentation still needed $\checkmark$
- What to do if there is no validation set for multi-fit? currently just using training set, but this is probably not the best idea. $\checkmark$
	- **Same process on training set** $\checkmark$
- Multiple weight candidates with decision thresholds supplements. (make sure to specify we use metric to determine best, not loss) $\checkmark$
- Code example for multiple class weights (with threshold varying in classification) $\checkmark$

#### Low Priority:
- Swap image classification and regression on tutorials $\checkmark$
- Update tutorials to reflect metric-based validation $\times$
- Allow user to specify thresholds in fits using validation
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

---
# 5/18/26

## Notes
- For using balanced sample weights for metrics
	- if we require the user to pass the weights to balanced fit, there is no need to have a custom metric (any existing metric can already be weighted)
	- If we require the user to pass the weights to the metric, then our approach is only compatible with that metric
- Questions about survey paper
	- Grading? By a rubric? More about having done it?
	- Revisions?

## Tasks
#### High Priority
- Update tutorials to reflect metric-based validation
- Store the class/sample weights and decision threshold (if classification) in the model, so that it can be retrieved with a "get" method.  The decision threshold is needed for testing. $\checkmark$
- Use `weighted_metrics`, then `metrics`, then default to F1 or weighted MAE. $\checkmark$
#### Low Priority:
- Allow user to specify thresholds in fits using validation
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

---
# 5/21/26
## Notes
- Classification/regression
- weighted metric/unweighted metric
- validation data/only training data
- single weight candidate/multiple weight candidates

1. Keras: When a user specifies a weighted metric, sample weights should be specified; otherwise, the weighted metric is not meaningful
2. `Imbal`: if there is only one set of sample weights specified, no need to eval multiple candidates with a consistent metric
	1. Except for classification, we must evaluate different decision thresholds
3. `Imbal`: if there are multiple sample weight candidates (from user or generated by us), we need to eval them fairly with the same/consistent metric that emphasizes rare instances more
4. Keras: the user can specify weighted metrics, metrics, or nothing
5. `Imbal`: if the user does not specify any metric, the consistent metric is
	1. Classification: F1 (implicitly weighted at the class level, no sample/class weights needed)
	2. Regression: balanced MAE, balanced sample weights needed
6. Keras: only one set of sample weights can be specified, it is used in loss and weighted metrics.


| User Specified Metric; on validation (if specified), otherwise on training set | User specifies one set of training sample weights for loss                                                                         | User specifies/`imbal` generates multiple candidates for training sample weights               | User specifies/`imbal` generates multiple candidates for training sample weights |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
|                                                                                |                                                                                                                                    | Classification                                                                                 | Regression                                                                       |
| Weighted metric                                                                | Regression: N/A; Classification: use sample weights for weighted metric on decision thresholds                                     | Use class-balanced weights for weighted metric on different candidates and decision thresholds | Use balanced sample weights for weighted metric on different candidates          |
| Unweighted metric                                                              | Regression: N/A; Classification: sample weights not applicable for unweighted metric, use unweighted metric on decision thresholds | Use unweighted metric on different candidates and decision thresholds                          | Use unweighted metric on different candidates                                    |
| None                                                                           | Regression: N/A; Classification: sample weights not applicable, use F1 on decision thresholds                                      | Use F1 on different candidates and decision thresholds                                         | Use balanced sample weights for balanced MAE on different candidates             |
- Either passing densities or a new parameter to obtain balanced sample weights for regression (rightmost column) $\checkmark$
- Classification:
	- Class weights for candidate eval (default: class balanced) $\checkmark$
	- Sample weights for candidate eval (overrides class weights) $\checkmark$
- Regression:
	- Sample weights for candidate eval (have to specify for multiple candidates) $\checkmark$
## Tasks
#### High Priority
- Update tutorials to reflect metric-based validation
- Store the class/sample weights and decision threshold (if classification) in the model, so that it can be retrieved with a "get" method.  The decision threshold is needed for testing. $\checkmark$
- Use `weighted_metrics`, then `metrics`, then default to F1 or weighted MAE. $\checkmark$
#### Low Priority:
- Allow user to specify thresholds in fits using validation
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

---
## 5/22/26

## Tasks
#### High Priority
- Update tutorials to reflect metric-based validation

Weighted metrics test:
- Classification
	- Reg $\checkmark$
	- Reg + AE $\checkmark$
	- Reg + val $\checkmark$
	- Bal $\checkmark$ $\checkmark$
	- Bal + AE $\checkmark$ $\checkmark$
	- Bal + val $\checkmark$ $\checkmark$
	- cRT $\checkmark$ $\checkmark$
	- cRT + AE $\checkmark$ $\checkmark$
	- cRT + val $\checkmark$ $\checkmark$
- Regression
	- Reg $\checkmark$
	- Reg + AE $\checkmark$
	- Reg + val $\checkmark$
	- Bal $\checkmark$
	- Bal + AE $\checkmark$ $\checkmark$
	- Bal + val $\checkmark$ $\checkmark$
	- rRT $\checkmark$
	- rRT + AE $\checkmark$ $\checkmark$
	- rRT + val $\checkmark$ $\checkmark$

'Final training with combined training and validation set' test:
- Classification
	- Reg + val $\checkmark$
	- Bal + val $\checkmark$ $\checkmark$
	- cRT + val $\checkmark$ $\checkmark$
- Regression
	- Reg + val $\checkmark$
	- Bal + val $\checkmark$ $\checkmark$
	- rRT + val $\checkmark$ $\checkmark$
#### High Priority:
- Updating Model documentation to reflect changes for `weighted_metrics` (in particular, with multiple candidates)
	- Combined training + validation
		- Trained for best epoch number from first stage, using best weights found across all candidates
- Update tutorials to reflect changes to fit functions (`weighted_metrics`, final fit train + validation?)
- See if there were low priority tasks that have been looked over from previous meetings
	- Send an email to Dr. Chan with the compiled list
#### Low Priority:
- Allow user to specify thresholds in fits using validation
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

