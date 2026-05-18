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
- Documentation still needed
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
- I made muti-fit throw an fatal error if it is called without validation data being specified. This now makes some of our tutorials "supplements" throw a fatal error (ex. balanced fit with multiple weight candidates) Should we get rid of those supplements?

| multiple weight candidates | No validation set                                                            | With validation set                                                            |
| -------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| classification             | metric (ex. F1) against training set, vary decision threshold after training | metric (ex. F1) against validaiton set, vary decision threshold after training |
| regression                 | metric (ex. MAE) against training set                                        | metric (ex. MAE) against validaiton set                                        |
## Tasks
#### High priority
- Documentation still needed
- What to do if there is no validation set for multi-fit? currently just using training set, but this is probably not the best idea. $\checkmark$
	- **Same process on training set** $\checkmark$
- Multiple weight candidates with decision thresholds supplements. (make sure to specify we use metric to determine best, not loss)
- Code example for multiple class weights (with threshold varying in classification)

#### Low Priority:
- Swap image classification and regression on tutorials $\checkmark$
- Update tutorials to reflect metric-based validation
- Allow user to specify thresholds in fits using validation
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability