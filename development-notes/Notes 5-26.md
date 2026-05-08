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
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability