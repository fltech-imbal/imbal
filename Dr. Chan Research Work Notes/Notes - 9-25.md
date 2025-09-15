# 9/3/25

## Prep:
- Minimize memory usage for metric import dictionary by updating TF dictionary directly $\checkmark$ 
- Fix: Critical Success Index was not updated to match new, more "TF-like" implementation $\checkmark$
- Metric optimization achieved using the [keras Callback class](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) $\checkmark$
	- At least in the small MNIST example, the benefits from optimization aren't entirely clear, though it definitely saves some memory
	- Note: I now better understand the conflicts that were arising when previously trying to implement the `ConfusionMatrix` class. May take another stab at a general `ConfusionMatrix` class in the future?
- `BoundedAUC`, decision threshold high to low, simply stop FPR/TPR calculation when max FPR is reached
> Since thresholds alone do necessarily map linearly to FPR or TPR values (a threshold value of 0.5 does not generate an FPR/x value of 0.5), we are left with two options: 1) Calculate all FPRs up until the threshold we are looking for, or 2) Calculate all FPRs, then trim the resulting list to contain only values within the specified bounds.
>
> If we are computing FPR in serial, the first option is the obvious choice. However, the built in TensorFlow implementation for AUC calculates FPRs for all thresholds in parallel, meaning it is likely more time efficient to calculate all FPRs, then trim the ones we wish not to keep (despite this approach using more memory)

* Documentation for metric classes using Sphinx $\checkmark$
* Refactored metrics to use `keras.src.metrics.metric_utils` $\checkmark$

## Meeting Notes:
- Stepping away from "single confusion matrix metrics"
	- We could maybe do it more seamlessly, but accuracy could suffer. Accuracy is paramount, followed by seamlessness.
## Tasks:
-  "this class can be passed as a metric, along with any of the following string type aliases: "in docs replace with "class instance OR string"
- J statistic and Youden's index are subclasses that directly inherit every function of TSS
	- **Show why formulas are equal, and point to TSS page**
- Convert "in code" formulas to $LaTeX$
	- Make sure to be using first/'definition" equation
	- Include intermediate values, with separate pages explaining them
- Override nondescriptive docstrings from TF
	- Mention "Overridden methods have been documented for additional clarity. for any undocumented methods, refer to the TF metric class"
- Add code examples for `model.compile()`
- **Stratified Sampling by inheriting some TF class? (!!)**
- "Hybrid" TF confusion matrix / custom computation where appropriate

# 9/8/25

## Prep
- "this class can be passed as a metric, along with any of the following string type aliases: "in docs replace with "class instance OR string" $\checkmark$
- J statistic and Youden's index are subclasses that directly inherit every function of TSS $\checkmark$
- Convert "in code" formulas to $LaTeX$ $\checkmark$
	- Include intermediate values, with separate pages explaining them $\checkmark$
- Override nondescriptive docstrings from TF $\checkmark$
	- Mention "Overridden methods have been documented for additional clarity. for any undocumented methods, refer to the TF metric class" $\checkmark$
- Add code examples for `model.compile()` $\checkmark$
-  "Hybrid" TF confusion matrix / custom computation where appropriate $\checkmark$

**Notes from 8/25/25**:
![[notes-screenshot-8-25-25.png|600]]

- Stratified sampling $\checkmark$
	- Implemented as extension of `tf.keras.utils.PyDataset`, which allows for multiple batches to be run in parallel if
		- Without stratified class sampling, TPR on 10% sparse data was 75-80%. With stratified class sampling, TPR jumped to 90-95%
		- Other metrics suffer, but that is more likely due to the fact that the model I am testing with is very simple ($28\times28\rightarrow32\rightarrow1$ feed-forward network).
	- Implemented even batching by class using stride (something I am quite happy with)
	- Two options:
		- `sample_weights` - Specifies the weights of each sample in a class. By default, each sample has equal weighting.
		- `class_weights` - Specifies the portion of the total weight of the data each class should take up. For example `{ 0 : 1.0, 1: 1.0 }`, means that the final weights for the data will be one part class 0, and one part class 1. In other words, 50/50, regardless of the number of samples in each class.
	- May have some areas that can still be optimized, but seems to work well for large batch sizes (smaller batch sizes take longer because of more array operations)
	- **ONE ISSUE:** Because of how the `tf.keras.util.PyDataset` class is handled by `model.fit` one single instance cannot handle performing the train/test split, it would have to be handled by the user or we can provide a means to do so as a separate function

## Tasks:
- Copy link to TF Metric class in all Metric classes
- Confirm functionality on much smaller (countable) data and batch size
	- Don't use MNIST, handpicked dataset
	- 2 instances per class, start with 2 classes and work up to 10 classes
	- Via `print` statements, make sure batch contents are as expected (distributed as expected, randomized each epoch)
- Randomize batch "membership" every epoch using `on_epoch_end()`
- Change default behavior to be each *CLASS* is weighted evenly, not data point

# 9/10/25

## Prep:
- Copy link to TF Metric class in all Metric classes $\checkmark$
- Randomize batch "membership" every epoch using `on_epoch_end()` $\checkmark$
- Change default behavior to be each *CLASS* is weighted evenly, not data point $\checkmark$
- Confirm functionality on much smaller (countable) data and batch size $\checkmark$

## Task:
- Does TF randomize batches per epoch with normal data passing? (x, y, weights)
- Remove all class weight related code from `StratifiedSampling`. It only "knows" instance/sample weights
	- Convert dictionary usage to list usage
	- Create a separate "class weight to instance weight converter" 
- Stratified sampling sub-package consists of `StratifiedBatcher` (class, `PyDataset`) and train/test split (wrapper for scikit-learn)
	- ***ENSURE*** `scikit-learn` handles data and labels the same as TensorFlow
- **Priority:**
	1. batching for classes
	2. train/test split for classes
	3. batching for continuous (regression)
	4. train/test split for continuous
	...
	5. reweighting

# 9/12/25
## Prep
- Does TF randomize batches per epoch with normal data passing? (x, y, weights) $\checkmark$
![[Pasted image 20250911214214.png]]
**Note:** Default: `shuffle=True`
- Remove all class weight related code from `StratifiedSampling` $\checkmark$
	- Convert dictionary usage to list usage $\checkmark$
- **FROM EMAIL: Implement 6 larger demo cases** $\checkmark$
- Ensure `scikit-learn` handles data and labels the same as TensorFlow
![[Pasted image 20250911230001.png]]
`train_test_split` expects data to either be a 1D vector, or a column vector of data points. A column vector tends to be what TF prefers, so `scikit-learn` is compatible in this case. Tests also prove this to be the case.
- **Priority:**
	1. batching for classes $\checkmark$
	2. train/test split for classes $\checkmark$
	3. batching for continuous (regression) $\checkmark$
	4. train/test split for continuous $\checkmark$
- Had spare time, looked at reweighting
## Tasks:
- Double check weights are preserved across all types of stratified sampling
	- For multiple copies of a sample, the sum of all copies is the same as before splitting
- Have a test procedure to compare the before and after weights for the same samples
	- Can be commented out later, but good for testing on large data amounts
	- Basically a unit test
- Remove weight normalization
- Use `PyDataset` to convert currently implemented stratify functions as a class
	- Classification and regression can be handled by same class (similar to batching) if convenient (classification or regression is a parameter)
# 9/15/25
## Prep:
- Double check weights are preserved across all types of stratified sampling $\checkmark$
	- Default weight per sample when no weights are specified is $\frac{1}{\text{num\_data}}$
	- For multiple copies of a sample, the sum of all copies is the same as before splitting $\checkmark$
- Have a test procedure to compare the before and after weights for the same samples $\checkmark$
	- Tests for batching and test/train split
- Remove weight normalization $\checkmark$
 - Use `PyDataset` to convert currently implemented stratify functions as a class
	- Classification and regression can be handled by same class (similar to batching) if convenient (classification or regression is a parameter)
- For train/test split, used similar interface to TF/Keras
	- The interface for TF/Keras is similar to sklearn, but with some notable differences
		- TF/Keras takes in a tuple or arbitrary TF `Dataset`, but we want data, labels, weights to be separate fields so we know what to stratify by
		- `left_size` and `right_size` instead of `test_size` and `train_size` (I feel like convention we already had is more clear)
	- Found: `sklearn` does not shuffle *and* stratify. Not sure why, but I have compensated for that.

![[Pasted image 20250915074038.png]]