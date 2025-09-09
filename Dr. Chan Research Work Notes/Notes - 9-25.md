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

# 9/5/25

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