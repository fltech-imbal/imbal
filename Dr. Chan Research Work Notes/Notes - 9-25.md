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
	- J1?
- Convert "in code" formulas to $LaTeX$
	- Make sure to be using first/'definition" equation
	- Include intermediate values, with separate pages explaining them
- Override nondescriptive docstrings from TF
	- Mention "Overridden methods have been documented for additional clarity. for any undocumented methods, refer to the TF metric class"
- Add code examples for `model.compile()`
- Stratified Sampling by inheriting some TF class? (!!)
- "Hybrid" TF confusion matrix / custom computation where appropriate