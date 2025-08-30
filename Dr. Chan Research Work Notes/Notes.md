# 8/18/25

Tasks:
- Look at metrics that were not discussed, with reasoning:
	- Critical Success Index
	- Gilbert Skill Score
- Simple TensorFlow implementation (MNIST fashion)
- Linear algebra TensorFlow library, familiarize with "GPU friendly" parallelization
- Implement [[F1]] (assume I have predictions and labels)
	- Build 2 versions, one with loops vs parallelization

# 8/20/25
## Prep

### Interpretation of [[Critical Success Index]]:
> The ratio between the number of **correctly predicted positive events** and the combined number of **positive predictions and actual positive data**. In essence, the CSI is the *percentage of the data set that was accurately predicted as positive, ignoring true negatives, which is likely a large majority of the correct predictions*.

### Interpretation of the [[Gilbert Skill Score]]:
>Like the [[Critical Success Index]], but adjusted for the number of [[Expected True Positive]] predictions. Therefore, a Gilbert Skill Score of 0 means the model receives the same [[Critical Success Index]] as an unskilled model, and a Gilbert Skill Score of 1 means the model is "100% skilled".

### ROC Curve:
[[True Positive Rate]] vs. [[False Positive Rate]]
- Perfect corner is (0, 1), therefore, indicative of a model that guesses negatives with 100% certainty (hence, 0 [[False Positive Rate]]), and positives with 100% certainty (hence, 1 [[True Positive Rate]]).
- Plotting all decision thresholds, a, from 0 to 1, results in a curve that ideally should like like a logarithm curve, which would appear flat on the right (for lower values of a), the drop steeply only at the very left of the curve (for higher values of a). This would indicate a model that is highly confident in its predictions.

### Derivation of product rule
For events that are independent of each other, their probability represents a (sometimes simplified) fraction where the numerator is the total number of possible outcomes that meet the event criteria, and the denominator is the total number of possible outcomes. Therefore, when considering the odds that two events occur together, the number of possible outcomes for this "combined event" becomes the product of the number of possible outcomes of each individual events (since for each outcome of the first event, every one of the second event's possible outcomes can occur). The total number of outcomes that meet the event criteria also is the product of the individual event outcomes (for a similar reason as with the denominators).
Therefore, the probability of two independent events occurring is equal to the product of the numerators divided by the product of the denominators, or simply just the product of the two probabilities.

## Tasks
- Expand implementation to include all skill scores
	- Use TensorFlow library
	- AUROC... does it exist in a library? F1?
		- The rest of the scores are more from physics than ML
	- Switch from randomly generating TP, FP, TN, and FN to generating P/N pairs.
	- (!!!) Calculate confusion matrix, 3 approaches, serial, matching, and arithmetic
		- Generate a table, comparing the time
		- 10,000 array size, 10 times, calculate average
- ROC
	- Think about how to do AUROC in parallel
	- how to count in parallel, at least 2 ways

# 8/22/25

## Prep

> For the 2x2 confusion matrix, you only need to find 3 numbers from the 2 input arrays, the 4th doesn't need to look at the 2 input arrays, why?

Because by definition, S = TP + FP + FN + TN, therefore, if we know S and 3 other values in the confusion matrix, we can find the fourth confusion matrix value.

Both F1 and AUROC exist in TensorFlow.

Ways to sum / count up elements in list:
- Recursion: Split array into two subarrays, count the contents of each subarray recursively and in parallel, then add result from each subarray and return.
- Pairwise sums: In parallel, add all pairs of numbers in the list (Ex: \[1, 2, 3, 4, 5, 6, 7, 8, 9, 10\] $\rightarrow$ \[3, 7, 11, 15, 19\]). Then, continue this process until there is only one element left, and return that element.
## Meeting Notes
## Tasks
- Use Pos - TP = FP in arithmetic approach for parallel ($\checkmark$)
- Try a data split of 2% TP, 1% FN, 2% FP, 95% TN ($\checkmark$)
- Use TensorFlow AUROC and compare (maybe with table?) (**$\checkmark$**)
- Generalize F1 to allow for a threshold parameter ($\checkmark$)
	- Then, implement in a way that is consistent with TensorFlow implementation ($\checkmark$, but my implementation is currently less versatile)
- Stratified sampling ($\checkmark$)
	- Pass split percent, number of classes (but could be default, so calculate), and data ($\checkmark$)
- Look into specifying weights for each sample in loss function (TensorFlow) ($\checkmark$)
- Stratified sampling for weight (when batch > samples, oversample, but compensate with weight) ($\checkmark$, but yet to compensate)
- Find out how minibatches are created in TensorFlow (is there a way to modify it?) ($\checkmark$)
	- Build a similar version that evenly distributes weights over minibatches, weights should sum to one across ALL samples ($\checkmark$, but all samples per batch, or all samples *globally?*)

> In addition to the same interface (parameters) as tf's F1, use the same vocabulary in terms of parameter names.  ($\checkmark$)
> Add checks on division by zero.  If so, add epsilon (e.g. 1e-10) to denominator. ($\checkmark$)

# 8/25/25
 
## Prep
- Update: Fastest way to compute confusion matrix thus far, using TP + FP = PPos and TP + FN = Pos
- Opted for 3% TP, 1% FN, 2% FP, 94% TN
- Generalized both F1 and AUROC
	- Note: TensorFlow has a lot of additional "pre-processing" they do for their metric implementations, so while my custom implementations seem faster on the surface, they are also currently less versatile than the TensorFlow equivalent
> Look into specifying weights for each sample in loss function (TensorFlow)
 ![[sample_weights tensorflow.png]]
> ![[better_stratified_weight_sampling.png]]

> Find out how minibatches are created in TensorFlow (is there a way to modify it?)

![[dataset-manual-batching.png]]

TensorFlow implementation always add epsilon for divisions in metrics, presumably because for most cases, adding epsilon will have little effect on the final result.



## Tasks:

Use TensorFlow `Metric` parent class for our metric implementation

Highest priority: ***SEAMLESS TensorFlow integration for metrics***
- Demo of model running with additional metrics a la Table 3
	- Passing in strings to `model.compile`... does it look up based on subclasses of `Metric` or similar? Or is it harder to make "seamless"

Extension (don't write something already written) and seamless (exact same as TensorFlow)

> For mini-batching: Should weights sum up to 1 per minibatch, or sum up to 1 *globally*?
- YES, 1 GLOBALLY, not per batch
- **All other metrics**... TensorFlow-like interface
	- "Mimic" implementations do not have to go deeper than "interface level"
	- Goal: interface similar to `model.compile`, but for our metrics (list of name strings)
- Faster stratified sampling:
	- "Collect" indices, work exclusively in index space. Is there a function to extract indices from a "bit vector"?
		- Parallelization to get totals of each class, then generate list of random indices, and fill P/N Train/Test "bins" until full
- (!!!) Make sure to leave in debugging prints for stratified samples with weights. Dr. Chan would like to see this.
- Update stratified sampling: Samples \* individual weight should equal desired total weight
	- User will specify the weight *per class*, which we will use that to compute individual weight
- Stratified sampling: Don't duplicate samples, weight per batch can be "relaxed", close but not perfect, as long as total still sums to 1

# 8/27/25

## Prep:
- So many issues... but seamless TensorFlow integration
## Meeting Notes:

- **Goal:** ***SEAMLESS TensorFlow integration for metrics***
## Tasks:
- Work on call by strings (want option for both) (!!!)
	- For seamless extensions, model after F1
- Seamless extension of AUC for lower maximum FPR (do not normalize) (!!)
	- Capping FPR leads to knowing you will have less "false alarms", also possibly minimum Precision (but not need to implement now)
	- AUROC, then AUPR
- Can we resolve conflicts with ConfusionMatrix class with Metrics? (add_weight conflicts and such) (!)
- In precision source code: (can we use this?)

![[confusion_matrix.png]]


# 8/29/25

## Prep:
- Work on calling by strings
	- **IT WORKS**, class names and aliases
> Notes on findings:
	- When passing a list of metrics to `model.compile()`, the list gets sent to some `CompileMetrics` class constructor
	- That class constructor, located in `keras/src/trainers/compile_utils`, saves the metrics to `self._user_metrics`, which is later accessed in a `build` function, which then calls a `_build_metrics_set` function, which passes each metrics to a `get_metric` functions, which calls a `get` function directly from the `keras/src/metrics` module
	- This `get` function checks a `dict[str, Metric]` object, which contains mappings for all metrics and their corresponding strings
> In short, updating this dictionary manually (it is importable) allows metrics to be referenced by string.

- Thought: On the topic of not computing the confusion matrix entries multiple times, there is a chance we could have a static class, similar to the metric class, that can accumulate the confusion matrix entries, which our other metrics can then access without having to compute them multiple times. It would have to involve some flag to show that user wants to use it, on top of being included as one of the metrics in `model.compile()`
- AUC source code seems to make use of the built-in confusion matrix features in a way that is much more digestible... might try to make use of it later on:
![[auc_confusion_matrix_usage.png]]
- What I have dubbed the "LimitedAUC" class has been implemented by extending TensorFlow's pre-existing AUC, but adjusting the `__init__()` and `result()` functions to account for any specified bounds.

## Meeting Notes:
- Temp

## Tasks:
- Minimize memory usage for metric import dictionary by updating TF dictionary directly
- Can we resolve conflicts with ConfusionMatrix class with Metrics? (add_weight conflicts and such) (!) 
- Stratified sampling tasks from 8/25/25 (!)
- LimitedAUC, decision threshold high to low, simply stop FPR/TPR calculation when max FPR is reached