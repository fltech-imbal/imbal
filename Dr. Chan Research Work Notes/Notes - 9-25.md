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

## Tasks:
- Currently, weights are actually not option during train/test split
	- For train/test split: No weights in, no weights out
- Assumption has been so far that regression data will be passed sorted
	- An oversight on my part, a fix is already mostly in place. I want to modify my unit tests to ensure everything still works as expected.
- 'classification' and 'regression' strings: store as constants that are referred to for future
- New "plain" `PyDataset` to be used as return values for train/test split
	- Allow for user basic functionality for ease of use (add/remove column, it has x, y, w)
- Double check: `PyDataset` is being accepted into `model.fit` and `model.predict`
- Begin work on reweighting
	- List in class order or dictionary
	- Weight rebalancing for regression:
		- Estimate PDF, density to weight conversion $w=\frac{1}{d}\rightarrow$ normalize 
- Read up on **kernel density estimation** to estimate PDF
	- scikit-learn has KDE
- Documentation for `sampling` sub-package
# 9/17/25

## Prep:
- Currently, weights are actually not optional during train/test split $\checkmark$ (fixed)
- 'classification' and 'regression' strings: store as constants that are referred to for future $\checkmark$
- Assumption has been so far that regression data will be passed sorted $\checkmark$ (fixed)
- Double check: `PyDataset` is being accepted into `model.fit` and `model.predict` $\checkmark$
	- As discovered last meeting, they seem to take the roughly the same inputs, but for clarity:
From `model.fit`:
![[Pasted image 20250916194833.png]]
From `model.predict`:
![[Pasted image 20250916194902.png]]
- Read up on **kernel density estimation** to estimate PDF $\checkmark$
	- Seems pretty straightforward. Biggest concern is handling distributions that are very non-gaussian. Quick KDE estimations (including the ones implemented in scikit-learn by default) seem to only be super effective when the distribution is already mostly gaussian.
- New "plain" `PyDataset` to be used as return values for train/test split $\checkmark$
	- Updated unit tests accordingly, very small changes required to make them work again.
- Begin work on reweighting $\checkmark$
	- List in class order or dictionary $\checkmark$
- Documentation for `sampling` sub-package $\checkmark$


## Tasks:
- Stratified sampling documentation, comments before example describing scenario (data)
- Hide `PyDataset` parameters that obscure main function
	- Double check for `Metric`s as well
- Provide example of duplication and weight balancing (2 batches, 1 "dragon", and 3 batches, 2 'dragons', 3 classes, 2 'dragons', 2 'unicorns')
		- Code example should also have all examples 
- 'rotation among the batches' instead of 'round robin'
- update `sampling` sub-package to `stratified_sampling`
- `StratifiedBatcher` $\rightarrow$ `DatasetWithBatching`
- `stratified_split` to `split`
- `GenericDataset` to `SimpleDataset`
- Change `reweighting` sub-package to `sample_weighting`
- `class_labels` to `labels` in `generate_sample_weights`
- Specify difference between regression and classification in documentation (both in plain words as well as in code)
- Separate `generate_sample_weights` into separate class and reg functions
	- For regression case, weight mappings are not specified, but rather a desired distribution (where the default is uniform)
- Reassess how batching is handled for regression...
	- Sorting by ascending should likely be changed to descending, unless there is a way around that
	- Am I properly reshuffling batches for regression case?
- Provide example of weight balancing $\rightarrow$ sampling by batch pipeline for Dr. Chan
- Provide option for bin-based bandwidth calculation

# 9/19/25
## Prep:
- Stratified sampling documentation, comments before example describing scenario (data) $\checkmark$
- - Provide example of duplication and weight balancing (2 batches, 1 "dragon", and 3 batches, 2 'dragons', 3 classes, 2 'dragons', 2 'unicorns') $\checkmark$
		- Code example should also have all examples $\checkmark$ 
- 'rotation among the batches' instead of 'round robin' $\checkmark$
- Hide `PyDataset` parameters that obscure main function $\checkmark$
	- Double check for `Metric`s as well $\checkmark$
- `StratifiedBatcher` $\rightarrow$ `DatasetWithBatching` $\checkmark$
- `stratified_split` to `split` $\checkmark$
- `GenericDataset` to `SimpleDataset` $\checkmark$
- Specify difference between regression and classification in documentation (both in plain words as well as in code) $\checkmark$
- Change `reweighting` sub-package to `sample_weighting` $\checkmark$
- `class_labels` to `labels` in `generate_sample_weights` $\checkmark$ 

- Separate `generate_sample_weights` into separate class and reg functions $\checkmark$
	- For regression case, weight mappings are not specified, but rather a desired distribution (where the default is uniform) $\checkmark$
		- *Note: Calculate uniform, multiply by pdf samples for each point, then re-normalize to 1*
- Reassess how batching is handled for regression... $\checkmark$
	- Sorting by ascending should likely be changed to descending, unless there is a way around that $\checkmark$
	- Am I properly reshuffling batches for regression case? $\checkmark$
- Provide example of weight balancing $\rightarrow$ sampling by batch pipeline for Dr. Chan $\checkmark$
- Provide option for bin-based bandwidth calculation $\ldots$
## Notes:
- Bin-based regression weight calculations
	- All combinations for comparing bin-bin pairs? so for 10 bins there is ${10\choose2}=45$ comparisons? 


## Tasks
- Split `classification` and `regression` creating aliases such that the user no longer has to specify `mode`
- Approaches for bin-based KDE fitting
	- AUC split at actuals, AUC split at evenly spaced trapezoids, average densities for samples in bin, average density for evenly spaced samples across bin
- Should have plot function for KDE to see visually how "well it fits"
- Bounds of $[0.01*std\_dev, 3*std\_dev]$ for KDE bandwidth
# 9/22/25
## Prep

- Split `classification` and `regression` creating aliases such that the user no longer has to specify `mode` $\checkmark$
	- Documentation needs an overhaul (!!!)
- Approaches for bin-based KDE fitting $\checkmark$
	- AUC split at actuals, AUC split at evenly spaced trapezoids, average densities for samples in bin, average density for evenly spaced samples across bin
	- Implemented average and AUC based on spaced samples $\checkmark$
- Should have plot function for KDE to see visually how "well it fits" $\ldots$
	- Working on it... need better data for better confirmation that everything is working
- Bounds of $[0.01*std\_dev, 3*std\_dev]$ for KDE bandwidth $\checkmark$
- Enum instead of string comparison $\checkmark$


## Tasks:
- Replace current AUC implementation with `(a/2 + b + c + d/2) * w`
- Fix: Frequency ratio should be highest frequency/lowest, not first bin/last 
- Integrate plotting as parameter to KDE weight balancing
- Document section 2.6 Visualization of latent space via t-SNE. Implement, using MNIST as example
	- Latent space from second to last layer (how to get?)
	- Different colors per classes, plot rarest classes last to prevent point overlap from obscuring rare samples
	- For regression, color gradient from low values (ex. blue) to high values (ex. red), plotting frequent "bins" first, least frequent bin last
	- Return plot form function, but have options for saving plot to `.png`
- For KDE testing...
	- `sep_event_1_filled_ie_trim.csv` from SEP-EC (last column is labels)
	- `sep_10mev_training.csv` from SEP-C (last column is labels)
- Double check unit tests
- Documentation fix

# 9/23/25

## Prep: 
- Replace current AUC implementation with `(a/2 + b + c + d/2) * w` $\checkmark$
- Fix: Frequency ratio should be highest frequency/lowest, not first bin/last  $\checkmark$
- Integrate plotting as parameter to KDE weight balancing $\checkmark$
- Document section 2.6 Visualization of latent space via t-SNE. Implement, using MNIST as example $\checkmark$
	- Latent space from second to last layer (how to get?) $\checkmark$
		- **Answer:** `model.get_layer(index=-2)` $\checkmark$
	- Different colors per classes, plot rarest classes last to prevent point overlap from obscuring rare samples $\checkmark$
	- For regression, color gradient from low values (ex. blue) to high values (ex. red), plotting frequent "bins" first, least frequent bin last $\checkmark$
	- Return plot form function, but have options for saving plot to `.png` $\checkmark$
- For KDE testing...
	- `sep_event_1_filled_ie_trim.csv` from SEP-EC (last column is labels) $\checkmark$
	- `sep_10mev_training.csv` from SEP-C (last column is labels) $\checkmark$
- Double check unit tests $\checkmark$
- SARCOS KDE $\checkmark$
- Documentation fix $\ldots$ (started fixes, focused on functionality first in the meanwhile)

#### Observation:
More bins does not necessarily mean better KDE. It seems to be related to whether or not there is a true "peak" and "trough" (if there are two bins of similar frequency next to each other that are the highest or lowest frequencies, it seems to cause issues)
- (this is with the averaging method, AUC does better)
![[Pasted image 20250923224301.png|500]]
### Above if 2 bins with average mode, below is 2 bins with AUC

![[Pasted image 20250923224946.png|500]]
![[Pasted image 20250923224325.png|500]]
![[Pasted image 20250923224341.png|500]]
![[Pasted image 20250923224354.png|500]]
### Above if 15 bins with average mode, below is 15 bins with AUC

![[Pasted image 20250923224641.png|500]]
![[Pasted image 20250923224408.png|500]]
![[Pasted image 20250923224420.png|500]]
#### MNIST t-SNE example
![[Pasted image 20250924072208.png|500]]
#### Example of rare plotted over common
![[Pasted image 20250924071642.png|500]]

#### KDE for SEP-C training data (seems off, but maybe not?)
![[Pasted image 20250924101632.png|500]]
#### KDE for SEP-EC training data
![[Pasted image 20250924090335.png|500]]

#### SARCOS KDE
![[Pasted image 20250924102247.png|500]]

## Tasks:
- Make sure user can specify manual bandwidth or `scott` method through our implementation
- KDE uniform - user can specify density function
- Clean up KDE code 
- Default number of KDE bins = 30
- KDE `bin_sample_count` to `samples_per_bin`
- Separate KDE functionality from regression weight generation
- `generate_weights` for regression is "wrapper" for generating densities and converting to weights 
- TSNE `perplexity` passable parameter for our implementation
	- TSNE *lower priority* than sample weighting KDE and such
- Show Dr. Chan:
	- KDE for 3 datasets (SEP-C, SEP-EC, SARCOS)
	- Look back at bins for KDE vs histogram... do they line up (SEP-C data look into)
		- Maybe manually pass bin splits to matplotlib instead of `(min, max, bin_count)`
	- Add $f_{max}, f_{min}$ and ratio to plot

# 9/26/25

## Prep
- Make sure user can specify manual bandwidth or `scott` method through our implementation $\checkmark$
- KDE uniform - user can specify density function $\checkmark$
- Default number of KDE bins = 30 $\checkmark$ (changed to 32)
- KDE `bin_sample_count` to `samples_per_bin` $\checkmark$
- Clean up KDE code $\checkmark$
	- Separate KDE functionality from regression weight generation $\checkmark$
- `generate_weights` for regression is "wrapper" for generating densities and converting to weights $\checkmark$
- TSNE `perplexity` passable parameter for our implementation $\checkmark$
	- TSNE *lower priority* than sample weighting KDE and such $\checkmark$
- Show Dr. Chan:
	- KDE for 3 datasets (SEP-C, SEP-EC, SARCOS)
	- Look back at bins for KDE vs histogram... do they line up (SEP-C data look into)
		- Maybe manually pass bin splits to matplotlib instead of `(min, max, bin_count)` $\checkmark$
	- Add $f_{max}, f_{min}$ and ratio to plot $\checkmark$

## Notes:
- SEP-C data:
	- 1476 instances of -1.6
	- About a dozen $0-1$, a dozen $1-2$, all the way up to 4
	- Two instances in 5 range, two in 6 range, 2 in 7 range, 1 in 8 range
- Will be looking into ways to optimize, however, since the 'binned' method requires typically dozens of KDEs to be generated, not sure how much faster I can get it.
- Revising documentation to be done this weekend
## Tasks
- Revise documentation
- Optimize bin-based KDE algorithms
- Percentage based padding for min-max KDE bin range (handle cases where high frequency data is ***AT*** the extremes) 
		- Add comment explaining reasoning
	- 1% default, but parameter
- Bin number in plot title
- Linear interpolation for KDE approximation for speedup
	- Compare error with actuals
	- Use same `sample_per_bin` to determine spaced points
	- Plot approximation KDE over true KDE (for debugging)

![[Pasted image 20250926082118.png|500]]

# 9/29/25

## Prep:
- Fix divide by zero error $\checkmark$ (ended up being an issue with how data was being handled during plotting (sorting step was being called after binning))
- Revise documentation 
- Optimize bin-based KDE algorithms $\checkmark$
- Percentage based padding for min-max KDE bin range (handle cases where high frequency data is ***AT*** the extremes) $\checkmark$
		- Add comment explaining reasoning $\checkmark$
	- 1% default, but parameter $\checkmark$
- Bin number in plot title $\checkmark$
- Linear interpolation for KDE approximation for speedup $\checkmark$
	- Compare error with actuals $\checkmark$
	- Use same `sample_per_bin` to determine spaced points $\checkmark$
	- Plot approximation KDE over true KDE (for debugging) $\checkmark$
	- Based on SARCOS data...
		- **Without optimization, time to compute KDE averaged over 10 runs = 34.1 seconds**
		- **With optimization, time to compute KDE averaged over 10 runs = 1.70 seconds**
		- Mean absolute error on the order of $10^{-8}$
		- Mean square error on the order of $10^{-13}$
- Delta experimentation:
	- SEP-C: bandwidth $=0.13$, $\delta=0.773$
	- SEP-EC: bandwidth $=0.102$, $\delta=0.650$
	- SARCOS: bandwidth $=0.139$, $\delta=0.949$
	- Based on SARCOS data...
		- **Without optimization, time to compute KDE averaged over 10 runs = 34.1 seconds**
		- **With optimization, time to compute KDE averaged over 10 runs = 57.3 seconds**
		- Mean absolute error on the order of $10^{-16}$
		- Mean square error on the order of $10^{-30}$
	- Why longer?
		- Ignoring the fact that I did not use binary search for finding the bounds for delta, I believe the main cause is that sklearn seems to do some pre-computation when instancing and fitting a KDE curve that presumably allows for KDEs to be slightly faster to computer after fitting
			- Tried to investigate source code, but not written in python, some strange in-between python language
		- It is a tree-based algorithm so likely nlog(n), but that operation is happening for every sample.
		- Additionally, majority of the data has majority of the data nearby (so time saves end up being not as impactful)
#### Linear Interpolation Approximation (32 bins, 10 samples per bin)
![[Pasted image 20250928182826.png|500]]
#### Approximated weights vs actuals (linear)
![[Pasted image 20250928185212.png|500]]

#### Delta-based approximations
SEP-C
![[Pasted image 20250929081016.png|500]]
SEC-EC
![[Pasted image 20250929081034.png|500]]
SARCOS
![[Pasted image 20250929081050.png|500]]