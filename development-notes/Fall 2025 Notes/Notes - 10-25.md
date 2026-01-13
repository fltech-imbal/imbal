# 10/1/25

## Prep:
 Investigate how KDE vs histogram curve changes between different bin sizes and constant bandwidth $\checkmark$
	- Also constant bins, varied bandwidth $\checkmark$ 

## Tasks:
-  # of bins proportional to number of samples
	- Default 100, or perhaps calculated based on stddev
- For ties with min bin, get min bin that is furthest from max
- Bin size determined by "points per bin"
- Ours vs scott vs silverman side by side
	- 3 datasets as benchmark

# 10/3/25

## Prep:
-  # of bins proportional to number of samples $\checkmark$
	- Bin size determined by "points per bin"
	- Default 100, or perhaps calculated based on stddev
		- Now can specify `bin_count`, `bin_width`, or `samples_per_bin`
- For ties with min bin, get min bin that is furthest from max $\checkmark$
- Ours vs scott vs silverman side by side $\checkmark$
	- 3 datasets as benchmark $\checkmark$

## Notes:


#### Far bin approximation (with old method)
![[sarcos-far-bin.png|500]]

![[sep-c-far-bin.png|500]]

![[sep-ec-far-bin.png|500]]

## Comparisons:
![[sarcos-ours-new-scott-silverman.png]]

![[sep-c-ours-new-scott-silverman.png]]

![[sep-ec-ours-new-scott-silverman.png]]

## Low Bin comparisons:
![[low-sarcos-ours-new-scott-silverman.png]]

![[low-sep-c-ours-new-scott-silverman.png]]

![[low-sep-ec-ours-new-scott-silverman.png]]

## High Bin comparisons:
![[high-sarcos-ours-new-scott-silverman.png]]

![[high-sep-c-ours-new-scott-silverman.png]]

![[high-sep-ec-ours-new-scott-silverman.png]] 

## Tasks:
- Make sure density calculations and weight calculations are separated in code $\checkmark$
- Maybe clean up/separate plotting from KDE calculations $\checkmark$
	- Fully separate KDE code form weight code $\checkmark$
	- Plotting still an option, maybe at least as a wrapper $\checkmark$
		- Wrapper in "KDE side" only $\checkmark$
- `mse` as default, `ratio` as option. $\checkmark$
- `average_samples_per_bin` = 100 by default $\checkmark$
- Remove `bin_width` option $\checkmark$
- Resolve documentation issues  $\checkmark$
	- With code examples $\checkmark$

# 10/6/25
## Prep:
- Missing code examples for KDE related methods

## Tasks
- Code examples for `regression` functions $\checkmark$
- `plot_kde` and `fit_kde` to just `regression` $\checkmark$
	- Things intended to be called by user should be under `classification` or `regression`, things meant to be "hidden" should be under `util` $\checkmark$
	- Wrappers should also be directly under `classification` or `regression` $\checkmark$
- Remove `mse` from regression $\checkmark$
- padding reasoning $\checkmark$
	- There are some instances where a lot of data points are at the edge of a bin. When comparing to the histogram, the peak may appear to be directly on the edge of a bin, which could appear as a bad fit. By padding, we are increasing the bin width as to move data points "away" from the extremes of the bins. This is more of an issue when bin count is high. $\checkmark$
- For `steps_per_bin` documentation, put (See: `optimization`) $\checkmark$
- Intro for regression `generate_weights`, show that KDE formula is $n^2$, and explain how optimizations provided are to speed up compute time at the cost of a small error. $\checkmark$
	- BOTH `local_approximation` and `linear_interpolation` $\checkmark$
	- Also explain `precision` in introduction (see below) $\checkmark$
- Get rid of `steps_per_bin`, `bin_count`, and `average_samples_per_bin` in `generate_weights`, replace with `distribution_samples` $\checkmark$
	- Default `n/10` $\checkmark$
	- Include `steps_per_bin*bin_count` in code example $\checkmark$
- add `precision` parameter for `local` optimization $\checkmark$
- add `k` parameter for `local` optimization $\checkmark$
- Change `local` to `local_approximation` $\checkmark$
- Can scikit-learn KDE object be skipped for local approximation $\times$
- Make sure to specify which parameters are ignored and when! $\checkmark$
-  Include math where applicable $\checkmark$
	- Maybe plots of error for difference datasets? As a footnote $\checkmark$
	- Maybe table of method, error, and CPU time (not O(n)) $\checkmark$
		- n/a for regular method error $\checkmark$
- Make sure function for `generate_weights` is ALWAYS A FUNCTION FROM DENSITY TO WEIGHT, not LABEL TO WEIGHT $\checkmark$
- new function: `get_density` $\checkmark$

## Notes:

#### SARCOS (44485 data points)
![[sarcos-optimization-error-hists.png|600]]

| KDE Optimization Method                     | Compute Time (sec) | MAE            |
| ------------------------------------------- | ------------------ | -------------- |
| Regular                                     | 74.58              | n/a            |
| Linear Approximation (480 bins)             | 0.45               | $1.42*10^{-3}$ |
| Local Approximation (k=480, precision=1e-6) | 17.29              | $1.78*10^{-3}$ |
#### SEP-C (1531 data points)
![[sep-c-optimization-error-hists.png|600]]

| KDE Optimization Method                     | Compute Time (sec) | MAE            |
| ------------------------------------------- | ------------------ | -------------- |
| Regular                                     | 0.13               | n/a            |
| Linear Approximation (480 bins)             | 0.00701            | $7.07*10^{-2}$ |
| Local Approximation (k=480, precision=1e-2) | 0.697              | ~0             |
#### SEC-EC (16720 data points)
![[sep-ec-optimization-error-hists.png|600]]

| KDE Optimization Method                     | Compute Time (sec) | MAE    |
| ------------------------------------------- | ------------------ | ------ |
| Regular                                     | 15.79              | n/a    |
| Linear Approximation (320 bins)             | 0.16               | $1.31$ |
| Local Approximation (k=320, precision=1e-6) | 9.51               | $1.31$ |

# 10/8/25
## Tasks:
- "misaligned" in padding explanation to "peak of KDE may appear to be on the edge, or slightly outside, of its corresponding histogram bin (due to limited pixel resolution when plotting), which is undesirable for visual comparison. Padding the minimum and maximum causes a slight shift to the bin bounds, allowing the peaks at these extremes to appear inside the bins". $\checkmark$
	- Add "binning for the histogram" first sentence $\checkmark$
	- Remove "gracefully" part $\checkmark$
- Add MAPE column $\checkmark$
- `KernelDensity` with `atol` vs `numpy` manual implementation vs current $\checkmark$
	- `scikit-learn`'s `atol` parameter had no change in performance. Manual implementation of KDE kernel saw a significant leap in performance, producing the same KDE values
- Change `precision` to `atol` $\checkmark$
## Notes:
- Fixed bug: As I thought, there was an issue with returned densities being sorted in ascending order, which was causing some issues both with the histogram displays along with the errors being calculated. This has now been fixed.


# 10/10/25
## Notes:

#### SARCOS (44485 data points)
![[dataset-1-error-histogram 1.png]]

| KDE Optimization Method                   | Compute Time (sec) | MAE             | MAPE      |
| ----------------------------------------- | ------------------ | --------------- | --------- |
| scikit-learn                              | 44.05              | n/a             | n/a       |
| scikit-learn w/ atol=1e-4                 | 21.77              | $2.71*10^{-5}$  | $0.003\%$ |
| Linear Approximation (320 bins)           | 0.25               | $1.88*10^{-4}$  | $0.152\%$ |
| Linear Approximation via scipy (320 bins) | 0.25               | $1.88*10^{-4}$  | $0.152\%$ |
| Local Approximation (k=320, atol=1e-4)    | 0.28               | $2.48*10^{-3}$  | $0.153\%$ |
| Local Approximation w/o k (atol=1e-4)     | 4.24               | $1.99*10^{-11}$ | ~$0\%$    |
#### SEP-C (1531 data points)
![[dataset-2-error-histogram 1.png]]

| KDE Optimization Method                         | Compute Time (sec) | MAE             | MAPE     |
| ----------------------------------------------- | ------------------ | --------------- | -------- |
| scikit-learn                                    | 0.057              | n/a             | n/a      |
| scikit-learn w/ atol=1e-4                       | 0.057              | $8.84*10^{-9}$  | ~$0\%$   |
| Linear Approximation (320 bins)                 | 0.0029             | $2.00*10^{-2}$  | $0.55\%$ |
| Linear Approximation via scipy (%% 320 bins) %% | 0.0036             | $2.00*10^{-2}$  | $0.55\%$ |
| Local Approximation (k=320, atol=1e-4)          | 0.012              | $4.56*10^{-13}$ | ~$0\%$   |
| Local Approximation w/o k (atol=1e-4)           | 0.013              | $4.57*10^{-13}$ | ~$0\%$   |
#### SEC-EC (16720 data points)
![[dataset-3-error-histogram 1.png]]

| KDE Optimization Method                   | Compute Time (sec) | MAE             | MAPE      |
| ----------------------------------------- | ------------------ | --------------- | --------- |
| scikit-learn                              | 7.33               | n/a             | n/a       |
| scikit-learn w/ atol=1e-4                 | 6.79               | $2.23*10^{-5}$  | $0.03\%$  |
| Linear Approximation (320 bins)           | 0.064              | $4.81*10^{-3}$  | $0.156\%$ |
| Linear Approximation via scipy (320 bins) | 0.072              | $4.81*10^{-3}$  | $0.156\%$ |
| Local Approximation (k=320, atol=1e-4)    | 0.14               | $4.0*10^{-3}$   | $0.140\%$ |
| Local Approximation w/o k (atol=1e-4)     | 1.02               | $1.52*10^{-11}$ | ~$0\%$    |

## Tasks:
- Add row for `sklearn` with same tolerance directly as `atol` (!!!)
- `sklearn` implementation is likely multidimensional. leave `atol` as option for `get_densities`
- Remove our `local` interpolation, extend our linear interpolation for multi-dimensional (!!!)
	- If our `local` is significantly faster than `sklean` with `atol`, there might be a reason to keep it
	- See table above... $0.014$ seconds or $0.14$ seconds?

# 10/13/25

## Prep:
 - Add row for `sklearn` with same tolerance directly as `atol` $\checkmark$
 - See table above... $0.014$ seconds or $0.14$ seconds? $\checkmark$ (it was, in fact, a typo)
 - If our `local` is significantly faster than `sklearn` with `atol`, there might be a reason to keep it
	 - While it is significantly faster, it is relatively comparable to the linear interpolation method in terms of performance and error, except in the rare case that nearly all of the data is "highly concentrated" (for example, SEP-C having 90% of data points be the same value)
## Notes:
- Now that we know that `atol` should just be passed as error per point, speed improvements are more noticeable, but still nothing crazy
	- Side note: Since `atol` is a parameter in `scikit-learn`'s `KernelDensity` object, it must be specified during `fit_kde`, rather than during `get_densities`
- I have confirmed that `scikit-learn`'s `KernelDensity` object does in fact work on multidimensional data
-  I have looked into the math for multidimensional lerp, there exists implementations from `scipy`, which is already a dependency of TensorFlow
## Tasks:
- Fifth row: `local_approximation` with no k value (full sampling)
	- To see how speed compares with `atol`
	- If speed is no longer significantly faster than `atol`, we will get rid of `local_approximation`
- Sixth row: `linear_interpolation` with `scipy.RegularGridInterpolator` (in 1D)
- Reorganize, return bandwidth, not `KernelDensity`, skip `scikit-learn`
# 10/15/25
## Prep:
- Fifth row: `local_approximation` with no k value (full sampling) $\checkmark$
	- To see how speed compares with `atol` $\checkmark$
	- If speed is no longer significantly faster than `atol`, we will get rid of `local_approximation` $\checkmark$
- Sixth row: `linear_interpolation` with `scipy.RegularGridInterpolator` (in 1D) $\checkmark$
- Reorganize code, return bandwidth, not `KernelDensity`, skip `scikit-learn` $\checkmark$

Notes:
- Updated code to work in multidimensions
	- Still not positive this works entirely as expected, but early tests are promising
	- Converting from 1D to nD took longer than I thought. Linear interpolation was an easy enough problem to solve, however nD iterative bandwidth using KL convergence was much trickier
- Once again, documentation needs a bit of an overhaul.

## Tasks:
- For `local_approximation`, get rid of `k`, but allow for our (faster and more accurate) method for 1D, otherwise leverage `scikit-learn`, which already has nD $\checkmark$
- For linear interpolation, always use `scikit-learn` $\checkmark$
- In documentation intro, explain methodology for local approximation in 1D (amortized $O(n)$ vs $O(nlogn)$) $\checkmark$
- Documentation for `fit_kde`
	- Introduction
		- Two methods for faster computation at the cost of lower accuracy, `atol` and interpolation $\checkmark$
		- 1D for local uses custom implementation for faster results $\checkmark$
			- `scikit-learn` for higher dimension $\checkmark$
		- Interpolation via `scipy` $\checkmark$
	- Parameters $\checkmark$
	- Examples
		- Where tables and histograms are
		- Show time differences for better in one dimensional
- KL divergence (or other metrics too) should occur per bin, not per sub-bin $\checkmark$
- instead of multiplying by bin width and then "dividing by one", sum up densities in bin, and divide by the sum of all sampled densities $\checkmark$
- Shift by half sub-bin width $\checkmark$
	- Argue that the sum of the sub-bin samples (height) is proportional to the AUC (just missing multiplication by a constant width v)
- 2D table $\checkmark$
	- Regular $\checkmark$
	- Local (`atol`) $\checkmark$
	- Linear interpolation $\checkmark$
- Make `colorbar` ranges for 2D histograms have same color range for visual comparison $\checkmark$ 
- Sorted vs unsorted before passing to `atol` $\checkmark$
## Notes:
- for `atol`, sorting, there was no significant difference between sorted and unsorted

2D data
![[true-2d-distribution.png|600]]
KDE
![[2d-kde-scatter.png|600]]
Errors
![[2d-data-error-histogram.png]]

| KDE Optimization Method            | Compute Time (sec) | MAE            | MAPE      |
| ---------------------------------- | ------------------ | -------------- | --------- |
| Regular                            | 5.43               | n/a            | n/a       |
| Local Approximation (atol=1e-4)    | 0.72               | $3.26*10^{-5}$ | $0.014\%$ |
| Linear Interpolation (64 bins/dim) | 0.40               | $1.12*10^{-2}$ | $4.4\%$   |

## Tasks:
- Redo 2D scatter plot to have evenly-spaced sampling across XY $\checkmark$
- Fix `plot_kde` link $\checkmark$
- `get_densities` - replace "optimization" with "estimation methods to reduce time" $\checkmark$
	- Add to linear interpolation description that there is no guarantee of some maximum error $\checkmark$
	- Double check parameter descriptions (ex. `atol` should include "absolute") $\checkmark$
	- Add examples for local estimation and linear interpolation $\checkmark$
		- Make sure regular example is up-to-date $\checkmark$
		- Remove error distribution charts $\checkmark$
		- change "precision" to "`atol`" $\checkmark$
- After parameters...
	- Organization of examples $\checkmark$
		- Example for regular $\checkmark$
		- Example for local $\checkmark$
		- Example for interpolation $\checkmark$
	- Comparison of methods $\checkmark$
		- Descriptions/distributions of datasets $\checkmark$
		- table of comparisons regular/local/linear $\checkmark$
	- Show difference between our local 1D and regular 1D (`KernelDensity` with `atol`) $\checkmark$
		- (To show why ours is better for 1D) $\checkmark$
		- For all 3 datasets, a table (two rows per table, theirs and ours) $\checkmark$
	- 2D example $\checkmark$
		- "Toy dataset with \[this covariance\]" $\checkmark$
		- Provide example w/ comparison $\checkmark$
- For `fit_kde`
	- Double check intro with KL divergence $\checkmark$
	- Get rid of "ratio" option $\checkmark$
	-  Code examples (do they need to be updated) $\checkmark$
	- Show example of comparison KL/`scott`/`silverman` $\checkmark$
	
- TSNE
	- Make sure to still do most/least frequent binning for regression $\checkmark$
	- For classification:
		- MNIST can be used, but make it imbalanced $\checkmark$
			- ex. 0 is frequency, all the way to 9 which is rare $\checkmark$
			- plot TSNE plot of second to last layer $\checkmark$
	- For regression:
		- MNIST can be "relabeled" to be a regression problem (one output instead of one-hot vector) $\checkmark$

# 10/23/25

## Notes:
- The reason why the grid was generating a scatter plot of seemingly all zeroes is because by default, our function builds the KDE from and then samples densities for the *same input data*.
	- The KDE being generated was not for the data used to generated the data, but rather the grid, but using the found fitted bandwidth for the gaussian data, resulting in a plot of nearly all zeroes
- Ours vs theirs 1D `atol` tables
	- SARCOS: 23.0 seconds, 2.714e-05 MAE, 0.003% MAPE
	- SEP-C: 0.057 seconds, 1.698e-5 MAE, ~0% MAPE
	- SEP-EC: 6.21 seconds, 2.251e-5 MAE, 0.003% MAPE
- 2D toy dataset table

| Method                             | Time | MAE            | MAPE     |
| ---------------------------------- | ---- | -------------- | -------- |
| Regular                            | 3.41 | n/a            | n/a      |
| Local Approximation (atol=1e-4)    | 0.78 | $3.83*10^{-5}$ | $0.04\%$ |
| Linear Interpolation (64 bins/dim) | 0.42 | $2.21*10^{-3}$ | $1.43\%$ |
- TSNE - MNIST reuse for regression did not yield the best results... but is good enough for proof of concept I think

# 10/27/25

## Prep:
- For documentation of `get_densities`, "approximation", not "estimation" $\checkmark$
- Change examples for `get_densities` so that errors are visible for all approximations $\checkmark$
	- Reduce size of all figures $\checkmark$
		- alternate plot-table... instead of plot-plot-plot table-table-table $\checkmark$
		- side-by-side comparisons for 2D data $\checkmark$
	- per-bin... can dots be bigger and square? fill whitespace $\checkmark$
		- Goal is near 1-to-1 with histogram $\checkmark$
- `fit_kde` - `fit_method` default not in code font $\checkmark$
- `kl_divergence` requires more description of algorithm $\checkmark$
	- 'beam search'... coarse grain to fine grain $\checkmark$
		- say "$k$ candidates" instead of searches $\checkmark$
	- provide explanation via examples of "$k$ candidates", "iterations", "coarse to fine grain" $\checkmark$
	- split $k$ into $r$ rounds and $k$ candidates $\checkmark$
		- mention default $k$, and $r$ is arbitrary $\checkmark$
	- mention and explain stopping criteria $\checkmark$
		- no improvement in kl from previous iteration $\checkmark$
		- kl divergence is within some tolerance of 0 $\checkmark$
		-  make sure tolerance value is being used for "little to no improvement" case $\checkmark$
	- Add time comparison table for `fit_kde` $\checkmark$
		- maybe add bin values to plots? $\checkmark$
	- Reserve section 3 for performance comparisons alone, no description of algorithms $\checkmark$
		- Move current section 3 description up to section 1, ensure kl divergence's dependence on `bin_count` is explained $\checkmark$
	
	- For `tsne` classification
		- don't use gradient, use discrete colors, and create legend instead of gradient color bar $\checkmark$
		- Shapes instead of color? $\checkmark$
			- **Doesn't seem as feasible as colors**
			- Is there a way to pass this as parameter?

# 10/29/25
## Tasks:
- For `fit_kde` $\checkmark$
	- Introduce term "stopping criteria" $\checkmark$
	- When describing $r$ in time complexity, refer back to "stopping criteria", then say rough expected value $\checkmark$
	- Describe stopping criteria scenarios $\checkmark$
		- `tolerance` example, explained an potentially in code $\times$ 
		- tolerance should default to $0$, not specified means use only "no improvement" stopping criteria $\checkmark$
- For `get_densities` $\checkmark$
	- Introduce/define/describe delta $\checkmark$
		- Not "some delta ... such that", rather "\[define delta\] ... which we call delta." $\checkmark$
		- Mention inverse gaussian for calculation of delta $\checkmark$
			- Range is +/- delta $\checkmark$
			- All values beyond delta have a density value below `atol/n` $\checkmark$
		- Mention motivation - further away points contribute very little to density $\checkmark$
			- some min distance, which we call delta $\checkmark$
- Binary classification example to highlight rarer classes being plotted "on top" $\checkmark$
	- Mention in documentation example (begin documentation) $\checkmark$
	- Examples: $\checkmark$
		- Rare plotted last $\checkmark$
		- Models can perform worse when data is imbalanced, which can be shown visually with TSNE $\checkmark$
- For `tsne` $\checkmark$
	- don't specify colors, instead make multiple scatter plots and make `matplotlib` handle it $\checkmark$
		- But still allow for user colors, shapes, size $\checkmark$
			- 3 lists, not list of tuples $\checkmark$
		- Provide documentation example $\checkmark$
		- Replace descending/ascending with binning, pass bins by descending frequency $\checkmark$
	- Example ordering: $\checkmark$
		- Rare plotted first $\checkmark$
		- shape/size/color $\checkmark$
		- Imbalance can be a problem, shown visually $\checkmark$
- plotting KDE $\checkmark$
	- default includes bandwidth and bin count in title, not min/max frequencies $\checkmark$
	- all three are parameters that can be enabled/disabled $\checkmark$

## Notes:
* Found a bug: was using `atol/n` for finding delta, but should have been using just `atol`, since Gaussians are divided by $n$ later on


## Tasks:
- For `fit_kde`: $\checkmark$
	- `tolerance` example, explained an potentially in code $\checkmark$ 
	- Make sure fine grain searches do not include redundant endpoints and redundant "best sample" $\checkmark$
		- Use "center-bin" sampling $\checkmark$

- Add inverse equation for $\delta$ ("in other words" or "concretely") $\checkmark$
	
- Update errors on `get_densities` tables $\checkmark$
	- Also update local approximation code example $\checkmark$
		- SARCOS time update as well $\checkmark$
- update `plot_kde_1d` documentation with examples of different plot parameters $\checkmark$
- `TSNE` $\checkmark$
	- Fix binary classification plots to be the same TSNE $\checkmark$
	- Use code for color/shape/size example $\checkmark$
	- Code examples for regression/classification $\checkmark$
- Read $2.4$ $\checkmark$
	- Find implementations of LIME and SHAP $\checkmark$
		- Papers have original implementations, but might be in `TF`/`scipy`/`scikit-learn` $\checkmark$


# 10/31/25

## Tasks:
- Skip floor(steps/2) in `fit_kde` $\checkmark$
- `plot_kde_1d` images smaller (pull images from docstring and put in markdown)
- frequent/rare example fix for regression
	- More than 2 classes for regression
- Import `LIME` and `SHAP`
	- Build basic explainer wrappers
	- Use MNIST for examples/prototyping
		- Classification **AND** regression
	- Two examples:
		- Image dataset
		- Tabular dataset