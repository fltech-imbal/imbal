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
	- Maybe plots of error for difference datasets? As a footnote $\times$
	- Maybe table of method, error, and CPU time (not O(n)) $\times$
		- n/a for regular method error $\times$
- Make sure function for `generate_weights` is ALWAYS A FUNCTION FROM DENSITY TO WEIGHT, not LABEL TO WEIGHT $\checkmark$
- new function: `get_density` $\checkmark$