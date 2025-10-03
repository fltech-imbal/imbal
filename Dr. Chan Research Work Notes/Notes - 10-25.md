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
- Make sure density calculations and weight calculations are separated in code
- Maybe clean up/separate plotting from KDE calculations
	- Fully separate KDE code form weight code
	- Plotting still an option, maybe at least as a wrapper
		- Wrapper in "KDE side" only
- `mse` as default, `ratio` as option.
- `average_samples_per_bin` = 100 by default
- Remove `bin_width` option
- Resolve documentation issues
	- With code examples