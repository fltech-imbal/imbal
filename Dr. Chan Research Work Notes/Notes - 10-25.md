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
		- Allowed option for both $\checkmark$
- For ties with min bin, get min bin that is furthest from max $\checkmark$
- Ours vs scott vs silverman side by side
	- 3 datasets as benchmark

## Notes:


#### Far bin approximation
![[sarcos-far-bin.png|500]]

![[sep-c-far-bin.png|500]]

![[sep-ec-far-bin.png|500]]