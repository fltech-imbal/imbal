# 11/3/26

## Thesis

In order to determine if gradient conflicts are reducing, we need a way to measure them. Also, most existing methods reduce conflicts either by performing projections or a weighted sum of the gradients.

| Paper                                                | How they reduce gradient conflict                                                                                                                                                                                                                        | How do they measure gradient conflict                                          |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Long-Tailed Learning as Multi-Objective Optimization | Weighted sum of group-based gradients. One term, minimizing the norm of the gradients.                                                                                                                                                                   | Similarity of class-level and batch-level gradients                            |
| Two Fists, One Heart                                 | Subtracting the sum of the projections onto every other gradient                                                                                                                                                                                         | They don't?                                                                    |
| TS-MOF                                               | Projection, only looking at conflicting gradients. Sum of differences between original and projected gradient. Gradient conflicts are reduced between experts. Each expert/tasks uses a different loss function, goal is to minimize all loss functions. | Using "conflict probability" (conflict in classification across experts/tasks) |
| Pareto Deep Long-Tailed Recognition                  | Average (original) gradient plus a weighted sum "adjustment" gradient. Two terms, one term enforces orthogonality between original gradient and adjustment, while other minimizes the norm of the adjustment gradient.                                   | Non-conflicting vs. conflicting vs. dominated conflicting (Fig. 2)             |
Two Hearts, eq. 13. An alternative to subtracting the sum of all projections could be to subtract the projection of the sum of all other gradients.
- Furthermore, eq. 6 in TS-MOF sums the gradients after their conflict-projections are subtracted

Also, Two Hearts does project-and-subtract regardless of whether gradients are conflicting or not. TS-MOF only performs projections on conflicting gradients.

Finish the above table for Thursday. Also, answer...
- How is group-level different from projection, different from weighted average
- Are there other ways we can do group-level "quickly"

**What is "conflict probability" (from TS-MOF, Figure 3a):**
- Since TS-MOF uses multiple prediction heads during training, my best inference as to what "conflict probability" is is the probability that two different heads predict different classes.
- From Figure 3a caption: "Subfigure (a) compares the probability of output conflicts occurring when different groups of strategies are fused using TS-MOF"
- They do no elaborate on conflict probability much beyond this
**How is group-level different from projection, different from weighted average**
- Group-level gradients group class gradients that are similar, such that there are a fixed number of groups whose gradient conflicts need to be resolved. In the case of *Long-Tailed Learning as Multi-Objective Optimization*, these conflicts are resolved by performing a weighted sum of the gradients.
- Projection reduces conflicts by subtracting the conflicting components of a gradient. The conflicting components can be found directly by projecting onto another gradient. This should only be done only with gradients that already conflict. Subtracting the projection onto a non-conflicting gradient may produce a conflict.
- Weighted average does not contain projection, instead resolving conflicts by computing weights for each gradient that reduce conflicts, then performing a weighted sum of the original gradients.
	- Weighted average can be viewed
**Are there other ways we can do group-level gradients "quickly"?**
- *Long-Tailed Learning as Multi-Objective Optimization* group gradients, then uses a weighted sum to reduce conflicts. It may be worth investigating grouping gradients, followed by projection to reduce conflicts.
- $k$-means clustering to group samples without needing to consider class (also works for regression)

- Combine gradient conflict test for previous method and new method (with/without cosine)
	- Include magnitude print-out per conflict-reduction method (a, b, c, original, and final gradient)
		- a, b, and c are all 1 now, but that could change in the future

- For paper
	- Existing methods (Two Heart, TSMOF) $\checkmark$
	- 2 methods tested (sum on "inside", w/ and w/o cosine) $\checkmark$
	- Ways of binning $\checkmark$
		- Number of bins based on labels $\checkmark$
		- Could look into grouping by $k$-means cluster of per-sample gradients as well $\checkmark$
	- Can include comparison of synthetic/hand-picked scenarios (based on toy script)

- Move to subset of SEP-C. Compare before/after.
	- Print magnitude of different classes. We expect that common class should be larger than smaller class.
- Thought: using weights of samples to determine magnitude of each conflict-reduced class gradient before summing to produce final conflict-reduced gradient?

---
## Paper
- ...

---
## NASA
- https://cs.fit.edu/~pkc/papers/sdm20.pdf for classification representation loss
	- Reference? (link?) on tutorial?
- `plot_similarity_matrix` compatibility with classification, then document + tutorials
	- Instead use SEP-C (above ln(10), background, and "elevated" (greater than background but below ln(10)))
	- Multiple solar flare classes for SDOBenchmark
\- Order of operations
	- distance matrix
		- regression **(1)** $\checkmark$
			- tabular$\checkmark$
			- image $\checkmark$
		- classification **(2)** $\checkmark$
			- tabular (script made, tutorial next)
			- image (script made, tutorial next)
	- representation loss $\checkmark$
		- regression **(2)** $\checkmark$
			- tabular $\checkmark$
			- image $\checkmark$
		- classfication **(3)**
			- tabular (script made, tutorial next)
			- image (script made, tutorial next)

---

# 11/10/26

## Thesis 

In order to determine if gradient conflicts are reducing, we need a way to measure them. Also, most existing methods reduce conflicts either by performing projections or a weighted sum of the gradients.

| Paper                                                | How they reduce gradient conflict                                                                                                                                                                                                                        | How do they measure gradient conflict                                          |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Long-Tailed Learning as Multi-Objective Optimization | Weighted sum of group-based gradients. One term, minimizing the norm of the gradients.                                                                                                                                                                   | Similarity of class-level and batch-level gradients                            |
| Two Fists, One Heart                                 | Subtracting the sum of the projections onto every other gradient                                                                                                                                                                                         | They don't?                                                                    |
| TS-MOF                                               | Projection, only looking at conflicting gradients. Sum of differences between original and projected gradient. Gradient conflicts are reduced between experts. Each expert/tasks uses a different loss function, goal is to minimize all loss functions. | Using "conflict probability" (conflict in classification across experts/tasks) |
| Pareto Deep Long-Tailed Recognition                  | Average (original) gradient plus a weighted sum "adjustment" gradient. Two terms, one term enforces orthogonality between original gradient and adjustment, while other minimizes the norm of the adjustment gradient.                                   | Non-conflicting vs. conflicting vs. dominated conflicting (Fig. 2)             |
Two Hearts, eq. 13. An alternative to subtracting the sum of all projections could be to subtract the projection of the sum of all other gradients.
- Furthermore, eq. 6 in TS-MOF sums the gradients after their conflict-projections are subtracted

Also, Two Hearts does project-and-subtract regardless of whether gradients are conflicting or not. TS-MOF only performs projections on conflicting gradients.

Finish the above table for Thursday. Also, answer...
- How is group-level different from projection, different from weighted average
- Are there other ways we can do group-level "quickly"

**What is "conflict probability" (from TS-MOF, Figure 3a):**
- Since TS-MOF uses multiple prediction heads during training, my best inference as to what "conflict probability" is is the probability that two different heads predict different classes.
- From Figure 3a caption: "Subfigure (a) compares the probability of output conflicts occurring when different groups of strategies are fused using TS-MOF"
- They do no elaborate on conflict probability much beyond this
**How is group-level different from projection, different from weighted average**
- Group-level gradients group class gradients that are similar, such that there are a fixed number of groups whose gradient conflicts need to be resolved. In the case of *Long-Tailed Learning as Multi-Objective Optimization*, these conflicts are resolved by performing a weighted sum of the gradients.
- Projection reduces conflicts by subtracting the conflicting components of a gradient. The conflicting components can be found directly by projecting onto another gradient. This should only be done only with gradients that already conflict. Subtracting the projection onto a non-conflicting gradient may produce a conflict.
- Weighted average does not contain projection, instead resolving conflicts by computing weights for each gradient that reduce conflicts, then performing a weighted sum of the original gradients.
	- Weighted average can be viewed
**Are there other ways we can do group-level gradients "quickly"?**
- *Long-Tailed Learning as Multi-Objective Optimization* group gradients, then uses a weighted sum to reduce conflicts. It may be worth investigating grouping gradients, followed by projection to reduce conflicts.
- $k$-means clustering to group samples without needing to consider class (also works for regression)

- Combine gradient conflict test for previous method and new method (with/without cosine)
	- Include magnitude print-out per conflict-reduction method (a, b, c, original, and final gradient)
		- a, b, and c are all 1 now, but that could change in the future

- For paper
	- Can include comparison of synthetic/hand-picked scenarios (based on toy script)

- Move to subset of SEP-C. Compare before/after. $\checkmark$
	- Print magnitude of different classes. We expect that common class should be larger than smaller class. $\checkmark$

* Should plot 1 plot per method (per the pareto deep learning paper): three lines that are the per-class similarity over time (include baseline as a point of comparison)
	* Use gradient similarities from last batch in each epoch
	* For clustering-based binning, plot average, min, and max per method
* Need AORE values, rare and common MAE
	* +true vs. predicted plots

* Make sure normalization can be enabled/disabled. May need a loss function builder $\checkmark$

- Normalization of vectors off. Vary label-based binning vs. gradient clustering $\checkmark$
- With time, try normalization on/off
* Stretch goal: Options for how to address the issue of magnitude (not "vanilla" distribution, and not "50/50"/normalized gradients per bin) 

- Thought: using weights of samples to determine magnitude of each conflict-reduced class gradient before summing to produce final conflict-reduced gradient?

---
## Paper
- ...

---
## NASA
- https://cs.fit.edu/~pkc/papers/sdm20.pdf for classification representation loss
	- Reference? (link?) on tutorial?
- `plot_similarity_matrix` compatibility with classification, then document + tutorials
	- Instead use SEP-C (above ln(10), background, and "elevated" (greater than background but below ln(10)))
	- Multiple solar flare classes for SDOBenchmark
\- Order of operations
	- distance matrix
		- regression **(1)** $\checkmark$
			- tabular$\checkmark$
			- image $\checkmark$
		- classification **(2)** $\checkmark$
			- tabular (script made, tutorial next)
			- image (script made, tutorial next)
	- representation loss $\checkmark$
		- regression **(2)** $\checkmark$
			- tabular $\checkmark$
			- image $\checkmark$
		- classfication **(3)**
			- tabular $\checkmark$
			- image $\checkmark$

---
# 11/12/26

## Thesis 

In order to determine if gradient conflicts are reducing, we need a way to measure them. Also, most existing methods reduce conflicts either by performing projections or a weighted sum of the gradients.

| Paper                                                | How they reduce gradient conflict                                                                                                                                                                                                                        | How do they measure gradient conflict                                          |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Long-Tailed Learning as Multi-Objective Optimization | Weighted sum of group-based gradients. One term, minimizing the norm of the gradients.                                                                                                                                                                   | Similarity of class-level and batch-level gradients                            |
| Two Fists, One Heart                                 | Subtracting the sum of the projections onto every other gradient                                                                                                                                                                                         | They don't?                                                                    |
| TS-MOF                                               | Projection, only looking at conflicting gradients. Sum of differences between original and projected gradient. Gradient conflicts are reduced between experts. Each expert/tasks uses a different loss function, goal is to minimize all loss functions. | Using "conflict probability" (conflict in classification across experts/tasks) |
| Pareto Deep Long-Tailed Recognition                  | Average (original) gradient plus a weighted sum "adjustment" gradient. Two terms, one term enforces orthogonality between original gradient and adjustment, while other minimizes the norm of the adjustment gradient.                                   | Non-conflicting vs. conflicting vs. dominated conflicting (Fig. 2)             |
Two Hearts, eq. 13. An alternative to subtracting the sum of all projections could be to subtract the projection of the sum of all other gradients.
- Furthermore, eq. 6 in TS-MOF sums the gradients after their conflict-projections are subtracted

Also, Two Hearts does project-and-subtract regardless of whether gradients are conflicting or not. TS-MOF only performs projections on conflicting gradients.

Finish the above table for Thursday. Also, answer...
- How is group-level different from projection, different from weighted average
- Are there other ways we can do group-level "quickly"

**What is "conflict probability" (from TS-MOF, Figure 3a):**
- Since TS-MOF uses multiple prediction heads during training, my best inference as to what "conflict probability" is is the probability that two different heads predict different classes.
- From Figure 3a caption: "Subfigure (a) compares the probability of output conflicts occurring when different groups of strategies are fused using TS-MOF"
- They do no elaborate on conflict probability much beyond this
**How is group-level different from projection, different from weighted average**
- Group-level gradients group class gradients that are similar, such that there are a fixed number of groups whose gradient conflicts need to be resolved. In the case of *Long-Tailed Learning as Multi-Objective Optimization*, these conflicts are resolved by performing a weighted sum of the gradients.
- Projection reduces conflicts by subtracting the conflicting components of a gradient. The conflicting components can be found directly by projecting onto another gradient. This should only be done only with gradients that already conflict. Subtracting the projection onto a non-conflicting gradient may produce a conflict.
- Weighted average does not contain projection, instead resolving conflicts by computing weights for each gradient that reduce conflicts, then performing a weighted sum of the original gradients.
	- Weighted average can be viewed
**Are there other ways we can do group-level gradients "quickly"?**
- *Long-Tailed Learning as Multi-Objective Optimization* group gradients, then uses a weighted sum to reduce conflicts. It may be worth investigating grouping gradients, followed by projection to reduce conflicts.
- $k$-means clustering to group samples without needing to consider class (also works for regression)

- Combine gradient conflict test for previous method and new method (with/without cosine)
	- Include magnitude print-out per conflict-reduction method (a, b, c, original, and final gradient)
		- a, b, and c are all 1 now, but that could change in the future

- For paper
	- Can include comparison of synthetic/hand-picked scenarios (based on toy script)

- Move to subset of SEP-C. Compare before/after. $\checkmark$
	- Print magnitude of different classes. We expect that common class should be larger than smaller class. $\checkmark$

* Should plot 1 plot per method (per the pareto deep learning paper): three lines that are the per-class similarity over time (include baseline as a point of comparison)
	* Use gradient similarities from last batch in each epoch
	* For clustering-based binning, plot average, min, and max per method
* Need AORE values, rare and common MAE
	* +true vs. predicted plots

* Make sure normalization can be enabled/disabled. May need a loss function builder $\checkmark$

- Normalization of vectors off. Vary label-based binning vs. gradient clustering $\checkmark$
- With time, try normalization on/off
* Stretch goal: Options for how to address the issue of magnitude (not "vanilla" distribution, and not "50/50"/normalized gradients per bin) 

For each method, explore which binning/normalization method does best, and investigate plots
* Do this for each dataset (4 methods * 2 plots * 2 datasets)
* Same "optimal" method across both datasets
* Then, report best singular methods across all possible combinations, across both datasets

**Use the phrasing "adjusted/modified gradient"**

## SEP-E

| Method           | Binning | Normalization? | Epochs | $MAE$      | $MAE_R$    | $AORE$       |
| ---------------- | ------- | -------------- | ------ | ---------- | ---------- | ------------ |
| Baseline         | N/A     | N/A            | 162.0  | 0.2358     | 0.2466     | 0.2402       |
| Two Fists        | Label   | $\times$       | 77.2   | 0.6488     | 0.6377     | 0.6443       |
| TS-MOF           | Label   | $\times$       | 370.2  | 0.2377     | 0.2479     | 0.2418       |
| Inner Sum        | Label   | $\times$       | 483.6  | 0.2349     | 0.2482     | 0.2403       |
| Inner Sum w/ Cos | Label   | $\times$       | 396.6  | ==0.2378== | 0.2406     | ==🟠0.2389== |
| Two Fists        | Cluster | $\times$       | 44.4   | 1.7017     | 1.0555     | 1.4411       |
| TS-MOF           | Cluster | $\times$       | 254.0  | 0.2703     | 0.2598     | 0.2661       |
| Inner Sum        | Cluster | $\times$       | 436.6  | 0.2384     | 0.2308     | ==🟠0.2354== |
| Inner Sum w/ Cos | Cluster | $\times$       | 430.4  | 0.2354     | ==0.2256== | ==0.2314==   |
| Two Fists        | Label   | $\checkmark$   | 91.0   | 0.4196     | 0.3267     | 0.3821       |
| TS-MOF           | Label   | $\checkmark$   | 332.2  | 0.2409     | 0.265      | 0.2506       |
| Inner Sum        | Label   | $\checkmark$   | 336.0  | 0.2422     | 0.2535     | 0.2468       |
| Inner Sum w/ Cos | Label   | $\checkmark$   | 338.0  | 0.2392     | 0.2537     | 0.245        |
| Two Fists        | Cluster | $\checkmark$   | 49.4   | 1.1022     | 1.3737     | 1.2117       |
| TS-MOF           | Cluster | $\checkmark$   | 369.2  | 0.2663     | 0.2753     | 0.2699       |
| Inner Sum        | Cluster | $\checkmark$   | 116.0  | 0.4832     | 0.4621     | 0.4747       |
| Inner Sum w/ Cos | Cluster | $\checkmark$   | 128.2  | 0.4241     | 0.4152     | 0.4205       |
## SEP-C

| Method           | Binning | Normalization? | Epochs | $MAE$      | $MAE_R$    | $AORE$       |
| ---------------- | ------- | -------------- | ------ | ---------- | ---------- | ------------ |
| Baseline         | N/A     | N/A            | 312.6  | 0.2344     | 3.0403     | 1.6093       |
| Two Fists        | Label   | $\times$       | 57.4   | 1.4007     | 3.2506     | 2.3072       |
| TS-MOF           | Label   | $\times$       | 121.4  | 0.9938     | 3.4485     | 2.1966       |
| Inner Sum        | Label   | $\times$       | 296.8  | ==0.2145== | 2.775      | ==1.4692==   |
| Inner Sum w/ Cos | Label   | $\times$       | 161.2  | 0.271      | 2.8229     | ==🟠1.5215== |
| Two Fists        | Cluster | $\times$       | 27.0   | 1.6766     | 4.3717     | 2.9973       |
| TS-MOF           | Cluster | $\times$       | 114.2  | 0.3122     | 2.9424     | 1.601        |
| Inner Sum        | Cluster | $\times$       | 96.0   | 0.5829     | ==2.6099== | ==🟠1.5762== |
| Inner Sum w/ Cos | Cluster | $\times$       | 189.4  | 0.2205     | 3.5962     | 1.8746       |
| Two Fists        | Label   | $\checkmark$   | 63.0   | 0.3614     | 4.3551     | 2.3184       |
| TS-MOF           | Label   | $\checkmark$   | 52.0   | 1.2166     | 3.949      | 2.5558       |
| Inner Sum        | Label   | $\checkmark$   | 272.4  | 0.2857     | 3.2265     | 1.7267       |
| Inner Sum w/ Cos | Label   | $\checkmark$   | 181.6  | 0.2964     | 3.1511     | 1.6952       |
| Two Fists        | Cluster | $\checkmark$   | 61.0   | 1.217      | 4.0919     | 2.6257       |
| TS-MOF           | Cluster | $\checkmark$   |        |            |            |              |
| Inner Sum        | Cluster | $\checkmark$   |        |            |            |              |
| Inner Sum w/ Cos | Cluster | $\checkmark$   |        |            |            |              |

- Thought: using weights of samples to determine magnitude of each conflict-reduced class gradient before summing to produce final conflict-reduced gradient?

---
## Paper
- ...

---
## NASA
- https://cs.fit.edu/~pkc/papers/sdm20.pdf for classification representation loss
	- Reference? (link?) on tutorial?
- `plot_similarity_matrix` compatibility with classification, then document + tutorials
	- Instead use SEP-C (above ln(10), background, and "elevated" (greater than background but below ln(10)))
	- Multiple solar flare classes for SDOBenchmark
\- Order of operations
	- distance matrix
		- regression **(1)** $\checkmark$
			- tabular$\checkmark$
			- image $\checkmark$
		- classification **(2)** $\checkmark$
			- tabular (script made, tutorial next)
			- image (script made, tutorial next)
	- representation loss $\checkmark$
		- regression **(2)** $\checkmark$
			- tabular $\checkmark$
			- image $\checkmark$
		- classfication **(3)**
			- tabular $\checkmark$
			- image $\checkmark$