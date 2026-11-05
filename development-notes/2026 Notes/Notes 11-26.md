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

- Move to subset of SEP-C. Compare before/after.
	- Print magnitude of different classes. We expect that common class should be larger than smaller class.
- Thought: using weights of samples to determine magnitude of each conflict-reduced class gradient before summing to produce final conflict-reduced gradient?

---
## Paper
- ...

---
## NASA
- https://cs.fit.edu/~pkc/papers/sdm20.pdf for classification representation loss
- `plot_similarity_matrix` compatibility with classification, then document + tutorials
	- Instead use SEP-C (above ln(10), background, and "elevated" (greater than background but below ln(10)))
	- Multiple solar flare classes for SDOBenchmark
\- Order of operations
	- distance matrix
		- regression **(1)** $\checkmark$
			- tabular$\checkmark$
			- image $\checkmark$
		- classification **(2)** $\checkmark$
			- tabular
			- image
	- representation loss $\checkmark$
		- regression **(2)** $\checkmark$
			- tabular $\checkmark$
			- image $\checkmark$
		- classfication **(3)**
			- tabular
			- image