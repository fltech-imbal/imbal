# 10/6/26

## Thesis

```
======== ideas to explore =======

1.  "tiny" network for distance ratio that outputs only in the
desirable range as discussed ~2 weeks ago. (9/7/26 email)

2.  importance functions as discussed in the email on 9/18.  The
modified sigmoid is not needed since it is solely based on labels, not
the network.
- Can use average, min, or max (start with max) for cutoff on computing cosine similarity

2.  normalization with max distance in hypercircle as discussed above.
- could still be worth weighting triples by distance (but still ignoring those that are very small distance) 
```

- Bad results, but this isn't very different before, since the mode can learn $R=1$. So what is going wrong? $\checkmark$
	- Old ratio loss had $\log$ in loss... maybe that? $\checkmark$
	- If we change the learned ratio loss to simply use $1$ as the learned ratio, do we still have the same issue? $\checkmark$
		- If we remove $\log$ from the old ratio loss, does that create an issue? $\checkmark$
	- It may be better to enforce average loss across all ratios rather than loss of average ratios, as this is more constrained $\checkmark$
	- **Note:** using $(\log(ratio) - \log(desired))^2$ instead of $(ratio - desired)^2$ to account for the fact that $r$ and $1/r$ should produce the same loss value
- Distance difference will have to be modified to support R $\checkmark$
- Reintroduce global anchor to non-hypersphere losses $\checkmark$
	- Probably a rewrite to how I am handling those losses will be required $\checkmark$
	- This will be enabled *guaranteed* from now on $\checkmark$
- For Thursday...
	- Fix above issues, and rerun all methods (with time, compare anchor vs w/o anchor)
	- cosine w/ hypersphere is well

| Representation space | Representation Loss                                                              | Ratio Range Loss (encouraged to be in desirable range) to prevent collapse ($r=0$) or explosion (r=$\inf$) |
| -------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| no hypersphere       | constant distance ratio (ex. entropy, variance) or cosine with similarity $= 1$  | Enabled                                                                                                    |
| w/ hypersphere       | Cosine with derived desirable cosine values (label distance / label range * $3$) | Disabled                                                                                                   |
- Implement "rank-based" approach, for similarity matrix
	- Use stratified sampling or binning to make this work

---

Axes of exploration:
- Latent space unit hypersphere: y/n
- (!!!) Inclusion/excluison of weighting samples inversely with respect to the distance in the label space (futher labels need not have similar representations

**Other thoughts...**
- Might be worth trying to weight samples by `t+6 - t` in the future
- Something for gradient conflicts
	- Think not only of direction, but length
	- More common samples means larger gradient vector
	- Considering magnitude, scaling such that the magnitude of the vectors are of the same length
- Options for weighting cosine representation loss
	- MDI with an appropriate $\alpha$
	- Cosine from 0 to $\frac{\pi}{2}$
	- Normalize distances from 0-1, using $1-x^2$
	- Something else concave?
	- Make sure to include a small epsilon as to not have 0 weights sum to $n$.
	
![[Pasted image 20260917170356.png|500]]
## Paper
- Add perplexity value (150) under the description of t-SNE plot $\checkmark$
## NASA
- After all of above, start updating the documentation and tutorials!
	- Use a simple representation loss
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate

---

# 10/8/26

## Thesis

```
======== ideas to explore =======

1.  "tiny" network for distance ratio that outputs only in the
desirable range as discussed ~2 weeks ago. (9/7/26 email)

2.  importance functions as discussed in the email on 9/18.  The
modified sigmoid is not needed since it is solely based on labels, not
the network.
- Can use average, min, or max (start with max) for cutoff on computing cosine similarity

2.  normalization with max distance in hypercircle as discussed above.
- could still be worth weighting triples by distance (but still ignoring those that are very small distance) 
```

- Compare w/ global anchor vs w/o global anchor $\checkmark$
- May want to re-run all experiments from last month $\checkmark$

| Representation space | Representation Loss                                                              | Ratio Range Loss (encouraged to be in desirable range) to prevent collapse ($r=0$) or explosion (r=$\inf$) |
| -------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| no hypersphere       | constant distance ratio (ex. entropy, variance) or cosine with similarity $= 1$  | Enabled                                                                                                    |
| w/ hypersphere       | Cosine with derived desirable cosine values (label distance / label range * $3$) | Disabled                                                                                                   |
- Implement "rank-based" approach, for similarity matrix $\checkmark$
	- Use stratified sampling or binning to make this work $\checkmark$
		- Will probably do the same for label-based $\checkmark$
- November and December papers will focus on gradient conflict methods $\checkmark$
	- Also, an idea for contrastive loss $\checkmark$

---

**Other thoughts...**
- Might be worth trying to weight samples by `t+6 - t` in the future
- Something for gradient conflicts
	- Think not only of direction, but length
	- More common samples means larger gradient vector
	- Considering magnitude, scaling such that the magnitude of the vectors are of the same length
- Options for weighting cosine representation loss
	- MDI with an appropriate $\alpha$
	- Cosine from 0 to $\frac{\pi}{2}$
	- Normalize distances from 0-1, using $1-x^2$
	- Something else concave?
	- Make sure to include a small epsilon as to not have 0 weights sum to $n$.
	
![[Pasted image 20260917170356.png|500]]
## Paper
- ...
## NASA
- After all of above, start updating the documentation and tutorials!
	- Use a simple representation loss
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate

---

# 10/8/26

## Thesis

```
======== ideas to explore =======

1.  "tiny" network for distance ratio that outputs only in the
desirable range as discussed ~2 weeks ago. (9/7/26 email)

2.  importance functions as discussed in the email on 9/18.  The
modified sigmoid is not needed since it is solely based on labels, not
the network.
- Can use average, min, or max (start with max) for cutoff on computing cosine similarity

2.  normalization with max distance in hypercircle as discussed above.
- could still be worth weighting triples by distance (but still ignoring those that are very small distance) 
```

- November and December papers will focus on gradient conflict methods (!!!)
	- Also, an idea for contrastive loss

---

**Other thoughts...**
- Might be worth trying to weight samples by `t+6 - t` in the future
- Something for gradient conflicts
	- Think not only of direction, but length
	- More common samples means larger gradient vector
	- Considering magnitude, scaling such that the magnitude of the vectors are of the same length
- Options for weighting cosine representation loss
	- MDI with an appropriate $\alpha$
	- Cosine from 0 to $\frac{\pi}{2}$
	- Normalize distances from 0-1, using $1-x^2$
	- Something else concave?
	- Make sure to include a small epsilon as to not have 0 weights sum to $n$.
	
![[Pasted image 20260917170356.png|500]]
## Paper
- ...
## NASA
- After all of above, start updating the documentation and tutorials!
	- Use a simple representation loss
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate


---

# 10/20/26

## Thesis

**Look at the papers, determine differences**

In order to determine if gradient conflicts are reducing, we need a way to measure them. Also, most existing methods reduce conflicts either by performing projections or a weighted sum of the gradients.

| Paper                                                | How they reduce gradient conflict                                                                              | How do they measure gradient conflict                              |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Long-Tailed Learning as Multi-Objective Optimization | Weighted sum of group-based gradients                                                                          | Similarity of class-level and batch-level gradients                |
| Two Fists, One Heart                                 | Subtracting the sum of the projections onto every other gradient                                               | They don't?                                                        |
| TS-MOF                                               | Projection, only looking at conflicting gradients. Sum of differences between original and projected gradient. | Using "conflict probability"                                       |
| Pareto Deep Long-Tailed Recognition                  | Average (original) gradient plus a weighted sum "adjustment" gradient                                          | Non-conflicting vs. conflicting vs. dominated conflicting (Fig. 2) |
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

- November and December papers will focus on gradient conflict methods (!!!)
	- Also, an idea for contrastive loss

---

**Other thoughts...**
- Might be worth trying to weight samples by `t+6 - t` in the future
- Something for gradient conflicts
	- Think not only of direction, but length
	- More common samples means larger gradient vector
	- Considering magnitude, scaling such that the magnitude of the vectors are of the same length
## Paper
- ...
## NASA
- After all of above, start updating the documentation and tutorials!
	- Use a simple representation loss
	- For `plot_similarity_matrix`, focus on **regression** on tabular and image data
		- May need to find something from UC Irvine dataset repo for tabular classification (see ML class paper)
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate

---

# 10/22/26

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

- November and December papers will focus on gradient conflict methods (!!!)
	- Also, an idea for contrastive loss

**Other thoughts...**
- Might be worth trying to weight samples by `t+6 - t` in the future

---
## Paper
- ...

---
## NASA
- After all of above, start updating the documentation and tutorials!
	- Use a simple representation loss
	- For `plot_similarity_matrix`, focus on **regression** on tabular and image data
		- May need to find something from UC Irvine dataset repo for tabular classification (see ML class paper)
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate

---
# 10/27/26

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
### By October 29th
Have some implementation for both grouping/clustering and gradient conflict reduction $\checkmark$
- See images. Draw examples for what might work/not work for possible dimensions $\checkmark$
- Could do some experiments to see if label-based or gradient-direction-based grouping is better $\checkmark$
	- Does grouping by label already produce high-similarity gradients? Compare cosine similarity with the mean within the group $\checkmark$
- Analysis of 3 factors in image (prioritizing factors (1) and (2)) $\checkmark$

---
## Paper
- ...

---
## NASA
- For `plot_similarity_matrix`, focus on **regression** on tabular and image data tutorials
- If possible, introduce table for documentation to indicate the different usages of `validation_split`

| value                       | validation_split                                   |
| --------------------------- | -------------------------------------------------- |
| value >= 2                  | k-fold                                             |
| 0 < value < 1               | %age of training set to be set aside as validation |
| value == 0 or value is None | No validation data                                 |
| default value = 5           |                                                    

- Yet to do classification tutorials for representation loss. Simple classification representation loss?
	- Per class distance to centroids, plus increase distance to centroid
		- https://cs.fit.edu/~pkc/papers/sdm20.pdf
- `plot_similarity_matrix` compatibility with classification, then document + tutorials
	- Instead use SEP-C (above ln(10), background, and "elevated" (greater than background but below ln(10)))
	- Multiple solar flare classes for SDOBenchmark
- Daniel's k-fold validation has not been documented. Should I do that? I may re-write it while I am at it... lots of copy-pasting instead of modularization.

---

# 10/27/26

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
### By November 3rd
- Redo test 2, considering all vectors are the same length (desirable result is a slight movement towards both B and C)
	- With all 3 options from notes during Test 2
- Add way to measure similarity of class-level and the original, non-conflict reduced gradient
- Also, switch to 3 bins (2 classes can't technically conflict with "vanilla" final gradient)
	- Calculate regular final gradient $g$
	- For each bin/class, print $\cos(g, g_i)$
	- Calculate gradient after conflict reduction $g'_i$
	- For each bin/class, print $\cos(g', g_i)$
		- We hope that this will be at least less dissimilar, or positively similar, to all $g_i$
- Thought: using weights of samples to determine magnitude of each conflict-reduced class gradient before summing to produce final conflict-reduced gradient?

---
## Paper
- ...

---
## NASA
- For `plot_similarity_matrix`, focus on **regression** on tabular and image data tutorials
- If possible, introduce table for documentation to indicate the different usages of `validation_split`

| value                       | validation_split                                   |
| --------------------------- | -------------------------------------------------- |
| value >= 2                  | k-fold                                             |
| 0 < value < 1               | %age of training set to be set aside as validation |
| value == 0 or value is None | No validation data                                 |
| default value = 5           |                                                    
**It seems like I was mistaken on the behavior 0 < value < 1 performs 5 validation runs, each run using a random stratified test split of the specified percentage. Is this the behavior we want? (5 seems arbitrary, as well as being not overridable by the user.)**


- Yet to do classification tutorials for representation loss. Simple classification representation loss?
	- Per class distance to centroids, plus increase distance to centroid
		- https://cs.fit.edu/~pkc/papers/sdm20.pdf
- `plot_similarity_matrix` compatibility with classification, then document + tutorials
	- Instead use SEP-C (above ln(10), background, and "elevated" (greater than background but below ln(10)))
	- Multiple solar flare classes for SDOBenchmark
- Daniel's k-fold validation has not been documented. Should I do that? I may re-write it while I am at it... lots of copy-pasting instead of modularization.