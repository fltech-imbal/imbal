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
- Distance difference will have to be modified to support R
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
- Add perplexity value (150) under the description of t-SNE plot
## NASA
- After all of above, start updating the documentation and tutorials!
	- Use a simple representation loss
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate