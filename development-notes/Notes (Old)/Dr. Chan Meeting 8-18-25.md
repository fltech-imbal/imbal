Tasks:
- Look at metrics that were not discussed, with reasoning:
	- Critical Success Index
	- Gilbert Skill Score
- Simple TensorFlow implementation (MNIST fashion)
- Linear algebra TensorFlow library, familiarize with "GPU friendly" parallelization
- Implement [[F1]] (assume I have predictions and labels)
	- Build 2 versions, one with loops vs parallelization

# Prep

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

# Meeting Notes

# Tasks
- Expand implementation to include all skill scores
	- Use TensorFlow library
	- AUROC... does it exist in a library? F1?
		- The rest of the scores are more from physics than ML
	- Switch from randomly generating TP, FP, TN, and FN to generating P/N pairs.
- ROC
	- Think about how to do AUROC in parallel
	- how to "sum up list"/count in parallel, at least 2 ways