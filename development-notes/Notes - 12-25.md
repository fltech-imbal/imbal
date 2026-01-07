# 12/3/25
## Tasks:
- See 11/28 email, warning if sample weights and densities are provided for `regression.balanced_fit` $\checkmark$
	- *Note: Have done the same for `class_weights` and `sample_weights` in classification case*
- Make sure ordering for parameters in documentation matches function $\checkmark$
	- class weights before sample weights $\checkmark$
	- sample weights before densities $\checkmark$
- `generate_weights` $\rightarrow$ `generate_sample_weights` $\checkmark$
- `get_densities` $\rightarrow$ `get_sample_densities` $\checkmark$
- make sure docstrings have `class_weights` and `sample_densities`, etc... $\checkmark$
- Row/column issue with MSE... rerun all three and update tables and figures $\checkmark$
- "Frequent" instead of "common" in documentation $\checkmark$
- F1score $\checkmark$
	- Split predictions and labels into one-hot vectors
		- Predictions sum to 1, 1 - prediction to get "confidence" for 0 class
	- Get F1score for each class, report rare and frequent F1
		- [See documentation](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/F1Score)
- AUC
	- Should be done correctly already
		- 1 is rare, 0 is frequent
	- Use toy data if struggling
	- Just confirm
- Begin looking at autoencoder in regression case
	- Look at model structure and build out decoder
	- Use autoencoder for stage 1, then freeze and swap in class

![[Pasted image 20251203122413.png]]

2 branches, loss for each branch (MSE for autoencoder branch)
- Don't worry about interface for now. Functional prototype first, interface/wrapper later.
- Decoupled/balanced fit takes priority