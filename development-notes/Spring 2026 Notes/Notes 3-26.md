## Tasks:
- For testing weights over folds:
	- Print out: All weights, average epoch, average metric used to determine best
	- In my case: $MAE, MAE_R, AORE, \alpha, \text{val\_loss}$
	- AORE uses ABSOLUTE mean error
	- Use AORE over `val_loss` for determining weights for both w/ and w/o CME data. Re-generate tables, save print outs across all alphas
- Allow fits to take in multiple lists of weights
	- Use same stopping criteria provided for user for validation across weights (finding best weights)
	- Refactoring when possible!!
- Outside `imbal`,  add variation of having an extra trained layers for decoupled fit $\times$
	- For tables, add two rows (since above always has -2 rep layer for first stage, then add additional layer in second stage)
- Inside `imbal`:
	- update balanced/decoupled fit to incorporate finding the "best" class/sample weights based on the validation set
		1.  Classification, iterate on different lists of class weights
		2.  Regression, iterate on different lists of sample weights via alpha for reciprocal importance
- Fit functions will run much slower, I suggest printing status messages. I suggest an int parameter to indicate message levels, this will also help debugging, for example: 
	 1. no messages (except those from `keras`/`tensorflow`)
	 2. main steps, found epoch number based on validation, class weights (alpha in reciprocal importance) based on validation..., training on the entire training set
	 3. Different class weights, alphas, ... being evaluated
- **Later on:** MDI, wPCC
- **30%:** Refactoring / rewrite of `generate_decoder_branch`