## Tasks:
- Instead of `sphinx`, rename folder to `doc` or similar $\checkmark$
	- Such that top level folders are `tutorials`, `src` (package), and `doc`
		- Also `tutorials/data`
			- examples within `tutorials`
			- For now, move data into `tutorials`
		- Does python have a convention for this?
- Move things around to be pip-installable (by referencing the downloaded repo)
**From end of Fall 2025...**
- SEP-C classification should have its own page, similar to the comparison already done
	- See email from before
	- Classification decoupled/balanced fit should link to both
- Comparison of methods should be split between autoencoder on/off
	- For classification, on/off and image/tabular
- Is there a reason why for regression, decoupled w/ AE does worse than decoupled w/o
	- Are weights being preserved between extended model and original model?
	- Are layers being frozen properly?
	- Structure of extension (is it correct?)
- `output_label_index` $\rightarrow$ `imbalanced_output_label_index`
- Get rid of `multi_input` and `multi_output` from `decoupled_fit` and `balanced_fit`
	- For decoupled fit, a separate function needs to be created to handle the second stage balanced fit for AE
	- `DatasetWithBatching` should also not have these, but internally have a `DatasetWithBatching` that supports it for use during AE training
		- New class, as a subclass of `DatasetWithBatching` to handle the additional outputs needed for AE
- For `util.generate_ae_branch`, explain branch generation algorithm $\checkmark$
	- Code example for decoder
	- Before and after of model structure (use `tf.keras.utils.plot_model` for documentation?)
	- Add developer parameter for `generate_decoder_branch` to output before and after plots
## For later...
- *Medium priority:* Refactoring decoupled/balanced fit from functions to wrapping around TF model object
- Documentation on web URL (readthedocs?)
## Notes:
- A chance to go back over/clean up/optimize `DatasetWithBatching` code would be nice
- Lots of refactoring will be necessary in the future

**High priority:** functionalities (with AE)  
  
**Medium priority:** re-factoring to make a subclass of Model.  
  
**Low priority:** pip after the implementation is more stable (maybe near  
the end of Spring)
# 1/13/26