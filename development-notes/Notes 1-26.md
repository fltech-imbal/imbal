## Tasks:
- Instead of `sphinx`, rename folder to `doc` or similar $\checkmark$
	- Such that top level folders are `tutorials`, `src` (package), and `doc` $\checkmark$
		- Also `tutorials/data` $\checkmark$
			- examples within `tutorials` $\checkmark$
			- For now, move data into `tutorials` $\checkmark$
		- Does python have a convention for this? $\checkmark$
			- **Yes. `docs` instead of `doc`** $\checkmark$

**From end of Fall 2025...**
- SEP-C classification should have its own page, similar to the comparison already done $\checkmark$
	- See email from before $\checkmark$

         > I suggest also using SEP-C, and use ln 10 as threshold for the target peak intensity to be Yes (vs No).

	- Classification decoupled/balanced fit should link to both $\checkmark$
- Comparison of methods should be split between autoencoder on/off $\checkmark$
	- For classification, on/off and image/tabular $\checkmark$
- Is there a reason why for regression, decoupled w/ AE does worse than decoupled w/o $\times$
	- Are weights being preserved between extended model and original model? $\times$
	- Are layers being frozen properly? $\times$
	- Structure of extension (is it correct?) $\times$
- `output_label_index` $\rightarrow$ `imbalanced_output_label_index` $\checkmark$
	- *Redundant, as we are getting rid of multi-output* $\checkmark$
- Get rid of `multi_input` and `multi_output` from `decoupled_fit` and `balanced_fit` $\checkmark$
	- For decoupled fit, a separate function needs to be created to handle the second stage balanced fit for AE $\checkmark$
		- *Resolved without separate function*
	- `DatasetWithBatching` should also not have these, but internally have a  `DatasetWithBatching` that supports it for use during AE training $\checkmark$
		- New class, as a subclass of `DatasetWithBatching` to handle the additional outputs needed for AE $\checkmark$
- For `util.generate_ae_branch`, explain branch generation algorithm $\checkmark$
	- Code example for decoder $\times$
	- Before and after of model structure (use `tf.keras.utils.plot_model` for documentation?) $\checkmark$
		- **NOTE: THIS HAD A LINUX PACKAGE DEPENDANCY TO WORK `sudo apt-get install graphviz`**.
	- Add developer parameter for `generate_decoder_branch` to output before and after plots $\times$
## For later...
- *Medium priority:* Refactoring decoupled/balanced fit from functions to wrapping around TF model object
- Documentation on web URL (readthedocs?)
- Make package installable via pip
	- At which point, change `imbal` folder to `src`
## Notes:
- A chance to go back over/clean up/optimize `DatasetWithBatching` code would be nice
- Lots of refactoring will be necessary in the future

**High priority:** functionalities (with AE)  
  
**Medium priority:** re-factoring to make a subclass of Model.  
  
**Low priority:** pip after the implementation is more stable (maybe near  
the end of Spring)


- SEP-C
	- regular
		- common - 7.57619
		- rare -  0.61429
	- Balanced
		- common - 6.21002
		- rare - 1.00909
	- decoupled
		- common - 1.41139
		- rare - 0.99887
	- regular w/ ae
		- common - 3.58386
		- rare - 1.04614
	- balanced w/ ae
		- common - 1.93625
		- rare - 1.03446
	- decoupled w/ ae
		- common - 5.29326
		- rare - 0.71710

![[model.png]]

![[extended-model.png]]
# 1/13/26