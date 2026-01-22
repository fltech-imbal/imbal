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
	- `DatasetWithBatching` should also not have these, but internally have a `DatasetWithBatching` that supports it for use during AE training $\checkmark$
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
- A chance to go back over/clean up/optimize `DatasetWithBatching` and `MultiDatasetWithBatching` code would be nice
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

![[model.png|400]]

![[extended-model.png]]
# 1/13/26
## Tasks:
- Send generated web pages in `.zip` file to Dan $\checkmark$
- For regression documentation, image regression using MNIST and treating labels as continuous values. $\checkmark$
- For classification documentation, SEP-C needs to be redone with $ln(10)$ as peak intensity for positive/negative, generate confusion matrix/TSNE/AUROC $\checkmark$
- Change `decoupled_fit` to `cRT_fit` on classification `rRT_fit` for regression $\checkmark$
- Double check all common layers can be translated in autoencoder $\times$
- *With spare time, medium priority:* Refactoring decoupled/balanced fit from functions to wrapping around TF model object $\times$
## For later...
- Documentation on web URL (readthedocs?)
- Make package installable via pip
	- At which point, change `imbal` folder to `src`
### For Friday...
- Read paper in email
- ON A WEBSITE (obsidian WikiMaker)
	- Title, publication venue, year, link
	- 1 paragraph summary of technique
## Notes:
- A chance to go back over/clean up/optimize `DatasetWithBatching` and `MultiDatasetWithBatching` code would be nice
- Lots of refactoring will be necessary in the future

**High priority:** functionalities (with AE)  
**Medium priority:** re-factoring to make a subclass of Model.  
**Low priority:** pip after the implementation is more stable (maybe near  
the end of Spring)
#### w/o AE
- regular
	- time - 60.96
	- F1 - 0.064
	- AUC - 0.864451476793249
	- epochs - 2000
- balanced 
	- time - 66.11
	- F1 - 0.06349205
	- AUC - 0.739451476793249
	- epochs - 2000
- decoupled
	- time - 91.34
	- F1 - 0.06349205
	- AUC - 0.875
	- epochs - 2000/1000
#### w/ AE
- regular
	- time - 71.71
	- F1 - 0.04799999
	- AUC - 0.12763713080168776
- balanced 
	- time - 69.06
	- F1 - 0.064
	- AUC - 0.7948312236286921
- decoupled
	- time - 106.27
	- F1 - 0.07111111
	- AUC - 0.8433544303797469

## Regression
#### w/o AE
- Regular
	- Time - 85.31
	- frequent - 0.49290
	- rare - 9.38969
- balanced
	- time - 80.95
	- frequent - 1.87644
	- rare - 2.39045
- decoupled
	- time - 111.84
	- frequent - 1.65914
	- rare - 3.64001
#### w/ AE
- Regular
	- Time - 220.24
	- frequent - 0.24541
	- rare - 4.76394
- balanced
	- time - 235.87
	- frequent - 0.60429
	- rare - 2.69737
- decoupled
	- time - 268.00
	- frequent - 0.24228
	- rare - 4.82185

# 1/15/26
## Tasks:
- For classification, no KDE curve on data distribution plot $\checkmark$
- Use log-scale for y axis to make sparse data more noticeable $\checkmark$
- Are the F1 scores in documentation consistent with confusion matrix (remember TF expected shape for F1) $\checkmark$
	- Need to re-run all test, get new F1Scores, AUCs, plots $\checkmark$
- In local KDE approximation, tests for unsigned Numpy and Keras types for labels $\checkmark$
	- For each bit size $\checkmark$
	- cast to respective size $\checkmark$
	- Throw warning if negative values appear when converting from unsigned to signed $\checkmark$
		- In comment for warning, include examples that can cause the problem (!!!) $\checkmark$
- In MNIST regression documentation, swap "Class 0/9" for "Digit 0/9" $\checkmark$
- Instead of linear proportions for MNIST, exponential (halving each time) $\checkmark$
	- Need to re-run tests, get new stats/plots $\checkmark$
- https://ibug.doc.ic.ac.uk/resources/agedb/ Obtain and use `AgeDB` for image regression documentation $\checkmark$
	- Reached out, waiting for response $\cdots$
	- If necessary, downsample / stratified sample to reduce dataset size (if resolution is too high) $\cdots$
- redo default layer for cRT/rRT (see email) $\checkmark$
	- `Besides the default is the second last trainable layer, for the examples, show one can choose another layer such as the third last trainable layer, and compare the performance with default.` $\checkmark$
		- Compare performance... in what manner?
- Double check all common layers can be translated in autoencoder $\times$
- *With spare time, medium priority:* Refactoring decoupled/balanced fit from functions to wrapping around TF model object $\times$

## For later...
- Documentation on web URL (readthedocs?)
- Make package installable via pip
	- At which point, change `imbal` folder to `src`

## Notes:
- A chance to go back over/clean up/optimize `DatasetWithBatching` and `MultiDatasetWithBatching` code would be nice
- Lots of refactoring will be necessary in the future

**High priority:** functionalities (with AE)  
**Medium priority:** re-factoring to make a subclass of Model.  
**Low priority:** pip after the implementation is more stable (maybe near  
the end of Spring)

## Tabular Classification
### w/o AE
regular
- time - 38.36
- f1 - 0.0
- auc - 0.883
balanced
- time - 41.34
- f1 - 0.5
- auc - 0.537
decoupled
- time - 57.93
- f1 - 0.625
- auc - 0.858
### w/ AE
regular
- time - 49.71
- f1 - 0.0
- auc - 0.094
balanced
- time - 51.31
- f1 - 0.625
- auc - 0.832
decoupled
 - time - 72.42
 - f1 - 0.0
 - auc - 0.866

## MNIST
#### w/o AE
regular
- time - 74.84
- frequent - 0.11985
- rare - 9.41916
balanced
- time - 76.61
- frequent - 0.97934
- rare - 5.09938
decoupled
- time - 91.47
- frequent - 1.17429
- rare - 3.90605
#### w/ AE
regular
- time - 116.24
- frequent - 0.13282
- rare - 9.89732
balanced
- time - 127.20
- frequent - 0.84396
- rare - 4.42349
decoupled
- time - 145.43
- frequent - 0.11277
- rare - 10.02015

## Tasks
- For classification, no KDE curve on data distribution plot $\checkmark$
	- Use log-scale for y axis to make sparse data more noticeable $\checkmark$
- In documentation, `decoupled_fit` to `cRT_fit`/`rRT_fit` $\checkmark$
- `Besides the default is the second last trainable layer, for the examples, show one can choose another layer such as the third last trainable layer, and compare the performance with default.`
	- Image/tabular classification/regression for cRT/rRT fit w/o AE, comparing second and third to last layer as representation layer.
	- Simplify models to do this
- Refactoring decoupled/balanced fit from functions to wrapping around TF model object $\checkmark$
	- Started, a LOT of questions (see below)
- Make sure names/code examples are consistent throughout documentation $\times$
- `representation_layer_index` over `latent_layer_index` for TSNE $\checkmark$
- `AgeDB` access as of this morning $\checkmark$
## Notes:
- A chance to go back over/clean up/optimize `DatasetWithBatching` and `MultiDatasetWithBatching` code would be nice
- Lots of refactoring will be necessary in the future

**High priority:** functionalities (with AE)  
**Medium priority:** re-factoring to make a subclass of Model.  
**Low priority:** pip after the implementation is more stable (maybe near  
the end of Spring)

## Questions about TF model refactoring
- Batch stratification, decoder branch generation, representation layer index... should these be set in `fit`/`balanced_fit`/`cRT_fit`/`rRT_fit`, or set with setting functions (i.e. `set_batch_stratification(True)`) 
	- Separating into individual flags would likely make things easier from an implementation perspective (otherwise a lot of if-else within the fit functions for handling this, when some of these things can be handled before fit is called)
	- Alternatively, we could extend `model.compile` to receive these True/False arguments?
		- Could be beneficial, as certain things, such as generating the decoder branch, require the model to be recompiled anyways. We could prevent recompilation by generating the branch before the model is compiled the first time.
- When going through documentation, should `SimpleDataset` be scrapped? It seems redundant at this point