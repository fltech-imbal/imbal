## Tasks
- HSS and F1 for final print in classification (no accuracies) $\checkmark$
	- Use following wording for regression print: $\checkmark$
	- Testing only (remove testing metrics) $\checkmark$
```
MAE for log10 flux >= -4 :
MAE for log10 flux < -4 :
```
- Add supplemental information for balanced/rRT regression: $\checkmark$
	- How to manually generate weights with handpicked alpha using `imbal`, then passing sample weights to `balanced_fit/rRT_fit`. $\checkmark$
	- Passing custom class weights for balanced/cRT classification $\checkmark$
- Make sure section numbers are consistent in tutorials $\checkmark$
- For tutorials (3) in email, AE is *off*. $\checkmark$
 1.  each tutorial of the 9 tutorials has one source code file -- plus  
usage of class weights and sample weights with alpha are commented out.  $\checkmark$
  
2.  importing libraries, loading data, and defining model structure are  
identical so they are going to be discussed on a separate page, which is  
linked from the 9 tutorials.  Please make sure the instructions in the  
source code are identical; if not, move them down to near compile() or  
further down.  $\checkmark$
  
custom sample_weights and class_weights should be, in my opinion, right  
before fit() and then used by fit().  $\checkmark$
  
3.  the rest: compile(), fit(), evaluate(), predict(), results, and  
plotting could be different in the 9 tutorials, so they are not in a  
common separate webpage.  $\checkmark$
  
Additionally, to make the tutorials easier to read, we are going to  
support basic plotting of predicted vs actual (regression) and confusion  
matrix (classification).  That is, one instruction to plot each of them.  $\checkmark$
  
1.  plot_confusion_matrix(actual_values, predicted_values)  $\checkmark$
2.  plot_actual_vs_predicted_values(actual_values, predicted_values)  
     with actual on x, predicted on y, and a diagonal dotted line  $\checkmark$


---
# 4/3/26

## Tasks:
- Do validation tutorials. $\checkmark$
	- Use best result to begin explanation and visualization tutorials (only one, using best method) LIME and t-SNE
	- Use rare instance for LIME tutorials
- For LIME implementations $\checkmark$
	- Set image `num_features` back to 5 $\checkmark$
	- Why was training data needed as parameter? $\checkmark$
		- Required by `lime_tabular.LimeTabularExplainer`, however the parameter is almost entirely undocumented in original LIME documentation. $\checkmark$
- For regression plot $\checkmark$
	- don't need to worry about different rare/frequent colors $\checkmark$
	- Only diagonal line (no rare bound) $\checkmark$
	- Perhaps some optional parameters with default values: actual_axis_label, predicted_axis_label, actual_range, predicted_range, shape, color, size $\checkmark$
	- Mention in documentation that points are plotted in the same order that they are supplied $\checkmark$
- Optional section 2.1s should be changed to section 3 (density vs weights) $\checkmark$
- Show results for handpicking class weights / alpha $\checkmark$
	- Add section to bottom, showing where code was changed, and new results $\checkmark$
1.  On the main tutorial page, for the 9 tutorials, instead of a list, I  
suggest a 3x3 table to make the organization easier to understand:  $\checkmark$
     a.  Column headings: regular training, balanced training, cRT/rRT  
(classifier/regressor re-training)  $\checkmark$
     b.  Row headings: basic, with autoencoder to enhance representation  
learning, with validation set to reduce overfitting and estimate  $\checkmark$
class/sample weights  
    c.  Cell: regular, balanced, cRT/rRT $\checkmark$
              regular+AE, ...  $\checkmark$
              regular+val, ...  $\checkmark$
2.  Between the 9 tutorials and t-SNE, add a tutorial on metrics  
      classification: F1, HSS, TSS, AUROC  
      regression: MAE, MSE, Correlation

---
## 4/7/26
## Tasks
- Add `plot_roc` $\checkmark$
	- Documentation $\checkmark$
- Follow up with Karen about "TA teaching" class $\checkmark$
- Make sure to register for Fall classes $\checkmark$
- Make sure percent signs show for LaTeX in documentation $\checkmark$
- Make sure new documentation numbers are in-order $\checkmark$
- Make sure new section 3 description for documentation is on all validation tutorials $\checkmark$
- Get rid of "callback" explanation in tutorials in favor of more general reasoning for why we include validation data $\checkmark$
- For validation tutorials, add additional supplement for specifying validation percent
- [Keras GradCam](https://keras.io/examples/vision/grad_cam/) how easily can we wrap this and implement in imbal?
	- Make sure to mention code is taken from that link in documentation
	- Also refer to paper link from paper list
- When making metric tutorial, considerations for putting metrics in/out of compile:
	- When in compile, metrics are tracked every batch/epoch (can be time consuming)
		- If you don't care what these metrics are during training, they should be left out
		- If the metric itself is already time consuming (AUROC), it should probably be left out of the compile
- Maybe profile memory usage to see what might be causing OOM issue?
2.  Between the 9 tutorials and t-SNE, add a tutorial on metrics  
      classification: F1, HSS, TSS, AUROC  
      regression: MAE, MSE, Correlation
#### Top Priority:
- Make sure multi-weight lists are support in `imbal.classification/regression.split` $\checkmark$
- Fix `imbal.Model` to take in multiple lists of weights for validation data
	- sample weights and validation weights should have same number of rows
	- Potentially, one "data cleaning" function to ensure data is always in a singular format, followed by the necessary calls (and loops for `multi_weight`)
	- Potentially, everything passes though `multi_weight` (with singular weight lists being reshaped to $(1, N)$)
	- For class weights, training and validation data can be class weighted individually (differences in distribution is minimal for larger datasets).
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

---
# 4/10/26
## Tasks
#### Top Priority:
- Make sure multi-weight lists are support in `imbal.classification/regression.split` $\checkmark$
- Fix `imbal.Model` to take in multiple lists of weights for validation data $\checkmark$
	- We are no longer supporting `PyDataset` or `Dataset` to be passed to our fit functions. $\checkmark$
		- Move `DatasetWithBatching` to `imbal.util.backend` $\checkmark$
	- sample weights and validation weights should have same number of rows $\checkmark$
	- Potentially, one "data cleaning" function to ensure data is always in a singular format, followed by the necessary calls (and loops for `multi_weight`) $\checkmark$
	- Potentially, everything passes though `multi_weight` (with singular weight lists being reshaped to $(1, N)$) $\checkmark$
	- For class weights, training and validation data can be class weighted individually (differences in distribution is minimal for larger datasets). $\checkmark$
#### Low Priority:
- For validation tutorials, add additional supplement for specifying validation percent
- When making metric tutorial, considerations for putting metrics in/out of compile:
	- When in compile, metrics are tracked every batch/epoch (can be time consuming)
		- If you don't care what these metrics are during training, they should be left out
		- If the metric itself is already time consuming (AUROC), it should probably be left out of the compile
- Maybe profile memory usage to see what might be causing OOM issue?
2.  Between the 9 tutorials and t-SNE, add a tutorial on metrics
      classification: F1, HSS, TSS, AUROC  (HSS inside compile, F1 outside)
      regression: MAE, MSE, Correlation
  - [KernelExplainer](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html#shap.KernelExplainer) vs general [Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer) $\checkmark$
  - [Keras GradCam](https://keras.io/examples/vision/grad_cam/) how easily can we wrap this and implement in imbal?
	- Make sure to mention code is taken from that link in documentation
	- Also refer to paper link from paper list
  - Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability
#### Miscellaneous:
- Fixed multiple LIME bugs for Daniel (LIME by default does not work on binary classification. It must be converted to a 2-class classification, where the first class is false, and the second class is true) $\checkmark$
- Fixed a bug related to using model with AE when no metric are provided $\checkmark$

---
# 4/14/26

## Tasks
#### Top Priority:
- Remove MSE metric from AE $\checkmark$
- Make sure documentation specifies that the we the Kernel SHAP (and link to paper if not linked already) $\checkmark$
	- For images, I have previously used SHAP's gradient explainer. Based on what I could find, `GradientExplainer` is faster for neural networks, but only works on neural networks and is therefore not truly model agnostic (like kernel SHAP) $\checkmark$
	- Is this an issue? Should I switch to kernel for everything? Should I maybe use `GradientExplainer` for everything? $\checkmark$
- LIME bug from Daniel? $\checkmark$
- Maybe profile memory usage to see what might be causing OOM issue? $\checkmark$
- Re-run all previous tutorials $\checkmark$
- For validation tutorials, add additional supplement for specifying validation percent $\checkmark$
- When making metric tutorial, considerations for putting metrics in/out of compile: $\checkmark$
	- When in compile, metrics are tracked every batch/epoch (can be time consuming) $\checkmark$
		- If you don't care what these metrics are during training, they should be left out $\checkmark$
		- If the metric itself is already time consuming (AUROC), it should probably be left out of the compile $\checkmark$
		- We recommend 0 or 1 metric $\checkmark$
			- classification: F1, HSS, TSS, AUROC  (HSS inside compile, F1 outside) $\checkmark$
				- Add GSS, J statistic, and Youden's index $\checkmark$
			- regression: MAE, MSE, Correlation $\checkmark$
				- For regression/single value metric, split into overall, frequent, and rare $\checkmark$
#### Low Priority:
- [Keras GradCam](https://keras.io/examples/vision/grad_cam/) how easily can we wrap this and implement in imbal?
	- Make sure it works for binary classification with single output unit (convert to 2 class)
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

# 4/21/26
## Tasks
#### High Priority
- Get rid of "\[3/3]" for multi-fit supplements $\checkmark$
- Add plots to metric tutorials (true v predicted, confusion matrix, AUROC) $\checkmark$
- Change "multi-weight fit" to "exploring class/sample weight candidates" $\checkmark$
- visualization tutorials  $\checkmark$
	- Also refer to paper link from paper list $\checkmark$
	- For LIME classification tutorial $\checkmark$
		- Have two examples:  $\checkmark$
			- Explaining an event that was detected correctly  $\checkmark$
			- A mistake (false negative or false positive) $\checkmark$
		- Focus on True Positive/False Negative $\checkmark$
			- Maybe False Positive, but True Negatives are too frequent  $\checkmark$
	- SHAP
	- TSNE $\checkmark$
- Replace `GradientExplainer` with `KernelExplainer`, because model-agnostic $\checkmark$
- Explain what "weight candidate index" means in output for documentation, help the user interpret what that means (refer to section where alphas are specified)
- Documentation for multi-weight fit should include that best weights/class weights and index are saved in the model object in particular fields.
- Make sure class weights are printed for multi-weighted fit on classification side (the stuff is already there to make it work, it just isn't happening)

#### Low Priority:
- AUROC plots for classification tutorials
- [Keras GradCam](https://keras.io/examples/vision/grad_cam/) how easily can we wrap this and implement in imbal?
	- Make sure it works for binary classification with single output unit (convert to 2 class)
	- Make sure to mention code is taken from that link in documentation
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability