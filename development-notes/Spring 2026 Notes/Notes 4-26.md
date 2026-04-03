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
	- Passing custom class weights for balanced/cRT classification
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
     Perhaps some optional parameters with default values:  
actual_axis_label, predicted_axis_label, actual_range, predicted_range,  
shape, color, size  **TBD**

**Redid all current regression + regular classification**