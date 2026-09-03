# 8/3/26

- Redraw / re-screenshot figures to make sure they are clear (there are a little blurry now) $\checkmark$
- In `3.3.1 Decoder Branch Generation`, add actual code segment which indicates how to enable the decoder branch. $\checkmark$
	- **Can be done inline** $\checkmark$
	- Do something similar for mentions of `class_weight` and `sample_weight` (or anything else that is a parameter rather than a function itself) $\checkmark$
- For `representation_layer_index`, specify what values such as `-1` and `-2` means $\checkmark$
- In `3.4 Explanation of predictions` include a LIME and `GradCam` example from the tutorials. $\checkmark$

- List all metrics that the tool has available in `3.5 Evaluation metrics` $\checkmark$
- Combine tables 2 and 3 by doing following columns in single table: category, range of intensity, number of time series in training, instance in training set, percentage of training set, number of test time series, instances in test set, percentage of test set. $\checkmark$
	- Can say "event" instead of time series $\checkmark$
	- **How should I handle explaining the amount of series?**
* Include information about number of time series/events used to make training/validation/test sets (to be clear that time series are not split across training/validation/test) $\checkmark$
* For results
	* Get results using 5 runs averaged, using median run for best result.
	* actual vs predicted (with each time series as a different color)
	* Later on, plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
		- Plot series that are highly over/underestimating (based on scatter plot) (2)
		- Also plot two that are quite accurate
	- SHAP will be used for explanations in section `4 SEP Forecasting tasks`

---
# 8/5/26
- Add the loss function and optimizer used + learning rate and maximum epoch number, early stopping $\checkmark$
- Handle all bullet points in 8/4/26 email
* For results
	* Get results using 5 runs averaged, using median run for best result.
	* actual vs predicted (with each time series as a different color)
	* In section 3.3, add reference to section 3.2.1 when discussing balanced fit.
	* For t-SNE description, include the purpose is to compress the representation space to 2D
	* Remove code examples.
	* Limitation of decoder branch generation is it assumed the original mode is sequential (containing no branches).
	* Later on, plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
		- Plot series that are highly over/underestimating (based on scatter plot) (2)
		- Also plot two that are quite accurate
	- SHAP will be used for explanations in section `4 SEP Forecasting tasks`

---
# 8/7/26
- Make sure model weights are reset after calculating $\lambda$ $\checkmark$
- Try 50-100 epoch average for calculating $\lambda$ $\checkmark$
	- In fact, second half of how long it ran for $\checkmark$
- No need to have sample weights for calculating $\lambda$ because sample weight cancels out in top an bottom of ratio $\checkmark$
- Handle all bullet points in 8/4/26 email $\checkmark$
- Make the above changes, wait for Daniel's changes, and **then** to average of 5 runs $\checkmark$
* For results
	* Get results using 5 runs averaged, using median run for best result.
	* actual vs predicted (with each time series as a different color)
	* In section 3.3, add reference to section 3.2.1 when discussing balanced fit.
	* For t-SNE description, include the purpose is to compress the representation space to 2D
	* Remove code examples.
	* Limitation of decoder branch generation is it assumed the original mode is sequential (containing no branches).
	* Later on, plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
		- Plot series that are highly over/underestimating (based on scatter plot) (2)
		- Also plot two that are quite accurate
	- SHAP will be used for explanations in section `4 SEP Forecasting tasks`

---
# 8/10/26
- Each fit has 4 configs: reg/bal/dec, +val, +ae, +ae-3 $\checkmark$
	- Alpha is not specified by default $\checkmark$
- In the table, report... $\checkmark$
	- The config (described above) $\checkmark$
	- the alpha of the median model (if applicable) $\checkmark$
- **While doing runs with balanced/decoupled, check intermediate results to make sure things are performing *as expected* (better than regular, at least similar to each other)** $\checkmark$
* For results $\checkmark$
	* Get results using 5 runs averaged, using median run for best result. $\checkmark$
	* actual vs predicted (with each time series as a different color)
	* In section 3.3, add reference to section 3.2.1 when discussing balanced fit.
	* For t-SNE description, include the purpose is to compress the representation space to 2D
	* Remove code examples.
	* Limitation of decoder branch generation is it assumed the original mode is sequential (containing no branches).
	* Later on, plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
		- Plot series that are highly over/underestimating (based on scatter plot) (2)
		- Also plot two that are quite accurate
	- SHAP will be used for explanations in section `4 SEP Forecasting tasks`

# 8/12/26
- For baseline, train to convergence rather than to arbitrary epoch count $\checkmark$
- In paper, P16.4 is the target channel, t+6 is the target
- Rerun DTW, then log-normalize new dataset and re-run everything $\checkmark$
- Update information about the dataset

- For results $\checkmark$
	* Get results using 5 runs averaged, using median run for best result. $\checkmark$
	* actual vs predicted (with each time series as a different color)
	* Later on, plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
		* Pick time series that are furthest from the line for for further analysis
			* See blue comment in overleaf
		- Plot series that are highly over/underestimating (based on scatter plot) (2)
		- Also plot two that are quite accurate
	- SHAP will be used for explanations in section `4 SEP Forecasting tasks`

# 8/18/26
## Paper
- In paper, P16.4 is the target channel, t+6 is the target $\checkmark$
- Update information about the dataset $\checkmark$

- For results $\checkmark$
	* Get results using 5 runs averaged, using median run for best result. $\checkmark$
	* actual vs predicted (with each time series as a different color) $\checkmark$
	* Later on, plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
		* Pick time series that are furthest from the line for for further analysis
			* See blue comment in overleaf
		- Plot series that are highly over/underestimating (based on scatter plot) (2)
		- Also plot two that are quite accurate
	- SHAP will be used for explanations in section `4 SEP Forecasting tasks
## Research
- How to incorporate representation loss?
	- Multiple experts?
	- Gradient conflicts?
- Planning to present a paper in a month-ish
## Thesis
- What could be a representation loss that encourages ordering in the trace but not constant ratio?  Note that each batch is sorted from stratified sampling.

---
# 8/20/26
## Paper
- Blue for "low error", pink for high
- Stratified sample for passing to SHAP
	- Add to documentation that SHAP has trouble with large datasets, and to use stratified split to "sub-sample" the dataset
* Later on, plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$)
	- Pick time series that are furthest from the line for for further analysis
		- See blue comment in overleaf
	- Plot series that are highly over/underestimating (based on scatter plot) (2)
	- Also plot two that are quite accurate
- SHAP will be used for explanations in section `4 SEP Forecasting tasks
## Research
- Work on representation loss usage $\checkmark$
	- Test with one (and more) of my representation losses

---
# 8/25/26

## Thesis
- Something for gradient conflicts
	- Think not only of direction, but length
	- More common samples means larger gradient vector
	- Considering magnitude, scaling such that the magnitude of the vectors are of the same length
## Paper
- Blue for "low error", pink for high $\checkmark$
- Stratified sample for passing to SHAP $\checkmark$
	- Add to documentation that SHAP has trouble with large datasets, and to use stratified split to "sub-sample" the dataset 
* Later on, plots like Torres 2025 paper, including individual time series on scatter plot and like Figure 3 true vs. predicted over time, and different channels (mind that predictions are at time $t+6$) $\checkmark$
	- Pick time series that are furthest from the line for for further analysis $\checkmark$
		- See blue comment in overleaf $\checkmark$
	- Plot series that are highly over/underestimating (based on scatter plot) (2) $\checkmark$
	- Also plot two that are quite accurate $\checkmark$
- SHAP will be used for explanations in section `4 SEP Forecasting tasks
## NASA
- Address bug with representation loss branch $\checkmark$
- Use toy dataset for ensuring representation losses are working properly

----
# 8/27/26

## Thesis
- Something for gradient conflicts
	- Think not only of direction, but length
	- More common samples means larger gradient vector
	- Considering magnitude, scaling such that the magnitude of the vectors are of the same length
## Paper
-  Add to documentation that SHAP has trouble with large datasets, and to use stratified split to "sub-sample" the dataset $\checkmark$
	- Other options for SHAP $\checkmark$
- For time series plots $\checkmark$
	- Electrons can be lighter, "grayed-out" colors $\checkmark$
	- Actual proton is red, other protons are different colors (green, blue) $\checkmark$
		- Also, lighter/lower transparency for non-predicted channels $\checkmark$
	- Prediction is a dashed red line (maybe dark red) $\checkmark$
	- Actual and predicted are thicker (3-4?) vs other channels (2) $\checkmark$
	- Crop the time series to focus on the rising edge (get rid of 24 hour padding) $\checkmark$
	- Add date to x-axis, as well as labels for all axis $\checkmark$
	- Once happy with the time series plots, superimpose a circle on the pink plot that corresponds to the area of high error on the true vs predicted scatter plot $\checkmark$
- SHAP will be used for explanations in section `4 SEP Forecasting tasks
## NASA
- Use toy dataset for ensuring representation losses are working properly $\checkmark$
- SHAP can have some extra parameters for how many features to display, extra padding for the left side of the graph? Investigate

---
