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
- Handle all bullet points in 8/4/26 email
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