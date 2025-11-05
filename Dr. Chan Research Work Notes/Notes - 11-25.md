# 11/3/25

## Prep:
- Skip floor(steps/2) in `fit_kde` $\checkmark$
- `plot_kde_1d` images smaller (pull images from docstring and put in markdown) $\checkmark$
- frequent/rare example fix for regression $\checkmark$
	- More than 2 classes for regression $\checkmark$
- Import `LIME` and `SHAP` $\checkmark$
	- Build basic explainer wrappers
	- Use MNIST for examples/prototyping $\checkmark$
		- Classification **AND** regression $\checkmark$
	- Two examples:
		- Image dataset $\checkmark$
		- Tabular dataset $\checkmark$

## Notes:
- Lots to work through here
	- I assume SHAP and LIME are separated in our package
	- At what point do we draw the line between user work and "us" work?
		- Regression and classification is already separate
		- Tabular vs image data... separate functions? **separate**
		- SHAP and LIME have notebook friendly output (HTML, which they say is preferable), but it ONLY works on notebooks
			- Some plotting options are ONLY available via HTML. Can be saved to file though.
			- Opt for always-functional (but not as well formatted output)?
			- Require notebook for output but use better formatted output?
			- Extra parameter to pick?
		- LIME expects to always receive images in an RGB format. We can auto-convert if grayscale is provided, or have user perform the conversion on their end
		- Lots of optional (and frankly, poorly documentation) parameters... where to draw the line?
	- According to what I've seen online, LIME does not work particularly great on MNIST or low-resolution images. Should I consider using some other dataset?
		- In fact, I might need a different tabular dataset as well
- Having issues with SHAP
	- Should I focus on one, then the other?
## Tasks:
- Try and use LIME associated dataset
	- Read LIME paper - They likely reference the dataset
	- In general, try to swap in datasets that they used
- SHAP getting full plot
	- Look at documentation/source
	- May be related to current model outputting `(1, 10)` instead of `(10, 1)`
- Get "basics" of plotting to work first (matplotlib for images, HTML for tabular)
- We perform LIME grayscale to RGB conversion
- Focus on LIME first
	- Read up
- Draw the line for parameters at important/often necessary
