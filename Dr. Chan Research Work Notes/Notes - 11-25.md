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


# 11/5/25

## Prep:
- Try and use LIME associated dataset $\checkmark$
	- Read LIME paper - They likely reference the dataset $\checkmark$
	- In general, try to swap in datasets that they used $\checkmark$
- SHAP getting full plot
	- Look at documentation/source
	- May be related to current model outputting `(1, 10)` instead of `(10, 1)`
	
- Get "basics" of plotting to work first (matplotlib for images, HTML for tabular) $\checkmark$
- We perform LIME grayscale to RGB conversion $\checkmark$
- Focus on LIME first $\checkmark$
	- Read up $\checkmark$
- Draw the line for parameters at important/often necessary $\checkmark$

## Notes:
- Went through LIME paper, either the paper fails to mention where they get images from when showing an image, or refer to the images in their figures as "arbitrary images" without further elaboration.
	- Instead used data from this [STL-10](https://cs.stanford.edu/~acoates/stl10/) dataset, which is a small subset of ImageNet, $96\times96$ resolution
- LIME on regression data is not easily adaptable
	- LIME explains a single feature at a time (such as class)
	- For regression, LIME works fine when regression leads to a single numerical output
	- For multiple numerical outputs, a singular feature must be picked to be explained (ex. a model that predict the bounding box of an object in an image can only be explained in terms of one width, height, x position, or y position at a time)
- Should I be concerned with text explanations?
## Tasks:

- Better model? Pretrained or just make my model bigger $\checkmark$
- Pass class names for tabular classification (for wine, low, medium, high quality) $\checkmark$
	- Same for tabular regression $\checkmark$
- Don't worry about LIME image regression or text classification branch yet $\checkmark$
- Pull out code to separate `util` where necessary $\checkmark$
- Modify image classification: $\checkmark$
	- `actual_label` and `label_to_explain` $\checkmark$
		- `actual_label` is required $\checkmark$
		- `label_to_explain` defaults to predicted label from model $\checkmark$
		- Show indexes in title/image labels if string correspondences are not passed $\checkmark$
- Start documenting $\checkmark$
	- Still intro, parameters, then examples $\checkmark$
		- Make sure to refer to lime documentation when applicable (no need to discuss algorithm, time complexity, refer to LIME. Keep it simple. Just wrapping) $\checkmark$.
	- Explain that we do not do text classification (unlikely to be used for space applications, but may be added later on) $\checkmark$
	- Explain that we do not do image regression for LIME because LIME does not support it by default $\checkmark$
	- LIME has its own page on home page $\checkmark$
		- Links to all LIME wrappers $\checkmark$
		- Link to paper, GitHub, etc. $\checkmark$
		- Each function page should like back to the main LIME explanation page as well $\checkmark$
	- For image classification, show example of correct prediction, incorrect prediction, and overridden prediction $\checkmark$
## Notes: 
- Explicit code examples missing (but easy to add)
- Having issue with pyplot representation for tabular classification... not entirely sure why.
## Tasks:
- Explicit code examples missing (but easy to add) !!!
- Having issue with `pyplot` representation for tabular classification... not entirely sure why.
- Correct/incorrect/override examples for tabular classification
	- For regression, close/far/override
- `SHAP`
	- Look at paper
	- Find similar/used datasets
	- image/tabular, classification/regression (where supported)
- `SHAP` issues
	- SHAP getting full plot
		- Look at documentation/source
		- May be related to current model outputting `(1, 10)` instead of `(10, 1)`
- Start implementing `SHAP`, remember easy to use