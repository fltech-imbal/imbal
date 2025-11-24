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

# 11/7/25
## Notes: 
- Explicit code examples missing (but easy to add)
- Having issue with pyplot representation for tabular classification... not entirely sure why.
## Tasks:
- Explicit code examples missing (but easy to add) !!!
- Having issue with `pyplot` representation for tabular classification... not entirely sure why.
- Correct/incorrect/override examples for tabular classification
	- For regression, close/far/override
- `SHAP` $\checkmark$
	- Look at paper $\checkmark$
	- Find similar/used datasets $\checkmark$
	- image/tabular, classification/regression (where supported)
- `SHAP` issues
	- SHAP getting full plot
		- Look at documentation/source
		- May be related to current model outputting `(1, 10)` instead of `(10, 1)`
- Start implementing `SHAP`, remember easy to use

# 11/10/25

## Prep:
 - Explicit code examples missing (but easy to add) !!! $\checkmark$
- Having issue with `pyplot` representation for tabular classification... not entirely sure why. $\checkmark$
	- LIME documentation has no examples where they shoe pyplots for tabular data... it might be best to just scrap it in favor of HTML. All LIME code examples show HTML output  
- Correct/incorrect/override examples for tabular classification $\checkmark$
	- For regression, close/far/override $\checkmark$
- `SHAP` $\checkmark$
	- Look at paper $\checkmark$
	- Find similar/used datasets $\checkmark$
	- image/tabular, classification/regression (where supported) $\checkmark$
	- **Note:** Image classification, tabular classification, and tabular regression are supported
- `SHAP` issues 
	- SHAP getting full plot $\checkmark$
		- Look at documentation/source $\checkmark$
		- May be related to current model outputting `(1, 10)` instead of `(10, 1)` $\checkmark$
- Start implementing `SHAP`, remember easy to use $\checkmark$

## Tasks:
- `LIME` $\checkmark$
	- For regression override, explain that predicted label doesn't update, but explanation updates to explain passed label. $\checkmark$
- `SHAP` $\checkmark$
	- Can you increase resolution? Should be able to, but how? $\checkmark$
		- **Note:** Have results, but they don't look good. Might need to make model that outputs logits, or just need to do some more testing to ensure I'm using it right
	- Add titles, labels and such to plots $\checkmark$
	- Allow for most plot types, even multi-sample plots $\checkmark$
	- Include any appropriate/intuitive plots $\checkmark$
		- **Note:** Decided to exclude `scatter` plot, as it either produces an unwieldy plot, or requires an extra parameter that other plots would not require
![[Pasted image 20251112073805.png]]
	- Stick to explaining one class at a time (not plotting all classes as with images) $\checkmark$
- Rename functions to reflect singular sample vs across dataset $\checkmark$
- Note: dataset documentation examples don't have the correct/incorrect/override
	- BUT include all three scenarios for all per-sample explanation documentation
## Tasks:
- `SHAP` figure out and finalize wrapper for image classification $\checkmark$
- Documentation $\checkmark$
	- Image classification SHAP documentation needs to be completely done $\times$
	- Code and plot examples for tabular regression $\checkmark$
- Remove LIME `pyplot` code for tabular data $\checkmark$
- LIME add title to HTML plots by modifying HTML object $\checkmark$
- For dataset plots, use same label for all 3 plots (so you can compare the methods) (probably region 2/3) $\checkmark$
- Add paragraph to SHAP page explaining that in addition to sample explanations, SHAP can do "dataset explanations" $\checkmark$
- Read paper section $2.3$ $\checkmark$

# 11/14/25

## Notes:
- After further inspection, SHAP does not have an "override" for regression values
	- Override for classification is based on the fact that a confidence value is produced for each class, so filtering the confidences for a particular class can generate an explanation for that class for any sample
	- Regression only outputs a single value, hence no "override" for SHAP
	- Looking into LIME (and again, their documentation is very sparse) I'm no longer confident that the regression override for LIME is doing exactly as we expect either (I ran some test, passing the exact label the model predicts as the label to explain produces different results than passing no label to explain, which is also supposed to default to using the models prediction)
		- Should override be removed from regression sample explanations altogether?
- SHAP image explanation is being a hassle
	- It differentiates between Python lists and numpy arrays in the worst possible way... still trying to understand how to replicate MNSIT example, but getting closer
		- Essentially, SHAP interprets `[np.array([1, 2, 3])]` differently from `np.array([[1, 2, 3]])`

# 11/17/25
## Tasks:
- Complete SHAP image classification documentation $\checkmark$
- Change SHAP page from 'one important difference' to "one difference" $\checkmark$
- In my documentation for SHAP wrappers, explain that all list-like data should be passed as numpy arrays $\checkmark$
	- Make sure to raise an exception for when non-numpy arrays are passed $\checkmark$
- Prefer to use STL-10 for SHAP image classification examples in documentation, but if results stay bad, use MNIST $\checkmark$
- Remove override examples for SHAP tabular regression (it does not allow for it) $\checkmark$
	- Include a note explaining this in documentation $\checkmark$
- Think about how to implement a `decoupled_fit` wrapper (Figure $1$, section $2.3$) $\checkmark$
	- Consider latent space is second to last layer $\checkmark$
	- Call `tf.model.fit` twice $\checkmark$
- Think about AED inclusion
	- Implemented with flag, default to `True`
	- "Invert" the model passed by the user
	- **WRAPPER FIRST, then this AED branch** $\checkmark$ 

## Notes:
- After some tinkering with TensorFlow, I've come to the realization that by default, TensorFlow weights samples with a weight of `1` per sample when no weights are provided, not `1/n` as our previous methods have implemented
	- The difference in weights was very noticeable in training losses (sharp decrease in loss from low "under-weighting")
	- I have made the necessary changes to stay consistent with this, but can revert if necessary
- Also went back and rewrote previous code for classification to support one-hot vector labels on top of integer labels
- Review parameters for `decoupled_fit`
	- For `compile_function`, include `stage` parameter?

# 11/19/25

## Tasks:
- Add which explainer is used for all SHAP documentation (first part) $\checkmark$
	- Anything else that might be relevant $\checkmark$
- Change code that looks for representation layer to by default search for *last layer with parameters* $\checkmark$
	- For when layer is specified, specified index should be less than or equal to index of last layer with parameter $\checkmark$
		- If not, warn and override $\checkmark$
	- Extract logic into `util` function, also add to TSNE $\checkmark$
	- Consideration for layer: Representation layer could be in separate branches? $\checkmark$
- Change documentation of generated weights and dataset with batching to say weights sum to n, and this is what TensorFlow expects $\checkmark$
	- Also code examples $\checkmark$
- Can I decouple weight logic in generate weights and `DatasetWithBatching` to separate function? $\checkmark$ (no) 
- Notes for one-hot vectors in documentation for weights and batching $\checkmark$
- (See photos) Wrapper function to store compile parameters $\checkmark$
	- Return an object which we create, and **document it** $\checkmark$
- Allow for different compile parameters for each stage $\checkmark$
	- By default, assume stages are the same $\checkmark$
- Epochs should be `int` or `tuple` $\checkmark$

## Notes:
- `Consideration for layer: Representation layer could be in separate branches?`
	- Layers are always indexable, but it is a little hard to detect a branch on our end...
# 11/21/25

## Prep:
- Substitute in layer finding for TSNE (see previous notes) $\checkmark$
- Add ICLR 2020 after Kang et al. in documentation for `decoupled_fit` $\checkmark$
- Reword `compile_parameters`, `stage_one_compile_parameters`, and `stage_two...` $\checkmark$
	- Stage one and two refer to `compile_parameters` if not specified, `compile_parameters` refer to TF defaults if not specified $\checkmark$
	- Make sure to mention TF's `model.compile()` defaults, which are used with `compile_parameters` is `None` $\checkmark$
- For appropriate `model.fit` parameters, note (Same as `model.fit()`) (and link) $\checkmark$
- If integer epoch is specified, halve for second stage (in code and documentation) $\checkmark$
- Swap `util` with `util.backend` and `util.helpers` to `util` $\checkmark$
- For code examples for decoupled fit, generate examples (with table) for imbalanced MNIST example, showing "normal" training procedure vs decoupled, comparing performance
	- Show differences in code (or rather, how few changes are needed)
	- Split into binary classification ($0$ and $1$, $0$ is common, $1$ is rare)
		- If TF's F1Score already supports multi-class. If so, add this example as well!
	- Table entries
		- Method
		- Time
		- F1 score
		- AUC
## Notes
- `Consideration for layer: Representation layer could be in separate branches?`
	-  can we know full graph, not just indices?

### 80:1, LR 1e-3, 20 epochs
- decoupled:
	AUC: 0.9158 - F1Score: 0.3444 - accuracy: 0.6098 - loss: 1.2162
- regular:
	AUC: 0.9513 - F1Score: 0.3437 - accuracy: 0.6728 - loss: 0.9193

### 80:1, LR 2e-5, 40 epochs
- decoupled
	AUC: 0.9356 - F1Score: 0.3087 - accuracy: 0.6162 - loss: 1.0570
- regular:
	AUC: 0.9418 - F1Score: 0.3204 - accuracy: 0.6380 - loss: 1.0002

### 100:1, LR 2e-5, 30 epochs, CNN structure	
- regular:
	AUC: 0.9487 - F1Score: 0.3443 - accuracy: 0.6480 - loss: 0.9729
- decoupled:
	 AUC: 0.9347 - F1Score: 0.3259 - accuracy: 0.6373 - loss: 1.0633

### 100:1, LR 2e-5, 20 epochs, CNN structure (FIXED AUC)
- regular
	 F1Score: 0.3219 - accuracy: 0.6400 - auc: 0.7442 - loss: 0.9689
- decoupled
	- F1Score: 0.3064 - accuracy: 0.6168 - auc: 0.7761 - loss: 0.9957
## Tasks:
- temp
## Tasks:
- Attempt implementation of AED decoupling approach
	- FULL DOCUMENTATION FOR NON-AED FIRST
- Representation layer finding is WRONG
	- Must be latest layer that is before the last trainable layer
	- Algorithm (much simpler than before) (see picture on phone):
		- user specifies layer
		- Check if there are weights beyond (closer to output) that layer
		- If not, move closer to input until a layer that does is found (it need not be a weighted layer)
- MNIST/other data (Maybe `iNaturalist 2018`???)
	- Binary classification
	- Table entries
		- Method
		- Time
		- F1 score
		- AUC
		- Take a look at confusion matrices, maybe save as well
- Confirm weights are being frozen by checking weights directly
- See if website has website/GitHub with link to dataset they used
- Read paper
	- Make sure CRT is implemented properly (Dr. Chan is not sure if there is more to it) 

# 11/24/25

## Prep:
- Attempt implementation of AED decoupling approach
	- FULL DOCUMENTATION FOR NON-AED FIRST
- Representation layer finding is WRONG $\checkmark$
	- Must be latest layer that is before the last trainable layer $\checkmark$
	- Algorithm (much simpler than before) (see picture on phone): $\checkmark$
		- user specifies layer $\checkmark$
		- Check if there are weights beyond (closer to output) that layer $\checkmark$
		- If not, move closer to input until a layer that does is found (it need not be a weighted layer) $\checkmark$
- MNIST/other data (CIFAR10) $\checkmark$
	- Binary classification $\checkmark$
	- Table entries $\checkmark$
		- Method $\checkmark$
		- Time $\checkmark$
		- F1 score $\checkmark$
		- AUC $\checkmark$
		- Take a look at confusion matrices, maybe save as well
- Confirm weights are being frozen by checking weights directly $\checkmark$
	- Confirmed working correctly!
- See if website has website/GitHub with link to dataset they used $\checkmark$
- Read paper $\checkmark$
	- Make sure CRT is implemented properly (Dr. Chan is not sure if there is more to it) $\checkmark$
	![[Pasted image 20251124091404.png]]

## Notes:
- All datasets that were used in the paper ([ImageNet 2014](https://image-net.org/index), [Places365](http://places2.csail.mit.edu/download.html), and [iNaturalist](https://github.com/visipedia/inat_comp/tree/master/2018#Data)) are far too large for to run on my system (100s of GB)
	- Sticking with either MNIST or STL10


#### First Test
- regular
	- F1Score: 0.2874 - accuracy: 0.5635 - auc: 0.7375 - loss: 1.2026
- decoupled
	- F1Score: 0.3241 - accuracy: 0.5382 - auc: 0.7855 - loss: 1.4098

#### First Test + 40% dropout, 300 epochs
- regular
	- F1Score: 0.3401 - accuracy: 0.6673 - auc: 0.7466 - loss: 1.1204
- decoupled
	- F1Score: 0.3233 - accuracy: 0.6260 - auc: 0.7902 - loss: 1.1640

#### CIFAR10, 20 epochs, ResNet50
- decoupled
	- F1Score: 0.6022 - accuracy: 0.6025 - auc: 0.9018 - loss: 1.4449
- regular
	- F1Score: 0.6126 - accuracy: 0.6134 - auc: 0.9080 - loss: 1.4085

#### CIFAR10, 20 epochs, simpler model
- regular
	- F1Score: 0.3192 - accuracy: 0.3331 - auc: 0.7985 - loss: 1.8219
- decoupled
	- F1Score: 0.3422 - accuracy: 0.3581 - auc: 0.8104 - loss: 1.7660

### BINARY CLASSIFICATION
#### CIFAR10, 20 epochs, simple model, 1:100
- regular
	- F1Score: 0.3332 - accuracy: 0.4998 - auc: 0.4574 - loss: 2.3868
	- Time: 9.37s
- decoupled
	- F1Score: 0.3332 - accuracy: 0.4998 - auc: 0.2715 - loss: 1.8800
	- Time: 16.58s
#### CIFAR10, 20 epochs, simple model, 1:10
- regular
	- F1Score: 0.3332 - accuracy: 0.4998 - auc: 0.9967 - loss: 1.1240
	- Time: 9.36
- decoupled
	- F1Score: 0.9850 - accuracy: 0.9850 - auc: 1.0000 - loss: 0.4123
	- Time: 15.62
