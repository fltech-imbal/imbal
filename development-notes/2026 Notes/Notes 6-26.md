# 6/1/26

## Tasks
#### High Priority
- Allow user to specify decision thresholds in fits? $\checkmark$
	- Instead, using the pre-existing metric sweep function. Add note in documentation saying it should be used if you want to have a more thorough decision threshold sweep $\checkmark$
- Update instances of "metric threshold" to "decision threshold" $\checkmark$
- Rerun and update SDObenchmark tutorials
- SPE-E
	- Three rows: regular, balanced, decoupled
	- Columns: basic, validation, `validation+AE`
	- SEP-EC data, but electron only, no CME data $\checkmark$
		- Create separate dataset files that have the CME data pre-removed $\checkmark$
#### Low Priority
- `GradCam`: Make sure to mention code is taken from that link in documentation $\checkmark$ (will be handled by Daniel)
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

## Notes
Toy dataset for gradient conflict
- Linear model with 2 features: $w_0+w_1x_1+w_2x_2$
- Gradient wrt. $w_1$ $= \frac{\partial E}{\partial w_1} =\frac{\partial E}{\partial o}\cdot\frac{\partial o}{\partial w_1}= \frac{1}{2}(t - o)^2 = (t-o)(\frac{\partial}{\partial \mathbf{w}}(w_0+w_1x_1+w_2x_2)) = (t-o)(w_1)$ 

$$
\begin{gathered}
E(t, o) = \frac{1}{2}(t-o)^2,\\
o = w_0+w_1x_1+w_2x_2,\\\\
\frac{\partial E}{\partial w_1} =\frac{\partial E}{\partial o}\cdot\frac{\partial o}{\partial w_1}= \frac{\partial}{\partial o}\cdot\frac{1}{2}(t - o)^2\cdot\frac{\partial o}{\partial w_1} = (t-o)(\frac{\partial}{\partial w_1}(w_0+w_1x_1+w_2x_2)) = (t-o)(w_1)
\\\\
\frac{\partial E}{\partial\mathbf{w}}=
(t-o)
\begin{bmatrix}
x_1\\x_2\\1
\end{bmatrix}
\end{gathered}
$$
$$
\begin{gathered}
G_1 = (t-o)\begin{bmatrix}1\\1\\1\end{bmatrix},~~
G_2 = (t-o)\begin{bmatrix}-1\\-1\\1\end{bmatrix}
\end{gathered}
$$
- We can control $t, x_1, x_2$ in the dataset
	- We can control $t$ to be close to 1
	- We can control $o$ to be close to $0$ by initializing the weights close to $0$ and by having a very small learning rate

$$
\begin{gathered}
G_1 = (1-o)\begin{bmatrix}1\\1\\1\end{bmatrix},~~
G_2 = (1.5-o)\begin{bmatrix}-1\\-1\\1\end{bmatrix}, ~~
G_3 = \left(1.6-o\right)\begin{bmatrix}-1\\0\\1\end{bmatrix}
\end{gathered}
$$
For generating more instances, using three existing points and add gaussian noise to the $x_1, x_2,$ and $t$ values, with mean $\mu=0, \sigma\approx0.1$. For example, $x_1'=x_1+gaussian(\mu=0, \sigma=0.1)$
#### Representation learning
- When the representation space is 2D, it can be plotted directly (no need for TSNE)
	- Therefore, the model will be $2\times2\times1$
- $t = x_1^2 + x_2^2$ = squared distance from origin (a unit circle if $t$ is kept constant)
- $x_1 = gaussian(\mu=0, \sigma=1), x_2=gaussian(\mu=0, \sigma=1)$
- 100+ instances ($\approx 500$)

---
# 6/3/26

#### High priority:
- Rerun and update SDObenchmark tutorials
- SPE-E $\checkmark$
	- Three rows: regular, balanced, decoupled $\checkmark$
	- Columns: basic, validation, `validation+AE` $\checkmark$


---
# 6/5/26

#### High Priority:
- Rerun and update SDObenchmark tutorials $\checkmark$
- Allow for more epochs (use EarlyStopping for validation) $\checkmark$
- Use `fold0` or similar to get training/validation split based on time series $\checkmark$ 
- Report AORE on top of plots (on top of common/rare MAE) $\checkmark$

---
# 6/10/26

#### High Priority:
- Vary AE representation layer, see if there are better results with third to last (instead of default second to last) $\checkmark$
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

---
# 6/12/26
#### High Priority
- **Using `imbal`, but not changing `imbal`**
	- Use MDI and DenseLoss weights as alternative weighting methods to reciprocal importance $\checkmark$
	- Use `MSE+wPCC` loss (use same $\lambda$ value from CISIR paper)
	- Whatever the best model result is, use T-SNE to visualize representation and then examine erroneous samples
- **NASA Report**
	- Accomplishments
		1. Features of the tool (same things/order shown in presentation)
			- Mention the tools works with tabular and image datasets, and classification and regression
			- Contains metrics used in heliophysics (TSS, HSS)
			- Fit methods
			- Optional features (validation, AE)
			- Multiple weight candidates
			- Candidate evaluation
			- TSNE visualization of latent space
			- Prediction explanation (LIME, SHAP, `GradCAM`)
		2. Experimental Evaluation
			- Describe SEP-EC dataset
				- Features, target
			- How does the tool perform for 9 configurations + including/excluding electron data
			- How does the tool perform with *external* additions (MDI, DenseWeight, `MSE+wPCC`) for 9 configurations
			- Using t-SNE and prediction explanation tools on model with best results
		3. Documentation and Tutorials
			- Documentation for each method
				- Describing all parameters, with code examples
			- Tutorials
				- 3x3 for tutorials + TSNE, LIME, SHAP, `GradCAM`
			- Installation guide
	-  Product
		- GitHub link
		- Documentation link
		- Papers
		
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

#### Class
- (`mean_ratio` - 1)^2 + variance is the loss function? Prevent collapse to 0 ratio
		- Or return to trying 1/`mean_ratio`?

---
# 6/15/26

## Tasks
#### High Priority
- 
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability
## Class
- Observation: when `stddev` is low, it seems to mainly be because the ratios are low, even if they are not necessarily "similar"
	- Ex. 0.1, 0.01, and 0.001 are three ratios that would have a small `stddev`, despite them being vastly different ratios (1 to 10, 1 to 100, 1 to 1000)
	- Solution: Minimize `stddev`(`log`(ratios)), as opposed to the previously minimizing `stddev`(ratios)

Before (enforcing `stddev`(ratios) of distances from endpoints and next point)
![[Pasted image 20260614161610.png|500]]
![[Pasted image 20260614161627.png|500]]
![[Pasted image 20260614161706.png|500]]

After (enforcing `stddev`(`log`(ratios)) of distances from endpoints and next point)![[Pasted image 20260614161850.png]]
![[Pasted image 20260614162000.png]]
![[Pasted image 20260614162022.png]]

Baseline (Just MSE, no representation enforcement loss)
![[Pasted image 20260614161934.png]]

t-SNE does not preserve linearity of representation space
![[Pasted image 20260614162054.png]]
![[Pasted image 20260614162132.png|400]]

4D Representation Space

True vs. Predicted                                               Dim 0 vs Dim 1
![[Pasted image 20260614162217.png]]
Dim 0 vs Dim 2                                                        Dim 0 vs Dim 3
![[Pasted image 20260614162409.png]]
Dim 1 vs Dim 2                                                         Dim 2 vs Dim 3
![[Pasted image 20260614162455.png]]
Dim2 vs Dim 3                                                                     t-SNE
![[Pasted image 20260614162539.png]]

**Extra thought:** If the enforcement of the representation learning performs well enough, there is actually no need to apply MSE error to the model during training at all. Instead, a simple least squares linear regression model can be applied to the output of the model (fit to the line `label = m*prediction + b`). Then, predictions from the neural network can be fed to the linear "correction" function after the fact.

True vs. predicted of model with MSE                Learned enforced representation
![[Pasted image 20260614164451.png]]

True vs. predicted after correction via simple linear fit
![[Pasted image 20260614164532.png|500]]

**Issue:** This method is highly unstable. I decided to try swapping to using MSE of the representation and label distances. We do lose the "degree of freedom" that is the
representation line being scaled arbitrarily, but this seems to not impact performance on the test set, and eliminates the instability issue that made itself apparent when trying to apply this method to SEP-EC.

Using MSE-based approach
![[Pasted image 20260614193904.png]]





Preliminary results for applying to SEP-EC

**Without representation enforcement loss (2 dimensional representation space, test set)**
True vs predicted                                                Explicit representation plot
![[Pasted image 20260614194130.png]]
![[Pasted image 20260614194302.png]]
t-SNE
![[Pasted image 20260614194153.png]]

**With representation enforcement loss (2 dimensional representation space, test set)**
True vs predicted                                                Explicit representation plot
![[Pasted image 20260614194302.png]]
t-SNE
![[Pasted image 20260614194328.png]]

The same idea of only applying the representation loss can be applied to this as well.

**Note on GD vs SGD**
When introducing a small amount of noise to the dataset, and allowing for a high-dimensional representation space, the enforced representation struggled to perform well when the data is not mini-batched.

**On whole dataset with noise (32-D representation space)**
True vs predicted (test)                 Dim 0 vs Dim 1 (test)                           t-SNE
![[Pasted image 20260615111905.png]]

**20 batches (no stratification), with noise (32-D representation space)**
True vs predicted (test)                 Dim 0 vs Dim 1 (test)                           t-SNE
![[Pasted image 20260615112301.png]]

Extra note: In the example above using MSE for representation learning, I am enforcing the following
- Distance between a sample $n$ and its neighbor $n+1$
- Distance between a sample $n$ and the first sample
- Distance between a sample $n$ and the last sample

Including all 3 allows for the quickest representation learning, but enforcing just the distances between neighbors and the distance to one endpoint is sufficient, but learning can be slower

---

# 6/15/26

Tried adding PCC to the loss
`1 - abs(PCC)`

`2 - tf.reduce_mean(tf.math.abs(tfp.stats.correlation(representations))) - tf.reduce_mean(tf.math.abs(tfp.stats.correlation(distance_pairs)))`

Worked for toy dataset, led to collapse on SEP-EC

Additional thought: Instead of individually trying to maximize the average correlation for the representation features and the average correlation between label and representation distances, we can instead append the label of each sample to its corresponding feature vector, and then simply maximize average correlation of these augmented representations.


Let $d$ be the dimensionality of the feature space, let $f$ be a feature vector with corresponding label $y$ :
$$
\begin{gathered}
f = \begin{bmatrix}z_1&z_2&\cdots&z_d\end{bmatrix}
\\\\\
\tilde{f} = \begin{bmatrix}z_1&z_2&\cdots&z_d&y\end{bmatrix}
\\\\
\mathbf{F}=\begin{bmatrix}f_1\\f_2\\\vdots\\f_n\end{bmatrix}\\
\\
\tilde{\mathbf{F}} = \begin{bmatrix}\tilde{f}_1\\\tilde{f}_2\\\vdots\\\tilde{f}_n\\\end{bmatrix}\\
\\
\mathcal{L}=1-\frac{1}{(d+1)^2}\sum_{i=1}^{d+1}\sum_{j=1}^{d+1}\rho(\tilde{\mathbf{F}}_{:,i},~\tilde{\mathbf{F}}_{:,j})
\end{gathered}
$$
This also gets rid of the need for sorting (which, we were assuming our batches would already be sorted, but this now works without that assumption)

**Results on toy test dataset using new loss approach:**
True vs. Predicted                                               Dim 0 vs Dim 1
![[Pasted image 20260617134114.png]]

**Results on SEP-EC test data using new loss approach**
![[Pasted image 20260617153802.png]]

Possibilities:
- $\frac{d_l(\text{extremes})}{d_f(\text{extremes})}$ for ratio, and $\left( x + \frac{1}{x}\right)^2$ as ratio penalty
- Instead of mean ratio, ratio of mean distances (instead of $\frac{1}{n}\sum^n\frac{d_l}{d_f}$, we do $\frac{\sum^n d_l}{\sum^n d_f}$)
	- No need for epsilon in this case (probably)
- Sum of values in each channel can be used as a substitute for fluence

#### High Priority
- **Using `imbal`, but not changing `imbal`**
	- Use MDI and DenseLoss weights as alternative weighting methods to reciprocal importance $\checkmark$
	- Use `MSE+wPCC` loss (use same $\lambda$ value from CISIR paper (found: 0.5))
		- Were the weights used at all in wPCC (aka, was it just normal PCC) **yes** $\checkmark$
	- Whatever the best model result is, use T-SNE to visualize representation and then examine erroneous samples
	- Include important function/parameter/class named from our tool
- **NASA Report**
	- Accomplishments
		1. Features of the tool (same things/order shown in presentation)
			- Mention the tools works with tabular and image datasets, and classification and regression
			- Contains metrics used in heliophysics (TSS, HSS)
			- Fit methods
			- Optional features (validation, AE)
			- Multiple weight candidates
			- Candidate evaluation
			- TSNE visualization of latent space
			- Prediction explanation (LIME, SHAP, `GradCAM`)
		2. Experimental Evaluation
			- Describe SEP-EC dataset
				- Features, target
			- How does the tool perform for 9 configurations + including/excluding electron data
			- How does the tool perform with *external* additions (MDI, DenseWeight, `MSE+wPCC`) for 9 configurations
			- Using t-SNE and prediction explanation tools on model with best results
		3. Documentation and Tutorials
			- Documentation for each method
				- Describing all parameters, with code examples
			- Tutorials
				- 3x3 for tutorials + TSNE, LIME, SHAP, `GradCAM`
			- Installation guide
	-  Product
		- GitHub link
		- Documentation link
		- Papers
		
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability


# 6/24/26
## NASA
#### High Priority:
- Only need to run test on best from `imbal` $\checkmark$
	- Try MDI/DW with wPCC for with/without electron data (and varied alpha) $\checkmark$
- Only discuss proton only/no electron dataset in report
	- Mention that we have electron data, but we took it out since we got better results without it
- In report, up to and including table for dataset complete by Friday $\checkmark$
	- Also complete section 1.1 $\checkmark$
		- Add picture form presentation for decoder branch generation $\checkmark$
		- Add picture from presentation for decoupled fit $\checkmark$
		- Add picture from validation split when explaining stratified split $\checkmark$
#### Low Priority:
- Check "collapsed" results, is wPCC close to 1 or close to 0? $\checkmark$
	- It was closer to 1, meaning PCC was taking priority over MSE. Lowering $\lambda$ to 0.02 on a regular fit allowed for MSE to taken precedence again.

---
# 6/26/26

## NASA:
#### High Priority:
- Handle 1.3.1 Documentation in report $\checkmark$
- Handle all red sections in report $\checkmark$

---
# 6/29/26
- Add circles to tabular LIME, and t-SNE plots
- Get rid of wine dataset plot in report $\checkmark$
	- Add column explanations further down in results section $\checkmark$
	- Table similar to Daniel's $\checkmark$
