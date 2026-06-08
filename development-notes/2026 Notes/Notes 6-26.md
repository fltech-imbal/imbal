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
#### Low Priority:
- Try to fix `generate_decoder_branch`.
	- Hopefully, we can avoid needing a Flatten layer for reliability

# 6/5/26

#### High Priority:
- Allow for more epochs (use EarlyStopping for validation)
- Use `fold0` or similar to get training/validation split based on time series 
- Report AORE on top of plots (on top of common/rare MAE)