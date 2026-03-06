## Tasks:
- For testing weights over folds: $\checkmark$
	- Print out: All weights, average epoch, average metric used to determine best $\checkmark$
	- In my case: $MAE, MAE_R, AORE, \alpha, \text{val\_loss}$ $\checkmark$
	- AORE uses ABSOLUTE mean error $\checkmark$
	- Use AORE over `val_loss` for determining weights for both w/ and w/o CME data. Re-generate tables, save print outs across all alphas $\checkmark$
- Allow fits to take in multiple lists of weights $\checkmark$
	- Use same stopping criteria provided for user for validation across weights (finding best weights) $\checkmark$
	- Refactoring when possible!! $\checkmark$
- Outside `imbal`,  add variation of having an extra trained layers for decoupled fit $\times$
	- For tables, add two rows (since above always has -2 rep layer for first stage, then add additional layer in second stage)
- Inside `imbal`: $\checkmark$
	- update balanced/decoupled fit to incorporate finding the "best" class/sample weights based on the validation set $\checkmark$
		1.  Classification, iterate on different lists of class weights $\checkmark$
		2.  Regression, iterate on different lists of sample weights via alpha for reciprocal importance $\checkmark$
- **Later on:** MDI, wPCC
- **30%:** Refactoring / rewrite of `generate_decoder_branch`

---
### Regression on SEP-EC w/ CME (3/3/26)
For all runs below:
- Stratified batching is enabled
- Same seed used for consistency
### Balanced fit k-fold analysis
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0405    0.222     0.131     0.00571   245       
reciprocal, 0.1     0.0417    0.205     0.124     0.0078    248       
reciprocal, 0.2     0.0427    0.166     0.104     0.0104    202       
reciprocal, 0.3     0.0464    0.145     0.0958    0.0128    208       
reciprocal, 0.4     0.0474    0.134     0.0905    0.0153    268  <---      
reciprocal, 0.5     0.0529    0.132     0.0925    0.0205    253       
reciprocal, 0.6     0.0606    0.135     0.0976    0.0297    244       
reciprocal, 0.7     0.0711    0.144     0.108     0.0503    182       
reciprocal, 0.8     0.0822    0.131     0.106     0.067     270       
reciprocal, 0.9     0.114     0.146     0.13      0.056     212       
reciprocal, 1.0     0.121     0.153     0.137     0.0626    250  
denseweight, 0.1    0.0406    0.225     0.133     0.00582   230       
denseweight, 0.2    0.0399    0.228     0.134     0.00659   229       
denseweight, 0.3    0.0407    0.225     0.133     0.00652   222       
denseweight, 0.4    0.0403    0.219     0.13      0.00728   215       
denseweight, 0.5    0.0394    0.23      0.135     0.0078    256       
denseweight, 0.6    0.0386    0.214     0.126     0.00779   259       
denseweight, 0.7    0.0416    0.215     0.128     0.00885   179       
denseweight, 0.8    0.0405    0.225     0.133     0.00912   251       
denseweight, 0.9    0.0408    0.212     0.126     0.0106    251       
denseweight, 1.0    0.0502    0.201     0.126     0.0139    161       
denseweight, 1.1    0.0511    0.2       0.125     0.0147    179       
denseweight, 1.2    0.052     0.207     0.13      0.0129    207       
denseweight, 1.3    0.0557    0.2       0.128     0.0143    174       
denseweight, 1.4    0.0552    0.208     0.132     0.0146    239       
denseweight, 1.5    0.0639    0.195     0.13      0.0173    143       
denseweight, 1.6    0.0618    0.214     0.138     0.0157    241       
denseweight, 1.7    0.0627    0.191     0.127     0.0159    237       
denseweight, 1.8    0.0598    0.213     0.136     0.0167    254       
denseweight, 1.9    0.071     0.208     0.14      0.0174    228       
denseweight, 2.0    0.0634    0.207     0.135     0.0172    232     
```
### Decoupled fit k-fold analysis (stage 2, representation layer = -2)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0469    0.221     0.134     0.00609   45        
reciprocal, 0.1     0.0463    0.205     0.126     0.00787   56        
reciprocal, 0.2     0.046     0.192     0.119     0.0104    67        
reciprocal, 0.3     0.0456    0.182     0.114     0.0141    41        
reciprocal, 0.4     0.0454    0.173     0.109     0.0194    47        
reciprocal, 0.5     0.0448    0.168     0.106     0.0271    44        
reciprocal, 0.6     0.0439    0.162     0.103     0.0367    54        
reciprocal, 0.7     0.0439    0.159     0.101     0.0509    60        
reciprocal, 0.8     0.0442    0.155     0.0994    0.0678    37        
reciprocal, 0.9     0.0444    0.148     0.0963    0.0799    62        
reciprocal, 1.0     0.0447    0.145     0.0948    0.0982    57  <---
reciprocal, 1.1     0.0475    0.165     0.106     0.0957    24        
reciprocal, 1.2     0.0484    0.16      0.104     0.11      49        
reciprocal, 1.3     0.0483    0.161     0.105     0.122     27        
reciprocal, 1.4     0.0503    0.162     0.106     0.135     51        
reciprocal, 1.5     0.0509    0.162     0.106     0.141     51        
reciprocal, 1.6     0.0511    0.158     0.105     0.149     53        
reciprocal, 1.7     0.0495    0.162     0.106     0.165     30        
reciprocal, 1.8     0.0534    0.161     0.107     0.16      53        
reciprocal, 1.9     0.054     0.154     0.104     0.163     62        
reciprocal, 2.0     0.0506    0.155     0.103     0.17      27
denseweight, 0.1    0.0487    0.218     0.133     0.00666   21        
denseweight, 0.2    0.0485    0.212     0.13      0.00698   12        
denseweight, 0.3    0.0486    0.214     0.131     0.00744   31        
denseweight, 0.4    0.0483    0.211     0.13      0.00794   15        
denseweight, 0.5    0.0484    0.21      0.129     0.00852   24        
denseweight, 0.6    0.0481    0.208     0.128     0.00921   17        
denseweight, 0.7    0.0481    0.207     0.128     0.0101    40        
denseweight, 0.8    0.0481    0.207     0.128     0.0112    32        
denseweight, 0.9    0.0478    0.202     0.125     0.0125    22        
denseweight, 1.0    0.0477    0.2       0.124     0.0144    26        
denseweight, 1.1    0.0475    0.2       0.124     0.0157    31        
denseweight, 1.2    0.0474    0.198     0.123     0.0164    26        
denseweight, 1.3    0.0474    0.198     0.123     0.0172    33        
denseweight, 1.4    0.0474    0.198     0.123     0.0179    33        
denseweight, 1.5    0.0474    0.2       0.123     0.0185    53        
denseweight, 1.6    0.0471    0.196     0.122     0.0189    22        
denseweight, 1.7    0.047     0.195     0.121     0.0194    23        
denseweight, 1.8    0.0472    0.196     0.122     0.0198    30        
denseweight, 1.9    0.047     0.197     0.122     0.0202    26        
denseweight, 2.0    0.047     0.195     0.121     0.0206    32
```
### Decoupled fit k-fold analysis (stage 2, representation layer = -3)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0443    0.152     0.0982    0.00516   3         
reciprocal, 0.1     0.0435    0.147     0.0952    0.00639   8         
reciprocal, 0.2     0.043     0.142     0.0927    0.00779   6         
reciprocal, 0.3     0.0436    0.135     0.0893    0.0106    10        
reciprocal, 0.4     0.0427    0.131     0.0871    0.0139    14        
reciprocal, 0.5     0.0428    0.126     0.0846    0.019     12        
reciprocal, 0.6     0.043     0.117     0.0802    0.0271    15        
reciprocal, 0.7     0.042     0.121     0.0817    0.0378    12        
reciprocal, 0.8     0.0429    0.112     0.0776    0.0416    16        
reciprocal, 0.9     0.0422    0.118     0.0801    0.0435    11        
reciprocal, 1.0     0.0439    0.11      0.077     0.0456    28        
reciprocal, 1.1     0.0449    0.119     0.0818    0.0633    22        
reciprocal, 1.2     0.0429    0.117     0.08      0.0677    13  <---      
reciprocal, 1.3     0.0461    0.12      0.0833    0.0739    24        
reciprocal, 1.4     0.0487    0.119     0.0839    0.0692    34        
reciprocal, 1.5     0.0527    0.117     0.0851    0.0703    33        
reciprocal, 1.6     0.0566    0.132     0.0942    0.0882    19        
reciprocal, 1.7     0.0539    0.117     0.0855    0.14      31        
reciprocal, 1.8     0.0549    0.122     0.0885    0.11      22        
reciprocal, 1.9     0.066     0.126     0.0959    0.14      32        
reciprocal, 2.0     0.0671    0.127     0.0972    0.0956    28
denseweight, 0.1    0.0392    0.172     0.106     0.00447   6         
denseweight, 0.2    0.039     0.178     0.109     0.00464   7         
denseweight, 0.3    0.0385    0.181     0.11      0.00494   10        
denseweight, 0.4    0.038     0.189     0.113     0.00542   12        
denseweight, 0.5    0.0393    0.167     0.103     0.00567   6         
denseweight, 0.6    0.0383    0.178     0.108     0.00614   10        
denseweight, 0.7    0.0376    0.183     0.11      0.00657   12        
denseweight, 0.8    0.0375    0.185     0.111     0.00745   9         
denseweight, 0.9    0.0377    0.19      0.114     0.0084    13        
denseweight, 1.0    0.0379    0.198     0.118     0.00966   20        
denseweight, 1.1    0.0371    0.187     0.112     0.0103    13        
denseweight, 1.2    0.0374    0.185     0.111     0.0107    13        
denseweight, 1.3    0.0373    0.196     0.116     0.0114    14        
denseweight, 1.4    0.0383    0.2       0.119     0.0118    24        
denseweight, 1.5    0.0375    0.191     0.114     0.0122    19        
denseweight, 1.6    0.037     0.192     0.114     0.0121    12        
denseweight, 1.7    0.0381    0.192     0.115     0.0128    20        
denseweight, 1.8    0.0378    0.187     0.112     0.0129    18        
denseweight, 1.9    0.0378    0.195     0.117     0.0132    20        
denseweight, 2.0    0.037     0.191     0.114     0.0138    14
```


| Method                        | LR     | Epochs | Weights          | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----------------------------- | ------ | ------ | ---------------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Regular w/o AE                | $2e-4$ | 224    | ---              | 41.29    | 0.041           | 0.396             | 0.219            | 0.814         | 0.927           | 0.871          |
| Regular w/ AE (rep = $-2$)    | $2e-4$ | 224    | ---              | 58.68    | 0.038           | 0.329             | 0.184            | 0.823         | 0.925           | 0.874          |
| Regular w/ AE (rep = $-3$)    | $2e-4$ | 224    | ---              | 57.52    | 0.040           | 0.338             | 0.189            | 0.813         | 0.930           | 0.871          |
| Balanced w/o AE               | $2e-4$ | 268    | RI, $\alpha=0.4$ | 46.36    | 0.044           | 0.330             | 0.187            | 0.777         | 0.902           | 0.839          |
| Balanced w/ AE (rep = $-2$)   | $2e-4$ | 268    | RI, $\alpha=0.4$ | 69.24    | 0.053           | 0.309             | 0.181            | 0.734         | 0.911           | 0.823          |
| Balanced w/ AE (rep = $-3$)   | $2e-4$ | 268    | RI, $\alpha=0.4$ | 70.99    | 0.056           | 0.306             | 0.181            | 0.758         | 0.902           | 0.830          |
| Decoupled w/o AE (rep = $-2$) | $2e-4$ | 224/57 | RI, $\alpha=1$   |          |                 |                   |                  |               |                 |                |
| Decoupled w/o AE (rep = $-3$) | $2e-4$ | 224/13 | RI, $\alpha=1.2$ |          |                 |                   |                  |               |                 |                |
| Decoupled w/ AE (rep = $-2$)  | $2e-4$ | 224/57 | RI, $\alpha=1$   |          |                 |                   |                  |               |                 |                |
| Decoupled w/ AE (rep = $-3$)  | $2e-4$ | 224/13 | RI, $\alpha=1.2$ |          |                 |                   |                  |               |                 |                |
### Regression on SEP-EC w/o CME (3/3/26)
For all runs below:
- Stratified batching is enabled
- Same seed used for consistency

| Method                        | LR     | Epochs | Weights | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----------------------------- | ------ | ------ | ------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Regular w/o AE                | $2e-4$ | 118    | ---     | 53.90    | 0.023           | 0.131             | 0.077            | 0.961         | 0.980           | 0.971          |
| Regular w/ AE (rep = $-2$)    | $2e-4$ | 118    | ---     | 66.72    | 0.021           | 0.121             | 0.071            | 0.959         | 0.982           | 0.971          |
| Regular w/ AE (rep = $-3$)    | $2e-4$ | 118    | ---     | 67.65    | 0.024           | 0.101             | 0.063            | 0.962         | 0.984           | 0.973          |
| Balanced w/o AE               | $2e-4$ |        |         |          |                 |                   |                  |               |                 |                |
| Balanced w/ AE (rep = $-2$)   | $2e-4$ |        |         |          |                 |                   |                  |               |                 |                |
| Balanced w/ AE (rep = $-3$)   | $2e-4$ |        |         |          |                 |                   |                  |               |                 |                |
| Decoupled w/o AE (rep = $-2$) | $2e-4$ |        |         |          |                 |                   |                  |               |                 |                |
| Decoupled w/o AE (rep = $-3$) | $2e-4$ |        |         |          |                 |                   |                  |               |                 |                |
| Decoupled w/ AE (rep = $-2$)  | $2e-4$ |        |         |          |                 |                   |                  |               |                 |                |
| Decoupled w/ AE (rep = $-3$)  | $2e-4$ |        |         |          |                 |                   |                  |               |                 |                |

---

- RankSim
	- Two caveats
		- Non-differentiable
		- Impartial to the scaling/relative distance between features, regardless of the distances in the label space
		- Computational efficiency (is there a solution better than $O(n^2)$?)
			- $O(n\log n)$ seems feasible
		- Can we reduce redundant features? (t-SNE)
	- How to fix?
		- *Current idea:* Don't rank, normalize label/feature similarities

## Tasks
- Outside `imbal`,  add variation of having an extra trained layers for decoupled fit $\times$
	- For tables, add two rows (since above always has -2 rep layer for first stage, then add additional layer in second stage)
- Fit functions will run much slower, I suggest printing status messages. I suggest an int parameter to indicate message levels, this will also help debugging, for example: 
	 1. no messages (except those from `keras`/`tensorflow`)
	 2. main steps, found epoch number based on validation, class weights (alpha in reciprocal importance) based on validation..., training on the entire training set
	 3. Different class weights, alphas, ... being evaluated
- **Later on:** MDI, wPCC
- **30%:** Refactoring / rewrite of `generate_decoder_branch`