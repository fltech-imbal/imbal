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
| Decoupled w/o AE (rep = $-2$) | $2e-4$ | 224/57 | RI, $\alpha=1$   | 51.52    | 0.052           | 0.363             | 0.207            | 0.824         | 0.949           | 0.886          |
| Decoupled w/o AE (rep = $-3$) | $2e-4$ | 224/13 | RI, $\alpha=1.2$ | 45.03    | 0.047           | 0.377             | 0.212            | 0.799         | 0.927           | 0.863          |
| Decoupled w/ AE (rep = $-2$)  | $2e-4$ | 224/57 | RI, $\alpha=1$   | 74.58    | 0.047           | 0.327             | 0.187            | 0.803         | 0.925           | 0.864          |
| Decoupled w/ AE (rep = $-3$)  | $2e-4$ | 224/13 | RI, $\alpha=1.2$ | 68.59    | 0.042           | 0.320             | 0.181            | 0.817         | 0.932           | 0.875          |
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
- Use folds already present in the SEP-EC data to re-generate best weights. $\checkmark$
 - Outside `imbal`,  add variation of having an extra trained layers for decoupled fit $\checkmark$
	- For tables, add two rows (since above always has -2 rep layer for first stage, then add additional layer in second stage (no AE, adding AE)) $\checkmark$
- Fit functions will run much slower, I suggest printing status messages. I suggest an int parameter to indicate message levels, this will also help debugging, for example: 
	 1. no messages (except those from `keras`/`tensorflow`)
	 2. main steps, found epoch number based on validation, class weights (alpha in reciprocal importance) based on validation..., training on the entire training set
	 3. Different class weights, alphas, ... being evaluated
- **30%:** Refactoring / rewrite of `generate_decoder_branch`
## Notes
- Fixed issue with `imbal.regression.get_sample_densities`. Previously, distribution used for KDE had to be the same as the list of labels used to retrieve densities for. Separate lists may now be specified.

---
## Balanced w/ CME (3/9/26)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0829    0.708     0.395     0.0262    32        
reciprocal, 0.1     0.0895    0.61      0.35      0.0356    55        
reciprocal, 0.2     0.0886    0.503     0.296     0.0479    96        
reciprocal, 0.3     0.0958    0.485     0.291     0.0636    122       
reciprocal, 0.4     0.095     0.482     0.288     0.0823    143       
reciprocal, 0.5     0.103     0.45      0.277     0.116     172       
reciprocal, 0.6     0.116     0.334     0.225     0.0958    263       
reciprocal, 0.7     0.122     0.329     0.225     0.137     398  <--- 
reciprocal, 0.8     0.146     0.337     0.242     0.164     335       
reciprocal, 0.9     0.234     0.43      0.332     0.28      142       
reciprocal, 1.0     0.233     0.429     0.331     0.445     125
--------------------------------------------------------------------------------
denseweight, 0.1    0.0833    0.692     0.388     0.0286    20        
denseweight, 0.2    0.0837    0.729     0.406     0.0301    5     
denseweight, 0.3    0.0862    0.698     0.392     0.0356    6         
denseweight, 0.4    0.0837    0.546     0.315     0.0356    84        
denseweight, 0.5    0.0863    0.623     0.355     0.0351    46        
denseweight, 0.6    0.087     0.623     0.355     0.0379    38        
denseweight, 0.7    0.094     0.63      0.362     0.0461    32        
denseweight, 0.8    0.0912    0.59      0.34      0.0475    57        
denseweight, 0.9    0.0828    0.416     0.249     0.0375    182       
denseweight, 1.0    0.0875    0.416     0.252     0.0398    205       
denseweight, 1.1    0.0982    0.396     0.247     0.0457    183       
denseweight, 1.2    0.103     0.381     0.242     0.0499    166       
denseweight, 1.3    0.104     0.399     0.251     0.0471    195       
denseweight, 1.4    0.109     0.391     0.25      0.0528    199       
denseweight, 1.5    0.112     0.422     0.267     0.0574    130       
denseweight, 1.6    0.106     0.441     0.273     0.0608    182       
denseweight, 1.7    0.107     0.402     0.255     0.0573    166       
denseweight, 1.8    0.108     0.389     0.249     0.0614    192       
denseweight, 1.9    0.105     0.405     0.255     0.0594    186       
denseweight, 2.0    0.114     0.363     0.239     0.065     261
--------------------------------------------------------------------------------
mdi, 0.1            0.567     0.691     0.629     0.000382  309
mdi, 0.2            0.27      0.334     0.302     0.000373  362
mdi, 0.3            0.215     0.386     0.3       0.000131  285
mdi, 0.4            0.186     0.395     0.291     7.58e-05  295
mdi, 0.5            0.168     0.413     0.29      5.36e-05  311
mdi, 0.6            0.147     0.43      0.288     4.18e-05  414
mdi, 0.7            0.145     0.441     0.293     3.3e-05   457
mdi, 0.8            0.134     0.457     0.296     3.08e-05  389
mdi, 0.9            0.125     0.46      0.292     2.53e-05  498
mdi, 1.0            0.119     0.443     0.281     2.18e-05  512
mdi, 1.1            0.112     0.489     0.3       2.12e-05  520
mdi, 1.25           0.103     0.544     0.323     2.03e-05  431
mdi, 1.4            0.0972    0.547     0.322     1.67e-05  429
mdi, 1.66           0.0953    0.655     0.375     1.89e-05  231
mdi, 2              0.0907    0.622     0.357     1.56e-05  326
mdi, 2.5            0.0911    0.687     0.389     1.39e-05  147
mdi, 3.33           0.0889    0.698     0.394     1.15e-05  168
mdi, 5              0.0879    0.7       0.394     9.52e-06  184
mdi, 10             0.0871    0.701     0.394     7.88e-06  194
```
## Decoupled (rep layer = -2) w/ CME (3/9/26)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0589    0.301     0.18      0.0112    64        
reciprocal, 0.1     0.0598    0.273     0.166     0.0135    52        
reciprocal, 0.2     0.0597    0.245     0.152     0.0191    50        
reciprocal, 0.3     0.0661    0.24      0.153     0.0282    57        
reciprocal, 0.4     0.0714    0.254     0.163     0.0328    37        
reciprocal, 0.5     0.0732    0.238     0.156     0.0452    30        
reciprocal, 0.6     0.0857    0.249     0.167     0.0597    41        
reciprocal, 0.7     0.0988    0.231     0.165     0.0739    38        
reciprocal, 0.8     0.104     0.227     0.166     0.0944    15        
reciprocal, 0.9     0.11      0.23      0.17      0.111     17        
reciprocal, 1.0     0.139     0.221     0.18      0.117     10
--------------------------------------------------------------------------------
denseweight, 0.1    0.0565    0.294     0.175     0.012     94        
denseweight, 0.2    0.0605    0.3       0.18      0.0122    82        
denseweight, 0.3    0.0589    0.287     0.173     0.0121    98        
denseweight, 0.4    0.0582    0.3       0.179     0.0124    98        
denseweight, 0.5    0.0596    0.273     0.166     0.0128    116       
denseweight, 0.6    0.0609    0.283     0.172     0.0157    106       
denseweight, 0.7    0.0626    0.268     0.165     0.0151    92        
denseweight, 0.8    0.0642    0.257     0.161     0.0169    93   <---   
denseweight, 0.9    0.0663    0.285     0.175     0.0203    105       
denseweight, 1.0    0.0753    0.253     0.164     0.0221    90        
denseweight, 1.1    0.0743    0.274     0.174     0.0222    71        
denseweight, 1.2    0.0781    0.275     0.177     0.0228    66        
denseweight, 1.3    0.0732    0.253     0.163     0.0262    108      
denseweight, 1.4    0.0811    0.263     0.172     0.0254    98        
denseweight, 1.5    0.08      0.278     0.179     0.0274    74        
denseweight, 1.6    0.0794    0.263     0.171     0.0286    80        
denseweight, 1.7    0.0821    0.272     0.177     0.0289    70        
denseweight, 1.8    0.0851    0.254     0.17      0.0275    62        
denseweight, 1.9    0.0922    0.257     0.175     0.0302    57        
denseweight, 2.0    0.0907    0.249     0.17      0.0297    56
--------------------------------------------------------------------------------
```
## Decoupled (rep layer = -3) w/ CME (3/9/26)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0579    0.316     0.187     0.0109    106       
reciprocal, 0.1     0.059     0.311     0.185     0.015     130       
reciprocal, 0.2     0.061     0.262     0.161     0.0194    155   
reciprocal, 0.3     0.0643    0.266     0.165     0.0274    148       
reciprocal, 0.4     0.0771    0.248     0.162     0.0439    32        
reciprocal, 0.5     0.0849    0.267     0.176     0.0552    26        
reciprocal, 0.6     0.0973    0.24      0.169     0.0543    23        
reciprocal, 0.7     0.0954    0.256     0.176     0.0791    68        
reciprocal, 0.8     0.0993    0.232     0.166     0.0863    22        
reciprocal, 0.9     0.13      0.225     0.178     0.125     13        
reciprocal, 1.0     0.157     0.232     0.194     0.133     27
--------------------------------------------------------------------------------
denseweight, 0.1    0.0581    0.273     0.166     0.0106    82 
denseweight, 0.2    0.0594    0.279     0.169     0.0115    92        
denseweight, 0.3    0.059     0.268     0.163     0.0123    95        
denseweight, 0.4    0.0592    0.285     0.172     0.014     62        
denseweight, 0.5    0.0604    0.276     0.168     0.0129    100       
denseweight, 0.6    0.0605    0.279     0.17      0.0134    82        
denseweight, 0.7    0.0624    0.26      0.161     0.0146    85   <---     
denseweight, 0.8    0.0646    0.271     0.168     0.0167    74        
denseweight, 0.9    0.0686    0.258     0.163     0.0184    85        
denseweight, 1.0    0.0733    0.243     0.158     0.0214    86        
denseweight, 1.1    0.07      0.264     0.167     0.0253    54        
denseweight, 1.2    0.076     0.264     0.17      0.0229    118       
denseweight, 1.3    0.0782    0.251     0.165     0.0227    66        
denseweight, 1.4    0.0742    0.269     0.171     0.0266    70        
denseweight, 1.5    0.0822    0.252     0.167     0.0239    52        
denseweight, 1.6    0.0789    0.267     0.173     0.0268    54        
denseweight, 1.7    0.088     0.241     0.165     0.025     95        
denseweight, 1.8    0.0904    0.249     0.17      0.0288    90        
denseweight, 1.9    0.0861    0.264     0.175     0.0307    64        
denseweight, 2.0    0.0842    0.26      0.172     0.0274    66
--------------------------------------------------------------------------------
```

Regular best epochs: $55$
All runs have learning rate of $2e-4$


| Method                        | Epochs   | Weights          | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----------------------------- | -------- | ---------------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Regular w/o AE                | 55       | ---              | 19.37    | ==0.054==       | 0.435             | 0.244            | 0.625         | 0.856           | 0.740          |
| Regular w/ AE (rep = $-2$)    | 55       | ---              | 29.26    | 0.067           | 0.422             | 0.245            | 0.491         | 0.821           | 0.656          |
| Regular w/ AE (rep = $-3$)    | 55       | ---              | 28.25    | 0.074           | 0.421             | 0.247            | 0.485         | 0.829           | 0.657          |
| Balanced w/o AE               | 398      | RI, $\alpha=0.7$ | 60.51    | 0.080           | 0.246             | 0.163            | ==0.669==     | 0.918           | ==0.793==      |
| Balanced w/ AE (rep = $-2$)   | 398      | RI, $\alpha=0.7$ | 91.64    | 0.105           | 0.241             | 0.173            | 0.605         | 0.913           | 0.759          |
| Balanced w/ AE (rep = $-3$)   | 398      | RI, $\alpha=0.7$ | 91.64    | 0.087           | ==0.207==         | ==0.147==        | 0.636         | ==0.938==       | 0.787          |
| Decoupled w/o AE (rep = $-2$) | 55 / 93  | DW, $\alpha=0.8$ | 34.16    | 0.065           | 0.414             | 0.240            | 0.532         | 0.832           | 0.682          |
| Decoupled w/o AE (rep = $-3$) | 55 / 85  | DW, $\alpha=0.7$ | 33.76    | 0.060           | 0.416             | 0.238            | 0.609         | 0.880           | 0.744          |
| Decoupled w/ AE (rep = $-2$)  | 55 / 93  | DW, $\alpha=0.8$ | 42.87    | 0.066           | 0.407             | 0.237            | 0.512         | 0.847           | 0.679          |
| Decoupled w/ AE (rep = $-3$)  | 55 / 85  | DW, $\alpha=0.7$ | 43.61    | 0.064           | 0.412             | 0.238            | 0.543         | 0.836           | 0.689          |
| Decoupled w/ AE (rep = $-3$)  | 55 / 85  | RI, $\alpha=0.7$ | 43.11    | 0.084           | 0.345             | 0.215            | 0.523         | 0.862           | 0.692          |
| Decoupled w/ AE (rep = $-3$)  | 55 / 398 | RI, $\alpha=0.7$ | 80.44    | 0.093           | 0.331             | 0.212            | 0.523         | 0.877           | 0.698          |
| Extended Decoupled w/o AE     | 55 / 85  | DW, $\alpha=0.7$ | 35.61    | 0.058           | 0.417             | 0.238            | 0.574         | 0.859           | 0.717          |
| Extended Decoupled w/ AE      | 55 / 85  | DW, $\alpha=0.7$ | 44.48    | 0.062           | 0.421             | 0.242            | 0.543         | 0.806           | 0.675          |
| Method                        | Epochs   | Weights          | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |

---
## Tasks
- Extended decoupled $\checkmark$
	- For second stage, do not freeze encoder (add as rows to table) $\checkmark$
- Add MDI $\checkmark$
- Add wPCC to loss function $\checkmark$
	- First, no sample weights (basically just 1 - PCC), then add sample weights (using MDI) $\checkmark$
- Early stopping with minimum number epochs $\checkmark$
- `get_sample_densities` documentation update
- Make sure documentation for fits specifies that multiple weight lists are allowed.
- Fit functions will run much slower, I suggest printing status messages. I suggest an int parameter to indicate message levels, this will also help debugging, for example: 
	- DO NOT called it `verbose`. Different from the preexisting TF fit parameter.
	 0. no messages (except those from `keras`/`tensorflow`)
	 1. main steps, found epoch number based on validation, class weights (alpha in reciprocal importance) based on validation..., training on the entire training set
	 2. Different class weights, alphas, ... being evaluated
- **30%:** Refactoring / rewrite of `generate_decoder_branch`


---
## Balanced w/ CME (3/12/26)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0653    0.384     0.225     0.112     462       
reciprocal, 0.1     0.0679    0.357     0.212     0.106     497       
reciprocal, 0.2     0.0693    0.37      0.219     0.145     404       
reciprocal, 0.3     0.0607    0.34      0.2       0.141     582       
reciprocal, 0.4     0.072     0.35      0.211     0.317     440       
reciprocal, 0.5     0.0807    0.32      0.2       0.528     396       
reciprocal, 0.6     0.0774    0.282     0.18      0.576     415  <---     
reciprocal, 0.7     0.0863    0.305     0.196     1.84      366       
reciprocal, 0.8     0.0881    0.284     0.186     1.82      378       
reciprocal, 0.9     0.13      0.271     0.2       2.78      410       
reciprocal, 1.0     0.103     0.286     0.195     2.86      412
--------------------------------------------------------------------------------
denseweight, 0.1    0.0734    0.384     0.229     0.12      452       
denseweight, 0.2    0.0641    0.379     0.222     0.107     526       
denseweight, 0.3    0.0631    0.379     0.221     0.1       599       
denseweight, 0.4    0.0605    0.388     0.224     0.101     503       
denseweight, 0.5    0.0646    0.372     0.219     0.107     429       
denseweight, 0.6    0.0646    0.392     0.228     0.114     446       
denseweight, 0.7    0.0703    0.359     0.214     0.107     470       
denseweight, 0.8    0.0592    0.362     0.211     0.098     551       
denseweight, 0.9    0.0653    0.377     0.221     0.114     420       
denseweight, 1.0    0.0661    0.371     0.219     0.116     429       
denseweight, 1.1    0.0678    0.391     0.229     0.132     513       
denseweight, 1.2    0.0642    0.36      0.212     0.116     459       
denseweight, 1.3    0.0711    0.363     0.217     0.141     411       
denseweight, 1.4    0.0755    0.381     0.228     0.16      390       
denseweight, 1.5    0.0649    0.367     0.216     0.147     393       
denseweight, 1.6    0.0634    0.368     0.216     0.16      555       
denseweight, 1.7    0.0683    0.373     0.221     0.179     401       
denseweight, 1.8    0.0663    0.361     0.214     0.169     512       
denseweight, 1.9    0.0731    0.343     0.208     0.158     440  <---     
denseweight, 2.0    0.0717    0.353     0.212     0.158     468
--------------------------------------------------------------------------------
mdi, 0.1            0.161     0.614     0.387     0.0765    337       
mdi, 0.2            0.087     0.461     0.274     0.0427    483       
mdi, 0.3            0.0657    0.415     0.24      0.0351    648       
mdi, 0.4            0.0728    0.467     0.27      0.0453    487       
mdi, 0.5            0.067     0.461     0.264     0.047     556       
mdi, 0.6            0.0764    0.425     0.251     0.0419    583       
mdi, 0.7            0.0677    0.406     0.237     0.0425    640       
mdi, 0.8            0.0752    0.503     0.289     0.0522    462       
mdi, 0.9            0.0696    0.439     0.254     0.0484    580       
mdi, 1.0            0.0713    0.463     0.267     0.0444    538       
mdi, 1.1            0.0597    0.434     0.247     0.0453    617       
mdi, 1.25           0.0707    0.439     0.255     0.0529    530       
mdi, 1.4            0.0639    0.417     0.241     0.0506    634  <---    
mdi, 1.66           0.0692    0.437     0.253     0.0549    554       
mdi, 2              0.0636    0.429     0.246     0.056     622       
mdi, 2.5            0.0646    0.456     0.26      0.0636    524       
mdi, 3.33           0.0659    0.48      0.273     0.0711    500       
mdi, 5              0.0604    0.425     0.243     0.067     627       
mdi, 10             0.0677    0.424     0.246     0.071     679
```
## Decoupled (rep layer = -2) w/ CME (3/12/26)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0441    0.258     0.151     0.0578    206       
reciprocal, 0.1     0.041     0.247     0.144     0.055     199       
reciprocal, 0.2     0.0398    0.254     0.147     0.0589    156       
reciprocal, 0.3     0.0382    0.24      0.139     0.0672    169       
reciprocal, 0.4     0.0452    0.233     0.139     0.0894    148       
reciprocal, 0.5     0.0401    0.221     0.131     0.12      133       
reciprocal, 0.6     0.0489    0.215     0.132     0.198     82        
reciprocal, 0.7     0.0452    0.214     0.13      0.307     62        
reciprocal, 0.8     0.0482    0.209     0.129     0.437     123  <--- 
reciprocal, 0.9     0.0496    0.224     0.137     0.775     121       
reciprocal, 1.0     0.0525    0.227     0.14      1.15      63 
--------------------------------------------------------------------------------
denseweight, 0.1    0.0401    0.249     0.145     0.0575    132       
denseweight, 0.2    0.0435    0.25      0.147     0.0578    251       
denseweight, 0.3    0.0405    0.245     0.143     0.0559    176       
denseweight, 0.4    0.042     0.246     0.144     0.0564    128       
denseweight, 0.5    0.0402    0.261     0.15      0.0544    252       
denseweight, 0.6    0.0425    0.251     0.147     0.0528    142       
denseweight, 0.7    0.0413    0.252     0.146     0.0531    201       
denseweight, 0.8    0.0393    0.247     0.143     0.0502    104       
denseweight, 0.9    0.041     0.239     0.14      0.0492    143       
denseweight, 1.0    0.0402    0.236     0.138     0.0466    185       
denseweight, 1.1    0.0417    0.244     0.143     0.0496    204       
denseweight, 1.2    0.0405    0.233     0.137     0.0507    131       
denseweight, 1.3    0.0423    0.233     0.137     0.0536    135       
denseweight, 1.4    0.0394    0.241     0.14      0.0565    193       
denseweight, 1.5    0.0408    0.237     0.139     0.0588    120       
denseweight, 1.6    0.0418    0.234     0.138     0.0595    162       
denseweight, 1.7    0.0414    0.231     0.136     0.0597    137       
denseweight, 1.8    0.0427    0.233     0.138     0.0646    53        
denseweight, 1.9    0.0406    0.23      0.135     0.0644    207  <---     
denseweight, 2.0    0.0403    0.238     0.139     0.0683    146
--------------------------------------------------------------------------------
mdi, 0.1            0.106     0.173     0.139     0.0114    34   <---
mdi, 0.2            0.0531    0.225     0.139     0.0104    140
mdi, 0.3            0.0476    0.265     0.156     0.0132    327
mdi, 0.4            0.0519    0.254     0.153     0.0141    344
mdi, 0.5            0.0597    0.228     0.144     0.0167    239
mdi, 0.6            0.0514    0.238     0.144     0.0182    209
mdi, 0.7            0.053     0.268     0.161     0.0195    302
mdi, 0.8            0.0498    0.243     0.147     0.02      286
mdi, 0.9            0.0491    0.275     0.162     0.0223    277
mdi, 1.0            0.0521    0.279     0.166     0.0241    282
mdi, 1.1            0.0562    0.277     0.166     0.0246    387
mdi, 1.25           0.0493    0.268     0.159     0.0261    230
mdi, 1.4            0.0511    0.302     0.177     0.0271    374
mdi, 1.66           0.0525    0.259     0.156     0.0298    248
mdi, 2              0.0511    0.286     0.168     0.0327    273
mdi, 2.5            0.0541    0.299     0.176     0.036     257
mdi, 3.33           0.0516    0.261     0.156     0.0378    335
mdi, 5              0.0504    0.268     0.159     0.043     254
mdi, 10             0.0535    0.24      0.147     0.0456    325
```
## Decoupled (rep layer = -3) w/ CME (3/12/26)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0464    0.249     0.148     0.0605    186       
reciprocal, 0.1     0.0499    0.249     0.15      0.0627    229       
reciprocal, 0.2     0.048     0.249     0.148     0.0667    202       
reciprocal, 0.3     0.0444    0.231     0.138     0.0705    182       
reciprocal, 0.4     0.047     0.227     0.137     0.09      114       
reciprocal, 0.5     0.049     0.223     0.136     0.131     81        
reciprocal, 0.6     0.0455    0.212     0.129     0.191     101  <---     
reciprocal, 0.7     0.052     0.215     0.134     0.316     92        
reciprocal, 0.8     0.0481    0.212     0.13      0.5       75        
reciprocal, 0.9     0.0516    0.214     0.133     0.768     32        
reciprocal, 1.0     0.0485    0.209     0.129     0.995     38
--------------------------------------------------------------------------------
denseweight, 0.1    0.0394    0.261     0.15      0.0558    274       
denseweight, 0.2    0.0426    0.245     0.144     0.0553    212       
denseweight, 0.3    0.0405    0.244     0.142     0.0557    209       
denseweight, 0.4    0.0384    0.251     0.145     0.0558    261       
denseweight, 0.5    0.0396    0.248     0.144     0.0538    304       
denseweight, 0.6    0.0392    0.245     0.142     0.0512    254       
denseweight, 0.7    0.0402    0.253     0.146     0.049     364       
denseweight, 0.8    0.0395    0.24      0.14      0.0496    167       
denseweight, 0.9    0.0418    0.238     0.14      0.0495    150       
denseweight, 1.0    0.0379    0.24      0.139     0.0465    130       
denseweight, 1.1    0.0413    0.234     0.138     0.0478    94        
denseweight, 1.2    0.0414    0.227     0.134     0.0524    60        
denseweight, 1.3    0.0403    0.227     0.134     0.0523    172       
denseweight, 1.4    0.0408    0.232     0.137     0.0549    154       
denseweight, 1.5    0.0406    0.231     0.136     0.058     150       
denseweight, 1.6    0.0417    0.233     0.137     0.0595    117       
denseweight, 1.7    0.0416    0.221     0.131     0.0594    75   <---     
denseweight, 1.8    0.0425    0.223     0.133     0.0623    50        
denseweight, 1.9    0.0444    0.224     0.134     0.0647    83        
denseweight, 2.0    0.0398    0.228     0.134     0.0652    36
--------------------------------------------------------------------------------
mdi, 0.1            0.102     0.187     0.144     0.0127    21        
mdi, 0.2            0.0446    0.201     0.123     0.0133    34   <---      
mdi, 0.3            0.0486    0.199     0.124     0.0157    121       
mdi, 0.4            0.0433    0.202     0.123     0.0176    34        
mdi, 0.5            0.051     0.242     0.146     0.0187    198       
mdi, 0.6            0.0451    0.237     0.141     0.0214    134       
mdi, 0.7            0.0511    0.265     0.158     0.0216    206       
mdi, 0.8            0.053     0.284     0.169     0.0226    230       
mdi, 0.9            0.0438    0.254     0.149     0.0243    174       
mdi, 1.0            0.0461    0.287     0.167     0.0246    202       
mdi, 1.1            0.0517    0.271     0.161     0.0265    185       
mdi, 1.25           0.0491    0.294     0.171     0.027     244       
mdi, 1.4            0.0511    0.266     0.158     0.0288    178       
mdi, 1.66           0.0537    0.279     0.166     0.0317    280       
mdi, 2              0.0567    0.303     0.18      0.034     232       
mdi, 2.5            0.0507    0.278     0.164     0.0367    202       
mdi, 3.33           0.0535    0.278     0.166     0.0391    296       
mdi, 5              0.059     0.283     0.171     0.0431    260       
mdi, 10             0.0504    0.27      0.16      0.0481    215
```

Regular best epochs: $514$
All runs have learning rate of $2e-4$

| Method                               | Epochs  | Weights           | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ------------------------------------ | ------- | ----------------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Regular w/o AE                       | 514     | ---               | 81.54    | 0.038           | 0.234             | 0.136            | 0.869         | 0.961           | 0.914          |
| Regular w/ AE (rep = $-2$)           | 514     | ---               | 119.62   | 0.039           | 0.222             | 0.131            | 0.827         | 0.966           | 0.896          |
| Regular w/ AE (rep = $-3$)           | 514     | ---               | 130.15   | 0.038           | 0.176             | 0.107            | 0.839         | 0.972           | 0.905          |
| Method                               | Epochs  | Weights           | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| Balanced w/o AE                      | 415     | RI, $\alpha=0.6$  | 67.36    | 0.041           | 0.218             | 0.130            | 0.813         | 0.952           | 0.883          |
| Balanced w/o AE                      | 440     | DW, $\alpha=1.9$  | 68.31    | 0.035           | 0.241             | 0.138            | 0.880         | 0.968           | 0.924          |
| Balanced w/ AE (rep = $-2$)          | 415     | RI, $\alpha=0.6$  | 106.13   | 0.041           | 0.186             | 0.114            | 0.824         | 0.949           | 0.886          |
| Balanced w/ AE (rep = $-2$)          | 440     | DW, $\alpha=1.9$  | 107.40   | 0.033           | 0.188             | 0.110            | 0.876         | 0.969           | 0.922          |
| Balanced w/ AE (rep = $-3$)          | 415     | RI, $\alpha=0.6$  | 102.69   | 0.041           | 0.163             | 0.102            | 0.848         | 0.959           | 0.903          |
| Balanced w/ AE (rep = $-3$)          | 440     | DW, $\alpha=1.9$  | 108.37   | 0.040           | 0.194             | 0.117            | 0.864         | 0.971           | 0.918          |
| Method                               | Epochs  | Weights           | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| Decoupled w/o AE (rep = $-2$)        | 514/123 | RI, $\alpha=0.8$  | 96.41    | 0.039           | 0.187             | 0.113            | 0.874         | 0.947           | 0.911          |
| Decoupled w/o AE (rep = $-2$)        | 514/34  | MDI, $\alpha=0.2$ | 86.36    | 0.51            | 0.396             | 0.224            | 0.874         | 0.962           | 0.918          |
| Decoupled w/o AE (rep = $-3$)        | 514/101 | RI, $\alpha=0.6$  | 95.86    | 0.037           | 0.253             | 0.145            | 0.842         | 0.930           | 0.886          |
| Decoupled w/o AE (rep = $-3$)        | 514/34  | MDI, $\alpha=0.2$ | 91.04    | 0.059           | 0.379             | 0.219            | 0.869         | 0.961           | 0.915          |
| Decoupled w/ AE (rep = $-2$)         | 514/123 | RI, $\alpha=0.8$  | 136.84   | 0.042           | 0.189             | 0.115            | 0.801         | 0.951           | 0.876          |
| Decoupled w/ AE (rep = $-2$)         | 514/34  | MDI, $\alpha=0.2$ | 131.96   | 0.045           | 0.326             | 0.185            | 0.863         | 0.966           | 0.915          |
| Decoupled w/ AE (rep = $-3$)         | 514/101 | RI, $\alpha=0.6$  | 141.88   | 0.037           | 0.197             | 0.117            | 0.881         | 0.954           | 0.917          |
| Decoupled w/ AE (rep = $-3$)         | 514/34  | MDI, $\alpha=0.2$ | 136.99   | 0.051           | 0.351             | 0.201            | 0.864         | 0.965           | 0.914          |
| Method                               | Epochs  | Weights           | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| Extended Decoupled w/o AE            | 514/101 | RI, $\alpha=0.6$  | 100.37   | 0.040           | 0.245             | 0.143            | 0.846         | 0.928           | 0.887          |
| Extended Decoupled w/ AE             | 514/101 | RI, $\alpha=0.6$  | 148.00   | 0.039           | 0.198             | 0.119            | 0.802         | 0.946           | 0.874          |
| Extended Decoupled w/o AE (unfrozen) | 514/101 | RI, $\alpha=0.6$  | 104.79   | 0.094           | 0.264             | 0.179            | 0.784         | 0.950           | 0.867          |
| Extended Decoupled w/ AE (unfrozen)  | 514/101 | RI, $\alpha=0.6$  | 151.16   | 0.085           | 0.283             | 0.184            | 0.834         | 0.964           | 0.899          |
| Method                               | Epochs  | Weights           | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |

---
## Tasks:
- Seeing if wPCC improves performance when on/off $\checkmark$
- Comparison of MDI, DenseWeight, RecImp (expectation is that MDI improves) $\checkmark$
- `get_sample_densities` documentation update $\checkmark$
- Make sure documentation for fits specifies that multiple weight lists are allowed. $\checkmark$
- Fit functions will run much slower, I suggest printing status messages. I suggest an int parameter to indicate message levels, this will also help debugging, for example: $\checkmark$
	- DO NOT called it `verbose`. Different from the preexisting TF fit parameter. $\checkmark$
		- `verbose_imbal` $\checkmark$
	 0. no messages (except those from `keras`/`tensorflow`)  $\checkmark$
	 1. main steps, found epoch number based on validation, class weights (alpha in reciprocal importance) based on validation..., training on the entire training set $\checkmark$
	 2. Different class weights, alphas, ... being evaluated $\checkmark$
 - See Daniel's message from 3/10, either revert or use suggested change $\checkmark$

---
Regular best epochs with wPCC: $514$
Regular best epochs without wPCC: $527$
All runs have learning rate of $2e-4$
## Regular Fit
| AE?      | wPCC? | Epochs | Weights | Time (s) | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| -------- | ----- | ------ | ------- | -------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| No       | Yes   | 514    | ---     | 81.54    | 0.038           | 0.234             | 0.136            | 0.869         | 0.961           | 0.914          |
| No       | No    | 527    | ---     | 81.19    | 0.041           | 0.244             | 0.143            | 0.805         | 0.970           | 0.888          |
| Yes (-2) | Yes   | 514    | ---     | 119.62   | 0.039           | 0.222             | 0.131            | 0.827         | 0.966           | 0.896          |
| Yes (-2) | No    | 527    | ---     | 122.36   | 0.036           | 0.222             | 0.129            | 0.869         | 0.964           | 0.917          |
| Yes (-3) | Yes   | 514    | ---     | 130.15   | 0.038           | 0.176             | 0.107            | 0.839         | 0.972           | 0.905          |
| Yes (-3) | No    | 527    | ---     | 120.66   | 0.035           | 0.197             | 0.116            | 0.871         | 0.966           | 0.919          |
## Balanced Fit

| AE?      | wPCC?   | Epochs | Weights           | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| -------- | ------- | ------ | ----------------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| No       | Yes     | 415    | RI, $\alpha=0.6$  | 0.041           | 0.240             | 0.140            | 0.841         | 0.941           | 0.891          |
| No       | No      | 415    | RI, $\alpha=0.6$  | 0.067           | 0.214             | 0.140            | 0.720         | 0.932           | 0.826          |
| No       | Yes     | 440    | DW, $\alpha=1.9$  | 0.041           | 0.236             | 0.138            | 0.844         | 0.964           | 0.904          |
| No       | No      | 440    | DW, $\alpha=1.9$  | 0.062           | 0.296             | 0.179            | 0.751         | 0.960           | 0.855          |
| No       | Yes     | 634    | MDI, $\alpha=1.4$ | 0.034           | 0.290             | 0.162            | 0.901         | 0.964           | 0.933          |
| No       | No      | 634    | MDI, $\alpha=1.4$ | 0.059           | 0.328             | 0.193            | 0.684         | 0.859           | 0.771          |
| Yes (-2) | ==Yes== | 415    | RI, $\alpha=0.6$  | 0.040           | 0.192             | 0.116            | 0.859         | 0.949           | 0.904          |
| Yes (-2) | No      | 415    | RI, $\alpha=0.6$  | 0.056           | 0.237             | 0.146            | 0.724         | 0.929           | 0.827          |
| Yes (-2) | ==Yes== | 440    | DW, $\alpha=1.9$  | 0.038           | 0.191             | 0.115            | 0.900         | 0.969           | 0.935          |
| Yes (-2) | No      | 440    | DW, $\alpha=1.9$  | 0.065           | 0.230             | 0.147            | 0.754         | 0.964           | 0.859          |
| Yes (-2) | ==Yes== | 634    | MDI, $\alpha=1.4$ | 0.031           | 0.223             | 0.127            | 0.910         | 0.974           | 0.942          |
| Yes (-2) | No      | 634    | MDI, $\alpha=1.4$ | 0.063           | 0.758             | 0.410            | 0.062         | 0.296           | 0.179          |
| Yes (-3) | Yes     | 415    | RI, $\alpha=0.6$  | 0.044           | 0.181             | 0.112            | 0.806         | 0.952           | 0.878          |
| Yes (-3) | No      | 415    | RI, $\alpha=0.6$  | 0.069           | 0.233             | 0.151            | 0.695         | 0.928           | 0.811          |
| Yes (-3) | Yes     | 440    | DW, $\alpha=1.9$  | 0.038           | 0.208             | 0.123            | 0.862         | 0.962           | 0.911          |
| Yes (-3) | No      | 440    | DW, $\alpha=1.9$  | 0.068           | 0.209             | 0.138            | 0.773         | 0.964           | 0.868          |
| Yes (-3) | Yes     | 634    | MDI, $\alpha=1.4$ | 0.031           | 0.178             | 0.105            | 0.910         | 0.971           | 0.940          |
| Yes (-3) | No      | 634    | MDI, $\alpha=1.4$ | 0.067           | 0.735             | 0.401            | 0.146         | 0.175           | 0.160          |


## Decoupled Fit (rep=-3)

| AE? | wPCC? | Epochs  | Weights           | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| --- | ----- | ------- | ----------------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| No  | Yes   | 514/101 | RI, $\alpha=0.6$  | 0.037           | 0.238             | 0.138            | 0.851         | 0.938           | 0.895          |
| No  | No    | 527/101 | RI, $\alpha=0.6$  | 0.63            | 0.205             | 0.134            | 0.709         | 0.938           | 0.824          |
| No  | Yes   | 514/75  | DW, $\alpha=1.7$  | 0.033           | 0.219             | 0.126            | 0.888         | 0.969           | 0.928          |
| No  | No    | 527/75  | DW, $\alpha=1.7$  | 0.054           | 0.274             | 0.163            | 0.785         | 0.966           | 0.875          |
| No  | Yes   | 514/34  | MDI, $\alpha=0.2$ | 0.396           | 0.224             | 0.224            | 0.894         | 0.970           | 0.932          |
| No  | No    | 527/34  | MDI, $\alpha=0.2$ | 0.179           | 0.270             | 0.225            | 0.4432        | 0.910           | 0.671          |
| Yes | Yes   | 514/101 | RI, $\alpha=0.6$  | 0.043           | 0.192             | 0.118            | 0.820         | 0.956           | 0.888          |
| Yes | No    | 527/101 | RI, $\alpha=0.6$  | 0.068           | 0.187             | 0.127            | 0.722         | 0.943           | 0.832          |
| Yes | Yes   | 514/75  | DW, $\alpha=1.7$  | 0.036           | 0.179             | 0.107            | 0.844         | 0.969           | 0.906          |
| Yes | No    | 527/75  | DW, $\alpha=1.7$  | 0.059           | 0.206             | 0.133            | 0.780         | 0.969           | 0.874          |
| Yes | Yes   | 514/34  | MDI, $\alpha=0.2$ | 0.051           | 0.311             | 0.182            | 0.874         | 0.966           | 0.920          |
| Yes | No    | 527/34  | MDI, $\alpha=0.2$ | 0.149           | 0.683             | 0.416            | 0.092         | 0.217           | 0.155          |

## Extended (extra layer second stage, unfrozen)

| AE?      | wPCC? | Epochs  | Weights           | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| -------- | ----- | ------- | ----------------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| No       | Yes   | 514/415 | RI, $\alpha=0.6$  | 0.082           | 0.356             | 0.219            | 0.850         | 0.955           | 0.903          |
| No       | No    | 527/415 | RI, $\alpha=0.6$  | 0.102           | 0.385             | 0.243            | 0.791         | 0.928           | 0.859          |
| No       | Yes   | 514/440 | DW, $\alpha=1.9$  | 0.06            | 0.035             | 0.205            | 0.856         | 0.969           | 0.912          |
| No       | No    | 527/440 | DW, $\alpha=1.9$  | 0.076           | 0.367             | 0.221            | 0.818         | 0.971           | 0.895          |
| No       | Yes   | 514/634 | MDI, $\alpha=1.4$ | 0.039           | 0.400             | 0.220            | 0.921         | 0.976           | 0.948          |
| No       | No    | 527/634 | MDI, $\alpha=1.4$ | 0.222           | 0.386             | 0.304            | 0.870         | 0.925           | 0.897          |
| Yes (-2) | Yes   | 514/415 | RI, $\alpha=0.6$  | 0.017           | 0.440             | 0.278            | 0.842         | 0.936           | 0.889          |
| Yes (-2) | No    | 527/415 | RI, $\alpha=0.6$  | 0.051           | 0.439             | 0.245            | 0.736         | 0.929           | 0.833          |
| Yes (-2) | Yes   | 514/440 | DW, $\alpha=1.9$  | 0.079           | 0.336             | 0.207            | 0.867         | 0.974           | 0.920          |
| Yes (-2) | No    | 527/440 | DW, $\alpha=1.9$  | 0.090           | 0.442             | 0.266            | 0.849         | 0.971           | 0.910          |
| Yes (-2) | Yes   | 514/634 | MDI, $\alpha=1.4$ | 0.049           | 0.393             | 0.221            | 0.915         | 0.977           | 0.946          |
| Yes (-2) | No    | 527/634 | MDI, $\alpha=1.4$ | 0.063           | 0.751             | 0.407            | 0.718         | 0.812           | 0.765          |

---

## Notes
- For debug messages, you asked for alphas to be printed for reciprocal importance. However, alpha values are not passed to fit.
- Got code running on Ai-Panther (had some odd issues related to memory limits, despite the code working fine on my laptop, which does not have a lot of memory)
## Tasks
- Notes from emails (3/12)
	- `model.best_sample_weigts`, `model.best_class_weights`, and `model.best_metric_threshold`
	- Step size of $0.1$ for threshold testing
- For class weights in multi-fit, print out at `verbose_imbal>1`.
- See "more notes from 3/17"
	- On `verbose_imbal`
		- For classification, print the index and class weights; if more than 5, print the first 5 followed by ... 
		- For regression, print the index and first 5 sample weights followed by ...
- Restrict runs to only with AE (always at -2) $\checkmark$
	- Vary wPCC on/off $\checkmark$
	- RI, DW, MDI $\checkmark$
	- Decoupled is only at -2, (like Kang et al.) $\checkmark$
	- Extended stays (-2, but add extra layer) $\checkmark$
		- Stick with frozen, but add unfrozen if results are off $\checkmark$
	- For k-fold validation, prioritize decoupled and "extended", to ensure some results for Thursday $\checkmark$
- Remove in-batch randomization $\checkmark$

---

Regular best epochs with wPCC: $514$
Regular best epochs without wPCC: $527$

For all:
- Learning rate is $2e-4$
- AE is on
- wPCC is on
## Decoupled (rep layer = -2) w/ CME (3/19/26)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0579    0.202     0.13      0.102     105
reciprocal, 0.1     0.0581    0.201     0.13      0.0985    133
reciprocal, 0.2     0.0584    0.199     0.129     0.0965    87
reciprocal, 0.3     0.059     0.199     0.129     0.0986    66
reciprocal, 0.4     0.0594    0.198     0.128     0.109     62
reciprocal, 0.5     0.06      0.197     0.129     0.142     172
reciprocal, 0.6     0.0603    0.196     0.128     0.21      20
reciprocal, 0.7     0.0612    0.195     0.128     0.349     45  <---
reciprocal, 0.8     0.0609    0.195     0.128     0.573     50
reciprocal, 0.9     0.0619    0.195     0.129     0.905     64
reciprocal, 1.0     0.0626    0.196     0.129     1.26      163
--------------------------------------------------------------------------------
denseweight, 0.1    0.0579    0.201     0.13      0.1       105
denseweight, 0.2    0.0579    0.202     0.13      0.0989    137
denseweight, 0.3    0.0581    0.201     0.13      0.0972    88
denseweight, 0.4    0.0582    0.201     0.13      0.0952    86
denseweight, 0.5    0.0582    0.201     0.13      0.0928    84
denseweight, 0.6    0.0582    0.201     0.13      0.0901    104
denseweight, 0.7    0.0584    0.201     0.129     0.0869    73   <---
denseweight, 0.8    0.0585    0.201     0.13      0.0831    70
denseweight, 0.9    0.0584    0.201     0.13      0.0783    130
denseweight, 1.0    0.0586    0.201     0.13      0.0731    113
denseweight, 1.1    0.0587    0.201     0.13      0.0728    134
denseweight, 1.2    0.0589    0.2       0.13      0.0736    59
denseweight, 1.3    0.0589    0.201     0.13      0.0743    119
denseweight, 1.4    0.0588    0.201     0.13      0.075     171
denseweight, 1.5    0.0591    0.2       0.13      0.0762    54
denseweight, 1.6    0.0591    0.201     0.13      0.077     101
denseweight, 1.7    0.0592    0.2       0.13      0.0778    71
denseweight, 1.8    0.0592    0.2       0.13      0.0785    90
denseweight, 1.9    0.0591    0.2       0.13      0.0794    79
denseweight, 2.0    0.0594    0.2       0.13      0.0804    4
--------------------------------------------------------------------------------
mdi, 0.1            0.0382    0.178     0.108     0.00777   14  <---      
mdi, 0.2            0.0346    0.197     0.116     0.00871   52        
mdi, 0.3            0.0343    0.202     0.118     0.0099    20        
mdi, 0.4            0.0332    0.195     0.114     0.0111    11        
mdi, 0.5            0.0337    0.195     0.114     0.0124    53        
mdi, 0.6            0.0336    0.2       0.117     0.0135    28        
mdi, 0.7            0.0331    0.2       0.116     0.0146    18        
mdi, 0.8            0.0329    0.2       0.116     0.0156    66        
mdi, 0.9            0.0329    0.197     0.115     0.0166    80        
mdi, 1.0            0.0322    0.194     0.113     0.0175    180       
mdi, 1.1            0.0322    0.195     0.114     0.0183    55        
mdi, 1.25           0.0322    0.196     0.114     0.0196    86        
mdi, 1.4            0.0328    0.2       0.116     0.0207    86        
mdi, 1.66           0.0321    0.199     0.116     0.0225    110       
mdi, 2              0.0321    0.195     0.114     0.0246    139       
mdi, 2.5            0.0321    0.201     0.116     0.0271    92        
mdi, 3.33           0.0324    0.201     0.117     0.0301    132       
mdi, 5              0.0322    0.198     0.115     0.0339    52        
mdi, 10             0.0327    0.198     0.115     0.0379    116
```
## Extended (frozen) w/ CME (3/19/26)

```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------
instance            0.0555    0.293     0.174     0.102     137
reciprocal, 0.1     0.0562    0.285     0.17      0.1       228
reciprocal, 0.2     0.061     0.23      0.146     0.0995    294
reciprocal, 0.3     0.061     0.228     0.144     0.104     212
reciprocal, 0.4     0.0611    0.226     0.143     0.121     215
reciprocal, 0.5     0.0617    0.225     0.143     0.163     256
reciprocal, 0.6     0.0627    0.226     0.144     0.257     153
reciprocal, 0.7     0.0625    0.221     0.142     0.423     231
reciprocal, 0.8     0.0641    0.219     0.141     0.707     345  <---
reciprocal, 0.9     0.0669    0.22      0.144     1.09      362
reciprocal, 1.0     0.0681    0.224     0.146     1.63      266
--------------------------------------------------------------------------------
denseweight, 0.1    0.0556    0.293     0.174     0.101     137
denseweight, 0.2    0.056     0.287     0.172     0.0995    157
denseweight, 0.3    0.0557    0.289     0.172     0.0979    162
denseweight, 0.4    0.0561    0.289     0.172     0.0964    165
denseweight, 0.5    0.0562    0.284     0.17      0.0945    201
denseweight, 0.6    0.0603    0.234     0.147     0.0918    290
denseweight, 0.7    0.0603    0.232     0.146     0.089     333
denseweight, 0.8    0.0603    0.233     0.147     0.0856    298
denseweight, 0.9    0.0604    0.235     0.148     0.0819    324
denseweight, 1.0    0.0603    0.234     0.147     0.0773    356
denseweight, 1.1    0.0608    0.229     0.145     0.0777    331
denseweight, 1.2    0.0608    0.231     0.146     0.0792    239
denseweight, 1.3    0.061     0.23      0.145     0.0809    271
denseweight, 1.4    0.0609    0.233     0.147     0.0817    343
denseweight, 1.5    0.0616    0.224     0.143     0.0833    178
denseweight, 1.6    0.0611    0.231     0.146     0.0844    321
denseweight, 1.7    0.0612    0.23      0.146     0.0859    282
denseweight, 1.8    0.0611    0.225     0.143     0.087     212
denseweight, 1.9    0.0614    0.222     0.142     0.0881    209  <---
denseweight, 2.0    0.0611    0.23      0.145     0.0898    218
--------------------------------------------------------------------------------
mdi, 0.1            0.308     0.261     0.284     0.0215    529
mdi, 0.2            0.0638    0.211     0.137     0.0224    238  <---
mdi, 0.3            0.0643    0.216     0.14      0.0276    203
mdi, 0.4            0.0614    0.224     0.143     0.032     250
mdi, 0.5            0.0614    0.234     0.148     0.0357    304
mdi, 0.6            0.0595    0.247     0.153     0.0389    188
mdi, 0.7            0.0598    0.231     0.145     0.0417    199
mdi, 0.8            0.06      0.228     0.144     0.0442    185
mdi, 0.9            0.058     0.231     0.145     0.0463    166
mdi, 1.0            0.0594    0.235     0.147     0.0484    202
mdi, 1.1            0.0591    0.245     0.152     0.0504    239
mdi, 1.25           0.0583    0.232     0.145     0.0531    187
mdi, 1.4            0.0591    0.228     0.144     0.0555    203
mdi, 1.66           0.0577    0.241     0.149     0.0592    131
mdi, 2              0.058     0.236     0.147     0.0632    153
mdi, 2.5            0.0584    0.246     0.152     0.0678    160
mdi, 3.33           0.0579    0.239     0.148     0.0732    151
mdi, 5              0.0583    0.243     0.15      0.0794    161
mdi, 10             0.0577    0.235     0.146     0.0861    13
```

## Extended (unfrozen)
```
alpha               MAE       MAE_r     AORE      val_loss  epochs    
--------------------------------------------------------------------------------mdi, 0.1            0.339     0.345     0.342     0.024     80        
mdi, 0.2            0.0824    0.279     0.181     0.0103    151       
mdi, 0.3            0.0578    0.208     0.133     0.0111    111 <---
mdi, 0.4            0.0763    0.339     0.207     0.00982   167       
mdi, 0.5            0.0768    0.354     0.215     0.0104    183       
mdi, 0.6            0.088     0.384     0.236     0.012     114       
mdi, 0.7            0.0583    0.334     0.196     0.0133    115
```

---

For all:
- AE is on
With AE off
- Regular best epochs with wPCC: $514$
- Regular best epochs without wPCC: $527$
With AE on:
- Regular best epochs with wPCC: $923$
- Regular best epochs without wPCC: $576$
## Decoupled (rep=-2)

| wPCC? | Weights           | Epochs | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----- | ----------------- | ------ | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Yes   | RI, $\alpha=0.7$  | 923/45 | 0.036           | 0.216             | 0.126            | 0.880         | 0.945           | 0.912          |
| No    | RI, $\alpha=0.7$  | 576/45 | 0.080           | 0.201             | 0.141            | 0.687         | 0.939           | 0.813          |
| Yes   | DW, $\alpha=0.7$  | 923/73 | 0.032           | 0.223             | 0.128            | 0.892         | 0.974           | 0.933          |
| No    | DW, $\alpha=0.7$  | 576/73 | 0.039           | 0.221             | 0.130            | 0.845         | 0.959           | 0.902          |
| Yes   | MDI, $\alpha=0.1$ | 923/14 | 0.142           | 0.230             | 0.187            | 0.905         | 0.974           | 0.939          |
| No    | MDI, $\alpha=0.1$ | 576/14 |                 |                   |                  |               |                 |                |
|       |                   |        |                 |                   |                  |               |                 |                |
## Extended (frozen)

| wPCC? | Weights           | Epochs  | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----- | ----------------- | ------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Yes   | RI, $\alpha=0.8$  | 923/345 | 0.039           | 0.220             | 0.130            | 0.849         | 0.936           | 0.893          |
| No    | RI, $\alpha=0.8$  | 576/345 | 0.111           | 0.216             | 0.163            | 0.602         | 0.932           | 0.767          |
| Yes   | DW, $\alpha=1.9$  | 923/209 | 0.030           | 0.215             | 0.122            | 0.913         | 0.971           | 0.942          |
| No    | DW, $\alpha=1.9$  | 576/209 | 0.064           | 0.216             | 0.140            | 0.779         | 0.970           | 0.875          |
| Yes   | MDI, $\alpha=0.2$ | 923/238 | 0.044           | 0.123             | 0.084            | 0.888         | 0.967           | 0.927          |
| Yes   | MDI, $\alpha=0.2$ | 923/238 | 0.032           | 0.112             | 0.072            | 0.921         | 0.974           | 0.947          |
| No    | MDI, $\alpha=0.2$ | 576/238 |                 |                   |                  |               |                 |                |
## Extended (unfrozen)

| wPCC? | Weights           | Epochs  | $MAE\downarrow$ | $MAE_R\downarrow$ | $AORE\downarrow$ | $PCC\uparrow$ | $PCC_R\uparrow$ | $AORC\uparrow$ |
| ----- | ----------------- | ------- | --------------- | ----------------- | ---------------- | ------------- | --------------- | -------------- |
| Yes   | RI, $\alpha=0.8$  | 923/345 |                 |                   |                  |               |                 |                |
| No    | RI, $\alpha=0.8$  | 576/345 |                 |                   |                  |               |                 |                |
| Yes   | DW, $\alpha=1.9$  | 923/209 |                 |                   |                  |               |                 |                |
| No    | DW, $\alpha=1.9$  | 576/209 |                 |                   |                  |               |                 |                |
| Yes   | MDI, $\alpha=0.3$ | 923/111 | 0.060           | 0.302             | 0.181            | 0.830         | 0.977           | 0.903          |
| No    | MDI, $\alpha=0.3$ | 576/111 |                 |                   |                  |               |                 |                |
## Tasks:
- `SDODataset` (see 'tutorials for `imbal`' email) $\checkmark$
	- One image per sample (most frequent type of images from 10min before predictions) $\checkmark$
		- Opted instead for all images of from 10min before $\checkmark$
	- Normalize pixel values from 0-1 $\checkmark$
- For tutorials, three sections $\checkmark$
	- Description of what we are doing $\checkmark$
	- Source code $\checkmark$
	- Expected output screenshots $\checkmark$
	- For next week: Section 1 from 'tutorials for `imbal`' email complete $\checkmark$
- Notes from emails (3/12) $\checkmark$
	- `model.best_sample_weights`, `model.best_class_weights`, and `model.best_metric_threshold` $\checkmark$
	- Step size of $0.1$ for threshold testing $\checkmark$
- For class weights in multi-fit, print out at `verbose_imbal>1`. $\checkmark$
- See "more notes from 3/17" $\checkmark$
	- On `verbose_imbal` $\checkmark$
		- For classification, print the index and class weights; if more than 5, print the first 5 followed by ...  $\checkmark$
		- For regression, print the index and first 5 sample weights followed by ... $\checkmark$
## Tasks
- Test how many SDO data samples make training time "reasonable". Prune dataset such that only that many data samples are available. (1000 train, 300 test?) $\checkmark$
	- Any thing that can be pre-processed in SDO should be $\checkmark$
- Plot full SDO distribution... is there a big gap in the middle? Maybe use stratified sampling to pick 1000 samples from across the distribution. $\checkmark$
- Ask Daniel what sections he has in his tutorials. Make sure they are relatively the same (in terms of steps) $\checkmark$
- Number the steps of the tutorial $\checkmark$
- Add inline comments and more section descriptions to tutorial $\checkmark$
- Data and Results Visualization $\rightarrow$ Probability Density Distribution and Results Visualization $\checkmark$
- Add "Necessary files" section before "Import Packages"
	- source file $\checkmark$
	- training data folder $\checkmark$
	- testing data folder $\checkmark$
- Add new page to documentation "Image regression on `SDOBenchmark`", which describes the pre-processing done on the dataset, as well as linking to relevant tutorials $\checkmark$
- Lower priority: Make `imbal` pip-installable
- For later:
	- We will add one more visualization for `imbal` if there is time: `GradCam`

---
# 3/26/26

## Tasks
- Double check documentation for `class_weights` in `imbal.classification.Model.balanced_fit`. The wording seems to be butchered. $\checkmark$
- Documentation should reflect that if neither class nor sample weights are provided to `imbal.classification.Model.balanced_fit`, equal class weighting is used. $\checkmark$
- `imbal.regression.Model.balanced_fit` `sample_weights` description is wrong $\checkmark$
- `imbal.regression.Model.balanced_fit` `sample_densities` description should say that reciprocal importance is used when densities are provided. $\checkmark$
- Ensure the following is clear in the documentation: $\checkmark$
![[Pasted image 20260326161401.png|400]]
- Get rid of redundant descriptions at top of tutorial files $\checkmark$
- Get rid of `representation_layer=-2` for decoupled tutorials $\checkmark$
- in TSNE documentation... $\checkmark$
	- PyTorch to TensorFlow $\checkmark$
	- Representation layer should say -2 $\checkmark$
	- For perplexity, add that the suggested range from the original paper is 5-50 (and link the paper) $\checkmark$
- Update outputs within tutorials (oversight on my part, ran out of time to fix it) $\checkmark$
- Why does `rRT_fit` run OOM? $\checkmark$
- Reduce SDO from 1000/300 to 500/100 $\checkmark$
