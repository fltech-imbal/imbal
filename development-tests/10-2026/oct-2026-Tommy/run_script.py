from sep_e_regression import run_model
from tools.loss_functions import *
import gc


results = []
for i in range(5):
    print(f"Beginning run {i+1}...")
    results.append(run_model(
        LEARNABLE_RATIO=False,
        OUTPUT_POSTFIX=f'_enforced_cosine_{i+1}',
        PROVIDED_REP_LOSS=enforced_cosine
    ))
    gc.collect()

for result in results:
    print(result)