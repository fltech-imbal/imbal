from updated_sep_e_regression import run_model
from tools.updated_loss_functions import *
import gc

# tf.config.run_functions_eagerly(True)

def formatted_print(array):
    for value in array:
        if isinstance(value, tuple):
            print('-', *value)
        else:
            print(value)

results = []

"""
Round 2 (decorrelation)
"""

results.append("Distance Difference w/ implied learned ratio, decorrelation, nonlinear regressor:")
for i in range(5):
    print(f"Beginning run {i+1}...")
    results.append(run_model(
        LEARNABLE_RATIO=True,
        OUTPUT_POSTFIX=f'_distance_difference_learned_ratio_decorrelation_nonlinear_regressor_{i+1}',
        PROVIDED_REP_LOSS=distance_difference,
        NONLINEAR_REGRESSOR=True,
        INCLUDE_GLOBAL_ANCHOR=False,
        INCLUDE_RATIO=True,
        RATIO_LOSS_LAMBDA=0,
        DECORRELATION=True
    ))
    formatted_print(results)
    gc.collect()

results.append("Cosine Similarity of 1 w/ global anchor, fixed ratio=1, nonlinear regressor:")
for i in range(5):
    print(f"Beginning run {i+1}...")
    results.append(run_model(
        OUTPUT_POSTFIX=f'_cosine_similarity_global_hypersphere_{i+1}',
        PROVIDED_REP_LOSS=cosine_similarity,
        NONLINEAR_REGRESSOR=True,
        INCLUDE_GLOBAL_ANCHOR=True,
        FIXED_RATIO=1
    ))
    for result in results:
        if isinstance(result, tuple):
            print('-', *result)
        else:
            print(result)
    gc.collect()

results.append("Enforced Cosine w/ global anchor, hypersphere:")
for i in range(5):
    print(f"Beginning run {i + 1}...")
    results.append(run_model(
        LEARNABLE_RATIO=False,
        OUTPUT_POSTFIX=f'_enforced_cosine_global_hypersphere_{i + 1}',
        PROVIDED_REP_LOSS=enforced_cosine,
        UNIT_REPRESENTATIONS=False,
        NONLINEAR_REGRESSOR=True,
        USE_RAW_LABELS_AND_REPRESENTATIONS=True,
        INCLUDE_GLOBAL_ANCHOR=True,
        RATIO_LOSS_LAMBDA=0,
    ))
    for result in results:
        if isinstance(result, tuple):
            print('-', *result)
        else:
            print(result)
    gc.collect()

print()
formatted_print(results)

