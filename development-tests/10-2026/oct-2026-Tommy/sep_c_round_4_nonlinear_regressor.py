from updated_sep_c_regression import run_model
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
Round 4 (nonlinear regressor)
"""

results.append("Distance PCC w/ global anchor, nonlinear regressor:")
for i in range(5):
    print(f"Beginning run {i + 1}...")
    results.append(run_model(
        OUTPUT_POSTFIX=f'_distance_pcc_global_no_ratio_nonlinear_regressor_{i + 1}',
        PROVIDED_REP_LOSS=distance_pcc,
        UNIT_REPRESENTATIONS=False,
        INCLUDE_GLOBAL_ANCHOR=True,
        NONLINEAR_REGRESSOR=True
    ))
    formatted_print(results)
    gc.collect()

results.append("Entropy w/ global anchor, fixed ratio=1, nonlinear regressor:")
for i in range(5):
    print(f"Beginning run {i + 1}...")
    results.append(run_model(
        FIXED_RATIO=1,
        OUTPUT_POSTFIX=f'_entropy_global_nonlinear_regressor_{i + 1}',
        PROVIDED_REP_LOSS=maximize_entropy,
        UNIT_REPRESENTATIONS=False,
        INCLUDE_GLOBAL_ANCHOR=True,
        NONLINEAR_REGRESSOR=True
    ))
    formatted_print(results)
    gc.collect()

results.append("Distance Difference, implied ratio=1, nonlinear regressor:")
for i in range(5):
    print(f"Beginning run {i + 1}...")
    results.append(run_model(
        FIXED_RATIO=1,
        OUTPUT_POSTFIX=f'_distance_difference_nonlinear_regressor_{i + 1}',
        PROVIDED_REP_LOSS=distance_difference,
        UNIT_REPRESENTATIONS=False,
        INCLUDE_GLOBAL_ANCHOR=False,
        INCLUDE_RATIO=True,
        NONLINEAR_REGRESSOR=True
    ))
    formatted_print(results)
    gc.collect()

# results.append("Variance w/ global anchor, fixed ratio=1, nonlinear regressor:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         FIXED_RATIO=1,
#         OUTPUT_POSTFIX=f'_variance_global_fixed_nonlinear_regressor_{i+1}',
#         PROVIDED_REP_LOSS=minimize_variance,
#         UNIT_REPRESENTATIONS=False,
#         NONLINEAR_REGRESSOR=True,
#         INCLUDE_GLOBAL_ANCHOR=True,
#     ))
#     formatted_print(results)
#     gc.collect()
#
# results.append("Distance PCC w/ global, nonlinear regressor:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         OUTPUT_POSTFIX=f'_distance_pcc_global_no_ratio_nonlinear_regressor_{i+1}',
#         PROVIDED_REP_LOSS=distance_pcc,
#         UNIT_REPRESENTATIONS=False,
#         NONLINEAR_REGRESSOR=True,
#         INCLUDE_GLOBAL_ANCHOR=True,
#     ))
#     formatted_print(results)
#     gc.collect()
#
# results.append("Enforced Cosine w/ global anchor, learned ratio, nonlinear regressor:")
# for i in range(5):
#     print(f"Beginning run {i + 1}...")
#     results.append(run_model(
#         OUTPUT_POSTFIX=f'_enforced_cosine_global_learned_ratio_nonlinear_regressor_{i + 1}',
#         PROVIDED_REP_LOSS=enforced_cosine,
#         UNIT_REPRESENTATIONS=False,
#         NONLINEAR_REGRESSOR=True,
#         USE_RAW_LABELS_AND_REPRESENTATIONS=True,
#         INCLUDE_GLOBAL_ANCHOR=True,
#         LEARNABLE_RATIO=True
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()

print()
formatted_print(results)

