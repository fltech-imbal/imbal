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
Round 1 (local)
"""

# results.append("Cauchy-Schwartz w/ learned ratio:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_cauchy_schwartz_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=cauchy_schwartz,
#         UNIT_REPRESENTATIONS=False,
#         INCLUDE_GLOBAL_ANCHOR=False
#     ))
#     formatted_print(results)
#     gc.collect()
#
# results.append("Entropy w/ learned ratio:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_entropy_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=maximize_entropy,
#         UNIT_REPRESENTATIONS=False,
#         INCLUDE_GLOBAL_ANCHOR=False
#     ))
#     formatted_print(results)
#     gc.collect()
#
# results.append("Variance w/ learned ratio:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_variance_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=minimize_variance,
#         UNIT_REPRESENTATIONS=False,
#         INCLUDE_GLOBAL_ANCHOR=False
#     ))
#     formatted_print(results)
#     gc.collect()
#
# results.append("Distance PCC w/ learned ratio:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_distance_pcc_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=distance_pcc,
#         UNIT_REPRESENTATIONS=False,
#         INCLUDE_GLOBAL_ANCHOR=False
#     ))
#     formatted_print(results)
#     gc.collect()

results.append("Distance Difference w/ implied learned ratio:")
for i in range(5):
    print(f"Beginning run {i+1}...")
    results.append(run_model(
        LEARNABLE_RATIO=True,
        OUTPUT_POSTFIX=f'_distance_difference_learned_ratio_{i+1}',
        PROVIDED_REP_LOSS=distance_difference,
        UNIT_REPRESENTATIONS=False,
        INCLUDE_GLOBAL_ANCHOR=False,
        INCLUDE_RATIO=True,
        RATIO_LOSS_LAMBDA=0
    ))
    formatted_print(results)
    gc.collect()

# results.append("Cosine Similarity of 1 w/ learned ratio:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_cosine_similarity_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=cosine_similarity,
#         UNIT_REPRESENTATIONS=False,
#         INCLUDE_GLOBAL_ANCHOR=False
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()
#
results.append("Enforced Cosine w/ learned ratio:")
for i in range(5):
    print(f"Beginning run {i + 1}...")
    results.append(run_model(
        LEARNABLE_RATIO=True,
        OUTPUT_POSTFIX=f'_enforced_cosine_learned_ratio_{i + 1}',
        PROVIDED_REP_LOSS=enforced_cosine,
        UNIT_REPRESENTATIONS=False,
        USE_RAW_LABELS_AND_REPRESENTATIONS=True,
        INCLUDE_GLOBAL_ANCHOR=False
    ))
    for result in results:
        if isinstance(result, tuple):
            print('-', *result)
        else:
            print(result)
    gc.collect()

print()
formatted_print(results)