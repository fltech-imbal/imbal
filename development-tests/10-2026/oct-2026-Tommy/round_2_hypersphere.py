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

# results.append("Cauchy-Schwartz w/ unit representations:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         FIXED_RATIO=1,
#         OUTPUT_POSTFIX=f'_cauchy_schwartz_hypersphere_{i+1}',
#         PROVIDED_REP_LOSS=cauchy_schwartz,
#         DECORRELATION=False,
#         UNIT_REPRESENTATIONS=True,
#         INCLUDE_GLOBAL_ANCHOR=False
#     ))
#     formatted_print(results)
#     gc.collect()
#
# results.append("Entropy w/ unit representations:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         FIXED_RATIO=1,
#         OUTPUT_POSTFIX=f'_entropy_hypersphere_{i+1}',
#         PROVIDED_REP_LOSS=maximize_entropy,
#         DECORRELATION=False,
#         UNIT_REPRESENTATIONS=True,
#         INCLUDE_GLOBAL_ANCHOR=False
#     ))
#     formatted_print(results)
#     gc.collect()
#
# results.append("Distance PCC w/ unit representations:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         FIXED_RATIO=1,
#         OUTPUT_POSTFIX=f'_distance_pcc_hypersphere_{i+1}',
#         PROVIDED_REP_LOSS=distance_pcc,
#         DECORRELATION=False,
#         UNIT_REPRESENTATIONS=True,
#         INCLUDE_GLOBAL_ANCHOR=False
#     ))
#     formatted_print(results)
# #     gc.collect()
#
# results.append("Distance Difference w/ unit representations:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         FIXED_RATIO=1,
#         OUTPUT_POSTFIX=f'_distance_difference_hypersphere_{i+1}',
#         PROVIDED_REP_LOSS=distance_difference,
#         DECORRELATION=False,
#         UNIT_REPRESENTATIONS=True,
#         INCLUDE_GLOBAL_ANCHOR=False,
#         INCLUDE_RATIO=True
#     ))
#     formatted_print(results)
#     gc.collect()

# results.append("Cosine Similarity of 1 w/ global anchor, hypersphere:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         OUTPUT_POSTFIX=f'_cosine_similarity_global_hypersphere_{i+1}',
#         PROVIDED_REP_LOSS=cosine_similarity,
#         UNIT_REPRESENTATIONS=True,
#         INCLUDE_GLOBAL_ANCHOR=True,
#         DECORRELATION=False,
#         FIXED_RATIO=1
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()

results.append("Enforced Cosine w/ global anchor, hypersphere:")
for i in range(5):
    print(f"Beginning run {i + 1}...")
    results.append(run_model(
        LEARNABLE_RATIO=False,
        OUTPUT_POSTFIX=f'_enforced_cosine_global_hypersphere_{i + 1}',
        PROVIDED_REP_LOSS=enforced_cosine,
        UNIT_REPRESENTATIONS=True,
        USE_RAW_LABELS_AND_REPRESENTATIONS=True,
        INCLUDE_GLOBAL_ANCHOR=True
    ))
    for result in results:
        if isinstance(result, tuple):
            print('-', *result)
        else:
            print(result)
    gc.collect()

print()
formatted_print(results)