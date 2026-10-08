from updated_sep_e_regression import run_model
from tools.updated_loss_functions import *
import gc

# tf.config.run_functions_eagerly(True)

results = []

# results.append("Cauchy-Schwartz (learned ratio):")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_cauchy_schwartz_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=cauchy_schwartz,
#         UNIT_REPRESENTATIONS=False
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()

# results.append("Entropy (learned ratio):")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_entropy_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=maximize_entropy,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()


# results.append("Variance (learned ratio):")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_variance_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=minimize_variance,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()

# results.append("Distance PCC (learned ratio):")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_distance_pcc_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=distance_pcc,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()

# results.append("Distance Difference (implicit learned ratio):")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         RATIO_LOSS_LAMBDA=0,
#         INCLUDE_RATIO=True,
#         OUTPUT_POSTFIX=f'_distance_difference_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=distance_difference,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()

# results.append("Cosine Similarity of 1:")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         OUTPUT_POSTFIX=f'_cosine_similarity_{i+1}',
#         PROVIDED_REP_LOSS=cosine_similarity,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         if isinstance(result, tuple):
#             print('-', *result)
#         else:
#             print(result)
#     gc.collect()

results.append("Enforced Cosine:")
for i in range(5):
    print(f"Beginning run {i + 1}...")
    results.append(run_model(
        LEARNABLE_RATIO=False,
        OUTPUT_POSTFIX=f'_enforced_cosine_hypersphere_{i + 1}',
        PROVIDED_REP_LOSS=enforced_cosine,
        UNIT_REPRESENTATIONS=True,
        USE_RAW_REPRESENTATIONS=True,
        INCLUDE_GLOBAL_ANCHOR=False
    ))
    for result in results:
        if isinstance(result, tuple):
            print('-', *result)
        else:
            print(result)
    gc.collect()

print()
for result in results:
    if isinstance(result, tuple):
        print('-', *result)
    else:
        print(result)