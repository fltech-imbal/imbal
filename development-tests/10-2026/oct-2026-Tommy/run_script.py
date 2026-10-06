from sep_e_regression import run_model
from tools.loss_functions import *
import gc


results = []

# results.append("Cauchy-Schwartz (learned ratio)")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_cauchy_schwartz_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=cauchy_schwartz_w_ratio_loss,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         print(result)
#     gc.collect()
#
# results.append("Entropy (learned ratio)")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_entropy_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=maximize_entropy_w_ratio_loss,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         print(result)
#     gc.collect()
#
#
# results.append("Variance (learned ratio)")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_variance_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=minimize_variance_w_ratio_loss,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         print(result)
#     gc.collect()
#
# results.append("Distance PCC (learned ratio)")
# for i in range(5):
#     print(f"Beginning run {i+1}...")
#     results.append(run_model(
#         LEARNABLE_RATIO=True,
#         OUTPUT_POSTFIX=f'_distance_pcc_learned_ratio_{i+1}',
#         PROVIDED_REP_LOSS=distance_pcc_w_ratio_loss,
#         UNIT_REPRESENTATIONS=False,
#     ))
#     for result in results:
#         print(result)
#     gc.collect()

results.append("Distance Difference (learned ratio)")
for i in range(5):
    print(f"Beginning run {i+1}...")
    results.append(run_model(
        LEARNABLE_RATIO=True,
        OUTPUT_POSTFIX=f'_distance_difference_learned_ratio_{i+1}',
        PROVIDED_REP_LOSS=distance_pcc_w_ratio_loss,
        UNIT_REPRESENTATIONS=False,
    ))
    for result in results:
        print(result)
    gc.collect()

results.append("Cosine Similarity (learned ratio)")
for i in range(5):
    print(f"Beginning run {i+1}...")
    results.append(run_model(
        LEARNABLE_RATIO=True,
        OUTPUT_POSTFIX=f'_cosine_similarity_learned_ratio_{i+1}',
        PROVIDED_REP_LOSS=cosine_similarity_w_ratio_loss,
        UNIT_REPRESENTATIONS=False,
    ))
    for result in results:
        print(result)
    gc.collect()

print()
for result in results:
    print(result)