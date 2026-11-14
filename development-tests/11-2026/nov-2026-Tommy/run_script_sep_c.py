from conflicting_gradients_sep_c import run_model
from gradient_conflict_tools import *
import gc

# """
# Label-based binning
# """
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'baseline_{i+1}',
#         CONFLICT_REDUCTION_METHOD=None,
#         BATCH_SIZE=512
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'two_fists_one_heart_{i+1}',
#         CONFLICT_REDUCTION_METHOD=two_fists_one_heart,
#         BATCH_SIZE=512
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'ts_mof_{i+1}',
#         CONFLICT_REDUCTION_METHOD=ts_mof,
#         BATCH_SIZE=512
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'end_sum_{i + 1}',
#         CONFLICT_REDUCTION_METHOD=end_sum,
#         BATCH_SIZE=512
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'end_sum_with_cosine_{i + 1}',
#         CONFLICT_REDUCTION_METHOD=end_sum_w_cosine,
#         BATCH_SIZE=512
#     )
#     gc.collect()
#
# """
# Cluster-based binning
# """
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'clustered_two_fists_one_heart_{i+1}',
#         CONFLICT_REDUCTION_METHOD=two_fists_one_heart,
#         BATCH_SIZE=512,
#         GRADIENT_BINNING_METHOD='clustering'
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'clustered_ts_mof_{i+1}',
#         CONFLICT_REDUCTION_METHOD=ts_mof,
#         BATCH_SIZE=512,
#         GRADIENT_BINNING_METHOD='clustering'
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'clustered_end_sum_{i + 1}',
#         CONFLICT_REDUCTION_METHOD=end_sum,
#         BATCH_SIZE=512,
#         GRADIENT_BINNING_METHOD='clustering'
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'clustered_end_sum_with_cosine_{i + 1}',
#         CONFLICT_REDUCTION_METHOD=end_sum_w_cosine,
#         BATCH_SIZE=512,
#         GRADIENT_BINNING_METHOD='clustering'
#     )
#     gc.collect()

"""
Label-based binning (normalized gradients)
"""

# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'normalized_two_fists_one_heart_{i+1}',
#         CONFLICT_REDUCTION_METHOD=two_fists_one_heart,
#         BATCH_SIZE=512,
#         NORMALIZE_GRADIENTS_BEFORE_REDUCTION=True
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'normalized_ts_mof_{i+1}',
#         CONFLICT_REDUCTION_METHOD=ts_mof,
#         BATCH_SIZE=512,
#         NORMALIZE_GRADIENTS_BEFORE_REDUCTION=True
#     )
#     gc.collect()

# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'normalized_end_sum_{i + 1}',
#         CONFLICT_REDUCTION_METHOD=end_sum,
#         BATCH_SIZE=512,
#         NORMALIZE_GRADIENTS_BEFORE_REDUCTION=True
#     )
#     gc.collect()
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'normalized_end_sum_with_cosine_{i + 1}',
#         CONFLICT_REDUCTION_METHOD=end_sum_w_cosine,
#         BATCH_SIZE=512,
#         NORMALIZE_GRADIENTS_BEFORE_REDUCTION=True
#     )
#     gc.collect()
#
# """
# Cluster-based binning (normalized gradients)
# """
#
# for i in range(5):
#     run_model(
#         EPOCHS=10000,
#         EARLY_STOPPING_PATIENCE=25,
#         EARLY_STOPPING_DELTA=1e-4,
#         OUTPUT_POSTFIX=f'clustered_normalized_two_fists_one_heart_{i+1}',
#         CONFLICT_REDUCTION_METHOD=two_fists_one_heart,
#         BATCH_SIZE=512,
#         GRADIENT_BINNING_METHOD='clustering',
#         NORMALIZE_GRADIENTS_BEFORE_REDUCTION=True
#     )
#     gc.collect()

for i in range(5):
    run_model(
        EPOCHS=10000,
        EARLY_STOPPING_PATIENCE=25,
        EARLY_STOPPING_DELTA=1e-4,
        OUTPUT_POSTFIX=f'clustered_normalized_ts_mof_{i+1}',
        CONFLICT_REDUCTION_METHOD=ts_mof,
        BATCH_SIZE=512,
        GRADIENT_BINNING_METHOD='clustering',
        NORMALIZE_GRADIENTS_BEFORE_REDUCTION=True
    )
    gc.collect()

for i in range(5):
    run_model(
        EPOCHS=10000,
        EARLY_STOPPING_PATIENCE=25,
        EARLY_STOPPING_DELTA=1e-4,
        OUTPUT_POSTFIX=f'clustered_normalized_end_sum_{i + 1}',
        CONFLICT_REDUCTION_METHOD=end_sum,
        BATCH_SIZE=512,
        GRADIENT_BINNING_METHOD='clustering',
        NORMALIZE_GRADIENTS_BEFORE_REDUCTION=True
    )
    gc.collect()

for i in range(5):
    run_model(
        EPOCHS=10000,
        EARLY_STOPPING_PATIENCE=25,
        EARLY_STOPPING_DELTA=1e-4,
        OUTPUT_POSTFIX=f'clustered_normalized_end_sum_with_cosine_{i + 1}',
        CONFLICT_REDUCTION_METHOD=end_sum_w_cosine,
        BATCH_SIZE=512,
        GRADIENT_BINNING_METHOD='clustering',
        NORMALIZE_GRADIENTS_BEFORE_REDUCTION=True
    )
    gc.collect()