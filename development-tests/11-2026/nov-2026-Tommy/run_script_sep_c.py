from conflicting_gradients_sep_c import run_model
from gradient_conflict_tools import *

for i in range(5):
    run_model(
        EPOCHS=10000,
        EARLY_STOPPING_PATIENCE=25,
        OUTPUT_POSTFIX=f'baseline_{i+1}',
        CONFLICT_REDUCTION_METHOD=None,
        BATCH_SIZE=512
    )

for i in range(5):
    run_model(
        EPOCHS=10000,
        EARLY_STOPPING_PATIENCE=25,
        OUTPUT_POSTFIX=f'two_fists_one_heart_{i+1}',
        CONFLICT_REDUCTION_METHOD=two_fists_one_heart,
        BATCH_SIZE=512
    )

for i in range(5):
    run_model(
        EPOCHS=10000,
        EARLY_STOPPING_PATIENCE=25,
        OUTPUT_POSTFIX=f'ts_mof_{i+1}',
        CONFLICT_REDUCTION_METHOD=ts_mof,
        BATCH_SIZE=512
    )

for i in range(5):
    run_model(
        EPOCHS=10000,
        EARLY_STOPPING_PATIENCE=25,
        OUTPUT_POSTFIX=f'conflict_only_end_sum_{i + 1}',
        CONFLICT_REDUCTION_METHOD=conflict_only_end_sum,
        BATCH_SIZE=512
    )

for i in range(5):
    run_model(
        EPOCHS=10000,
        EARLY_STOPPING_PATIENCE=25,
        OUTPUT_POSTFIX=f'conflict_only_end_sum_with_cosine_{i + 1}',
        CONFLICT_REDUCTION_METHOD=conflict_only_end_sum_with_cosine,
        BATCH_SIZE=512
    )