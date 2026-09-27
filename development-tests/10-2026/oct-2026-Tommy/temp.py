from tools import FitType, load_sep_c_data, load_sep_ec_data, generate_plots, build_sep_ec_model, generate_weights, pcc


DATA_PATH = 'cleaned-SEP-C-data'
DATA_PREFIX = 'sep_c'
VALUE_MIN = -1.61
VALUE_MAX = 8.75
VAL_DELTA = 1e-6

"""
Load data
"""

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_c_data(
    f"{DATA_PATH}/{DATA_PREFIX}"
)

# DATA_PATH = 'cleaned-dtw-SEP-EC-data'
# DATA_PREFIX = 'sep_e_log_normalized'
# USE_DELTA = False
# VALUE_MIN = -10
# VALUE_MAX = 5.5
# VAL_DELTA = 1e-6
#
# """
# Load data
# """
#
# (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_sep_ec_data(
#     f"{DATA_PATH}/{DATA_PREFIX}"
# )



print(x_train.shape, x_val.shape, x_test.shape)