import os
import tensorflow as tf
import tools
import numpy as np
import imbal
import gc

MODE = 'E'

if MODE == 'C':
    PATH = 'models-sep-e'
    OUTPUT_PATH = 'results-sep-e/true-vs-predicted'
    DATA_PATH = 'cleaned-dtw-SEP-EC-data'
    DATA_PREFIX = 'sep_e_log_normalized'
    data_load_function = tools.load_sep_ec_data
else:
    PATH = 'models-sep-c'
    OUTPUT_PATH = 'results-sep-c/true-vs-predicted'
    DATA_PATH = 'cleaned-SEP-C-data'
    DATA_PREFIX = 'sep_c'
    data_load_function = tools.load_sep_c_data


(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_load_function(
    f"{DATA_PATH}/{DATA_PREFIX}",
)

y_test = y_test.reshape(-1)
print(np.min(y_test), np.max(y_test))
output_path_first = OUTPUT_PATH.split('/')[0]

outputs = []

count = 0
for index, filename in enumerate(os.listdir(PATH)):

    if 'normalized' not in filename:
        continue
    if '1' not in filename:
        continue

    file_prefix = filename.split('.')[0][:-2]

    epochs = []
    maes = []
    mae_rs = []
    aores = []
    for i in range(5):
        model_name = f'{file_prefix}_{i+1}'
        model = tf.keras.models.load_model(
            os.path.join(PATH, f'{model_name}.keras'),
            compile=False
        )

        with open(f'{output_path_first}/log_{model_name}.txt') as f:
            lines = f.readlines()
            epochs.append(float(lines[-1].split(',')[0]))


        test_predictions = model.predict(x_test).reshape(-1)

        common_predictions = test_predictions[y_test < np.log(10)]
        rare_predictions = test_predictions[y_test >= np.log(10)]
        y_test_common = y_test[y_test < np.log(10)]
        y_test_rare = y_test[y_test >= np.log(10)]

        common_mae = np.mean(np.abs(common_predictions - y_test_common))
        rare_mae = np.mean(np.abs(rare_predictions - y_test_rare))
        overall_mae = np.mean(np.abs(test_predictions - y_test))

        aore = (common_mae + rare_mae) / 2

        maes.append(overall_mae)
        mae_rs.append(rare_mae)
        aores.append(aore)

        gc.collect()

    outputs.append(file_prefix)
    outputs.append(f'{round(float(sum(epochs))/5, 4)} {round(float(sum(maes))/5,4)} {round(float(sum(mae_rs))/5,4)} {round(float(sum(aores))/5,4)}')
    print(outputs[-2])
    print(outputs[-1])


print()
print()
print()
for line in outputs:
    print(line)
