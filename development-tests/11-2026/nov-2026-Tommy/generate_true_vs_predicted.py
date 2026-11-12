import os
import tensorflow as tf
import tools
import numpy as np
import imbal

MODE = 'C'

if MODE == 'E':
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

count = 0
for index, filename in enumerate(os.listdir(PATH)):

    # if 'clustered' not in filename:
    #     continue
    count += 1
    if count <= 25:
        continue

    file_prefix = filename.split('.')[0]
    model = tf.keras.models.load_model(
        os.path.join(PATH, filename),
        compile=False
    )

    test_predictions = model.predict(x_test).reshape(-1)

    common_predictions = test_predictions[y_test < np.log(10)]
    rare_predictions = test_predictions[y_test >= np.log(10)]
    y_test_common = y_test[y_test < np.log(10)]
    y_test_rare = y_test[y_test >= np.log(10)]

    common_mae = np.mean(np.abs(common_predictions - y_test_common))
    rare_mae = np.mean(np.abs(rare_predictions - y_test_rare))
    overall_mae = np.mean(np.abs(test_predictions - y_test))

    aore = (common_mae + rare_mae) / 2

    imbal.regression.plot_true_vs_predictions(
        y_test,
        test_predictions,
        title=f"Common MAE: {float(common_mae):.4}, Rare MAE: {float(rare_mae):.4}, AORE: {float(aore):.4}",
        save_figure=f'{OUTPUT_PATH}/{file_prefix}-true-vs-predicted-aore-{float(aore):.4}.png',
    )


