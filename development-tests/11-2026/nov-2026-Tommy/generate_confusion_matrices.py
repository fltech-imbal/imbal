import os
import tensorflow as tf
import tools
import numpy as np
import imbal

PATH = 'models-sep-e'
OUTPUT_PATH = 'results-sep-e/confusion-matrices'
DATA_PATH = 'cleaned-dtw-SEP-EC-data'
DATA_PREFIX = 'sep_e_log_normalized'

data_load_function = tools.load_sep_ec_data

(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_load_function(
    f"{DATA_PATH}/{DATA_PREFIX}",
)

y_train = np.digitize(y_train, bins=[-np.log(10), np.log(10)])
y_test = np.digitize(y_test, bins=[-np.log(10), np.log(10)])
y_val = np.digitize(y_val, bins=[-np.log(10), np.log(10)])

for filename in os.listdir(PATH):
    file_prefix = filename.split('.')[0]
    model = tf.keras.models.load_model(
        os.path.join(PATH, filename),
        compile=False
    )

    test_predictions = model.predict(x_test)


    imbal.classification.plot_confusion_matrix(
        y_test,
        test_predictions,
        save_figure=f'{OUTPUT_PATH}/{file_prefix}.png',
    )


