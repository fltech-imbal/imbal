import os
import tensorflow as tf
import tools
import numpy as np
import imbal

MODE = 'E'

if MODE == 'E':
    PATH = 'models-sep-e'
    OUTPUT_PATH = 'results-sep-e/confusion-matrices'
    DATA_PATH = 'cleaned-dtw-SEP-EC-data'
    DATA_PREFIX = 'sep_e_log_normalized'
    data_load_function = tools.load_sep_ec_data
else:
    PATH = 'models-sep-c'
    OUTPUT_PATH = 'results-sep-c/confusion-matrices'
    DATA_PATH = 'cleaned-SEP-C-data'
    DATA_PREFIX = 'sep_c'
    data_load_function = tools.load_sep_c_data


(x_train, y_train), (x_val, y_val), (x_test, y_test) = data_load_function(
    f"{DATA_PATH}/{DATA_PREFIX}",
)

if MODE == 'E':
    y_train = np.digitize(y_train, bins=[-np.log(10), np.log(10)])
    y_test = np.digitize(y_test, bins=[-np.log(10), np.log(10)])
    y_val = np.digitize(y_val, bins=[-np.log(10), np.log(10)])
else:
    y_train = np.digitize(y_train, bins=[-1.609, np.log(10)])
    y_test = np.digitize(y_test, bins=[-1.609, np.log(10)])
    y_val = np.digitize(y_val, bins=[-1.609, np.log(10)])

print(np.min(y_test), np.max(y_test))

for filename in os.listdir(PATH):
    if 'two_fists' not in filename:
        continue
    file_prefix = filename.split('.')[0]
    model = tf.keras.models.load_model(
        os.path.join(PATH, filename),
        compile=False
    )

    test_predictions = model.predict(x_test)


    f1 = tf.keras.metrics.F1Score(
        average='macro',
        threshold=None
    )
    f1.update_state(tf.reshape(tf.one_hot(y_test, 3), [-1, 3]), test_predictions)

    test_predictions = np.argmax(test_predictions, axis=1)
    imbal.classification.plot_confusion_matrix(
        y_test,
        test_predictions,
        save_figure=f'{OUTPUT_PATH}/{file_prefix}-f1-{round(float(f1.result().numpy()), 3)}.png',
        square_font_size=20
    )


