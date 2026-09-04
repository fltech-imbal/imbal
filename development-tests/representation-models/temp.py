import glob
import tensorflow as tf
import imbal

files = glob.glob('models/*')

def stub_function(y_true, y_pred, weights=None):
    return 0

x_test = tf.clip_by_value(tf.random.normal((250, 2)) + tf.random.normal((250, 2), stddev=0.1), clip_value_min=-3, clip_value_max=3)
x_test = x_test.numpy()
y_test = tf.reshape(tf.linalg.norm(x_test, axis=1), (-1, 1))
y_test = y_test.numpy()

for index, file in enumerate(files):
    model = tf.keras.models.load_model(
        file,
        custom_objects={
            # 'loss_fn': loss_fn,
            # 'Model' : imbal.regression.Model,
            'stub_function': stub_function,
        }
    )
    model.summary()
    print(file, index)
    # if index < 15:
    #     continue
    if not 'fine' in file:
        continue
    disallowed = [1,8]

    predictions = model.predict(x_test)

    imbal.regression.plot_true_vs_predictions(
        y_test,
        predictions,
        save_figure=f'results/tvp/{file[7:-6]}.png'
    )

    if index not in disallowed:
        imbal.regression.tsne_visualization(
            model,
            x_test,
            y_test,
            save_figure=f'results/tsne/{file[7:-6]}.png'
        )
