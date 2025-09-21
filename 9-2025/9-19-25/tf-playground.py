import tensorflow as tf
# List available physical devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
# If GPUs are found, check if TensorFlow is built with CUDA/ROCm support
if len(physical_devices) > 0:
    print("TensorFlow is built with GPU support:", tf.test.is_built_with_cuda())


#########################################################
# Import packages
#########################################################

import numpy as np
import keras
from keras import layers
from tensorflow.keras.datasets import fashion_mnist
import imbal.metrics as metrics
from imbal.experimental import OptimizeConfusionMetricCallback

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

y_train = np.reshape(np.array([1 if y_train[i] == 0 else 0 for i in range(len(y_train))]), (-1, 1))

y_test = np.reshape(np.array([1 if y_test[i] == 0 else 0 for i in range(len(y_test))]), (-1, 1))

uniques, counts = np.unique(np.reshape(y_train, (-1,)), return_counts=True)

print(uniques, counts)

#########################################################
# Build Model
#########################################################

inputs = keras.Input(shape=(28,28))
flatten = layers.Flatten()(inputs)
hidden = layers.Dense(32, activation='relu')(flatten)
output = layers.Dense(1, activation='sigmoid')(hidden)

model = keras.Model(inputs=inputs, outputs=output)

# FOR DEBUG USE ONLY
# tf.config.run_functions_eagerly(True)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=["accuracy",
                       'tpr',
                       'false_positive_rate',
                       'tnr',
                       'expected_TP',
                       'expected_tn',
                       'EC',
                       'tss',
                       'hss',
                       'gs',
                       'critical_success_index',
                       metrics.BoundedAUC(num_thresholds=1000, x_max=0.01)])

from imbal.stratified_sampling import DatasetWithBatching
from imbal.sample_weighting import generate_classification_weights

sampler = DatasetWithBatching(
    x_train,
    y_train,
    batch_size=512,
    sample_weights=generate_classification_weights(y_train),
)

history = model.fit(sampler, epochs=20,
                    callbacks=[OptimizeConfusionMetricCallback()]
                    )
test_scores = model.evaluate(x_test, y_test, verbose=1, batch_size=250)


print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
print("Test TPR:", test_scores[2])
print("Test FPR:", test_scores[3])
print("Test TNR:", test_scores[4])
print("Test Expected TP:", test_scores[5])
print("Test Expected TN:", test_scores[6])
print("Test Expected Correct:", test_scores[7])
print("Test TSS:", test_scores[8])
print("Test HSS:", test_scores[9])
print("Test GS:", test_scores[10])
print("Test CSI:", test_scores[11])
print("Test Limited AUC:", test_scores[12])
