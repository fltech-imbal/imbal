from keras import layers
import tensorflow as tf

class TrainableScalar(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        # inputs is only used to determine the batch size.
        # The actual input values are ignored.
        batch_size = tf.shape(inputs)[0]
        return tf.broadcast_to(self.weight, [batch_size, 1])