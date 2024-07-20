import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects, register_keras_serializable
from tensorflow.keras import backend as K

# Define the custom weighted binary crossentropy function
@register_keras_serializable(name="weighted_binary_crossentropy")
def weighted_binary_crossentropy(target, output, weights):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    epsilon_ = tf.constant(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities
    bce = weights[1] * target * tf.math.log(output + epsilon_)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return -bce

@register_keras_serializable(name="WeightedBinaryCrossentropy")
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weights=[1.0, 1.0], label_smoothing=0.0, name="weighted_binary_crossentropy"):
        super().__init__(name=name)
        self.weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        return tf.reduce_mean(weighted_binary_crossentropy(y_true, y_pred, self.weights), axis=-1)

    def get_config(self):
        config = {
            "weights": self.weights.numpy().tolist(),
            "label_smoothing": self.label_smoothing,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        weights = tf.convert_to_tensor(config.pop("weights"), dtype=tf.float32)
        return cls(weights=weights, **config)
