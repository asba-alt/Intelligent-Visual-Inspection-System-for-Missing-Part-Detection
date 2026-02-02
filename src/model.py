"""Model building utilities using MobileNetV2 transfer learning."""
import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMAGE_SIZE


def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), train_base=False):
    """Create a MobileNetV2-based binary classifier with sigmoid output."""
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    base.trainable = bool(train_base)

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')],
    )
    return model
