import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from config import IMG_SIZE


def build_model(img_size=IMG_SIZE, pretrained=True):
    base = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet' if pretrained else None)
    base.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model


def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
