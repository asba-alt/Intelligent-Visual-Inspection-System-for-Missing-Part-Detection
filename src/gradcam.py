"""Grad-CAM implementation for Keras models (MobileNetV2 last conv layer).

Generates a heatmap and overlays on original image using OpenCV.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer(model):
    # Find the last Conv2D layer in the model
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError('No Conv2D layer found in model')


def make_gradcam(image_path, model, out_path, last_conv_layer_name=None, alpha=0.4):
    # Read original image (for overlaying)
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f'Cannot read image {image_path}')
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = orig.shape[:2]

    img = Image.open(image_path).convert('RGB')
    img = img.resize((model.input_shape[1], model.input_shape[2]))
    x = np.asarray(img).astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        heatmap = heatmap
    else:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, alpha, orig, 1 - alpha, 0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay)
    return out_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python -m src.gradcam PATH_TO_IMAGE OUT_PATH')
        sys.exit(1)
    import tensorflow as tf
    model = tf.keras.models.load_model(sys.argv[2])
    make_gradcam(sys.argv[1], model, sys.argv[3])
