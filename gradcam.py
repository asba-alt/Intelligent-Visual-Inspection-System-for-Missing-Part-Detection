import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models
from config import IMG_SIZE, MODEL_PATH, RESULTS_DIR


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(original_img_bgr, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_img_bgr.shape[1], original_img_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(heatmap_colored, alpha, original_img_bgr, 1 - alpha, 0)
    return overlay


def run_gradcam(image_path, output_path=None, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = tf.keras.models.load_model(model_path)

    # find a reasonable last conv layer for MobileNetV2
    # common names include 'Conv_1' or 'block_16_project'
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        raise ValueError('Could not find a convolutional layer in model')

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to read {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlay = overlay_heatmap_on_image(img_bgr, heatmap)

    os.makedirs(os.path.dirname(output_path) if output_path else RESULTS_DIR, exist_ok=True)
    if not output_path:
        base = os.path.basename(image_path)
        output_path = os.path.join(RESULTS_DIR, f"gradcam_{base}")

    cv2.imwrite(output_path, overlay)
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--out', help='Output path for overlay image')
    args = parser.parse_args()
    out = run_gradcam(args.image, output_path=args.out)
    print('Saved Grad-CAM overlay to', out)
