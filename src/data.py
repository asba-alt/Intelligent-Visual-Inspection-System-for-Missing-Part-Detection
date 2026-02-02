"""Dataset loading utilities."""
import os
import tensorflow as tf
from config import DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, SEED


def load_datasets(batch_size=BATCH_SIZE, img_size=IMAGE_SIZE, seed=SEED):
    """Load train, val, test datasets using image_dataset_from_directory.

    Expects DATASET_DIR to have `train/ val/ test/` subfolders, each with
    `complete/` and `defect/` subfolders.
    """
    def _ds(sub):
        path = os.path.join(DATASET_DIR, sub)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Dataset folder not found: {path}")
        return tf.keras.utils.image_dataset_from_directory(
            path,
            labels='inferred',
            label_mode='binary',
            image_size=(img_size, img_size),
            shuffle=(sub != 'test'),
            seed=seed,
            batch_size=batch_size,
        )

    train_ds = _ds('train')
    val_ds = _ds('val')
    test_ds = _ds('test')

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
