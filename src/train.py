"""Training script for the inspection model.

Run with: python -m src.train
"""
import os
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from config import SEED, EPOCHS, BATCH_SIZE, BEST_MODEL_PATH, LOGS_DIR
from src.data import load_datasets
from src.model import build_model


def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    set_seed()
    train_ds, val_ds, _ = load_datasets()

    model = build_model()

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    tb_logdir = os.path.join(LOGS_DIR, ts)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(tb_logdir),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save final model copy
    final_path = os.path.join(os.path.dirname(BEST_MODEL_PATH), 'final_model.keras')
    model.save(final_path)
    print('Saved final model to', final_path)


if __name__ == '__main__':
    main()
