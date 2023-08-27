import os
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

DATASET_PATH = '/workspaces/data_sets/fake_audio/generated_audio'

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels


def load_dataset():
    data_dir = pathlib.Path(DATASET_PATH)
    print(data_dir)
    print("Loading...")
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    # labels=None,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')

    label_names = np.array(train_ds.class_names)
    print()
    print("label names:", label_names)
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    for example_audio, example_labels in train_ds.take(1):  
        print(example_audio)
        print(example_labels.shape)
        plt.figure(figsize=(16, 10))
        rows = 3
        cols = 3
        n = rows * cols
        for i in range(n):
          plt.subplot(rows, cols, i+1)
          audio_signal = example_audio[i]
          plt.plot(audio_signal)
          plt.title(label_names[example_labels[i]])
          plt.yticks(np.arange(-1.2, 1.2, 0.2))
          plt.ylim([-1.1, 1.1])
          plt.savefig('test.png')


if __name__ == "__main__":   
    load_dataset()