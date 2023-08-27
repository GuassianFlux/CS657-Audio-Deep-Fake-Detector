import os
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

class Data_Processor:
    def __init__(self, dataFilePath):
        self.dataFilePath = dataFilePath

    def loadDataset(self):
        data_dir = pathlib.Path(self.dataFilePath)
        print("Loading", self.dataFilePath)
        self.train_ds, self.val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=64,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16000,
        subset='both')

    def plotFirstSample(self, outputFilename):
        label_names = np.array(self.train_ds.class_names)
        train_ds = self.train_ds.map(squeeze, tf.data.AUTOTUNE)
        for example_audio, example_labels in train_ds.take(1):  
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
                plt.savefig(outputFilename)