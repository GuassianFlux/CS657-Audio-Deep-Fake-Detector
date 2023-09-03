import os
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128
    )

    # Obtain the magnitude of the STFT
    spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used as image-like input data with convolution layers
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert frequencies to log scale and transpose, so that the time is represented on x-axis
    # Add an epsilon to avoid taking a log of zero
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    x = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    y = range(height)
    ax.pcolormesh(x,y,log_spec)

class Data_Processor:
    def __init__(self, dataFilePath):
        self.dataFilePath = dataFilePath

    def loadDataset(self):
        data_dir = pathlib.Path(self.dataFilePath)
        print("Loading", self.dataFilePath)
        self.train_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=64,
        validation_split=0,
        seed=0,
        output_sequence_length=16000)

    def makeSpectrogram(self, outputFilename):
        label_names = np.array(self.train_ds.class_names)
        train_ds = self.train_ds.map(squeeze, tf.data.AUTOTUNE)
        for example_audio, example_labels in train_ds.take(1):
            for i in range(3):
                label = label_names[example_labels[i]]
                waveform = example_audio[i]
                spectrogram = get_spectrogram(waveform)
                fig, axes = plt.subplots(2,1)
                timescale = np.arange(waveform.shape[0])
                axes[0].plot(timescale, waveform.numpy())
                axes[0].set_title('Waveform')
                axes[0].set_xlim([0, 16000])
                plot_spectrogram(spectrogram.numpy(), axes[1])
                axes[1].set_title('Spectrogram')
                plt.suptitle(label.title())
                plt.savefig(outputFilename+'_'+str(i)+'.png')

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