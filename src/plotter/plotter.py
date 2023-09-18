############################################
# Put this at the top of each file before 
# importing TensorFlow to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################################
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

SAMPLE_MAX=32768.0

def normalization(sample):
    return float(sample) / SAMPLE_MAX

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

class Plotter:
    @staticmethod
    def plot_dual_wave_spec(waveform, label):
        # Normalize and preprocess data
        wav = list(map(normalization,waveform))
        spectrogram = get_spectrogram(wav)

        # Initlialize plot
        fig, axes = plt.subplots(2,figsize=(16, 9))

        # Plot waveform
        axes[0].plot(wav)
        axes[0].set_title('Waveform')

        # Plot spectrogram
        plot_spectrogram(spectrogram.numpy(), axes[1])
        axes[1].set_title('Spectrogram')
        
        # Export plot
        plt.suptitle(label)
        plt.savefig(label + '_dual_plot.png')

    @staticmethod
    def plot_spectrogram(waveform, label):
        # Normalize and preprocess data
        wav = list(map(normalization,waveform))
        spectrogram = get_spectrogram(wav)
        
        # Initlialize plot
        fig, ax = plt.subplots(figsize=(16, 9))

        # Plot spectrogram
        plot_spectrogram(spectrogram.numpy(), ax)

        # Export plot
        ax.set_title(label)
        plt.savefig(label + '_spectrogram_plot.png')
    
    @staticmethod
    def plot_waveform(waveform, label):
        # Normalize data
        wav = list(map(normalization,waveform))

        # Initlialize plot
        plt.figure(figsize=(16, 10))
        plt.yticks(np.arange(-1.2, 1.2, 0.2))
        plt.title(label)
        plt.ylim([-1.1, 1.1])

        # Plot waveform
        plt.plot(wav)

        # Export plot
        plt.savefig(label + '_waveform_plot.png')