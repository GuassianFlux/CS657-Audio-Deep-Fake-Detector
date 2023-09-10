import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from keras.callbacks import CSVLogger
import uuid
from model_utilities.model_utils import Model_Utils
from keras import backend as kerasbackend

def test(waveform):
    return waveform

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
    def __init__(self):
        self.real_ds = []
        self.fake_ds = []
        self.real_spectrograms = []
        self.fake_spectrograms = []

    def load_real_dataset(self, data_file_path):
        data_dir = pathlib.Path(data_file_path)
        print("[INFO] Loading", data_file_path)
        self.real_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=64,
        validation_split=0,
        seed=0,
        output_sequence_length=16000)
    
    def get_real_dataset(self):
        return self.real_ds

    def load_fake_dataset(self, data_file_path):
        data_dir = pathlib.Path(data_file_path)
        print("[INFO] Loading", data_file_path)
        self.fake_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=64,
        validation_split=0,
        seed=0,
        output_sequence_length=16000)

    def get_fake_dataset(self):
        return self.fake_ds

    def make_spectrogram_datasets(self):
        real_ds = self.real_ds.map(squeeze, tf.data.AUTOTUNE)
        fake_ds = self.fake_ds.map(squeeze, tf.data.AUTOTUNE)
        self.real_spectrograms = real_ds.map(
            map_func=lambda audio, label: (get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)
        self.fake_spectrograms = fake_ds.map(
            map_func=lambda audio, label: (get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)

    def get_real_spectrograms(self):
        return self.real_spectrograms
    
    def get_fake_spectrograms(self):
        return self.fake_spectrograms

    def plot_dual_wave_spec(self):
        label_names = np.array(self.fake_ds.class_names)
        fake_ds = self.fake_ds.map(squeeze, tf.data.AUTOTUNE)
        for audio, labels in fake_ds.take(1):
            for i in range(3):
                label = label_names[labels[i]]
                waveform = audio[i]
                spectrogram = get_spectrogram(waveform)

                fig, axes = plt.subplots(2, figsize=(12, 8))
                timescale = np.arange(waveform.shape[0])
                axes[0].plot(timescale, waveform.numpy())
                axes[0].set_title('Waveform')
                axes[0].set_xlim([0, 16000])

                plot_spectrogram(spectrogram.numpy(), axes[1])
                axes[1].set_title('Spectrogram')
                plt.suptitle(label.title())
                plt.savefig('fake_wave_with_spec.png')

    def plot_first_spectrogram(self):
        label_names = np.array(self.fake_ds.class_names)
        for spectrograms, spect_labels in self.fake_spectrograms.take(1):
            rows = 3
            cols = 3
            n = rows*cols
            fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

            for i in range(n):
                r = i // cols
                c = i % cols
                ax = axes[r][c]
                plot_spectrogram(spectrograms[i].numpy(), ax)
                ax.set_title(label_names[spect_labels[i].numpy()])
            plt.savefig('fake_spectrograms_plot.png')
        label_names = np.array(self.real_ds.class_names)
        for spectrograms, spect_labels in self.real_spectrograms.take(1):
            rows = 1
            cols = 1
            n = rows*cols
            fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

            plot_spectrogram(spectrograms[i].numpy(), axes)
            axes.set_title(label_names[spect_labels[i].numpy()])
            plt.savefig('real_spectrograms_plot.png')   

    def plot_first_waveform(self):
        label_names = np.array(self.fake_ds.class_names)
        fake_ds = self.fake_ds.map(squeeze, tf.data.AUTOTUNE)
        for audio, labels in fake_ds.take(1):  
            plt.figure(figsize=(16, 10))
            rows = 3
            cols = 3
            n = rows * cols
            for i in range(n):
                plt.subplot(rows, cols, i+1)
                audio_signal = audio[i]
                plt.plot(audio_signal)
                plt.title(label_names[labels[i]])
                plt.yticks(np.arange(-1.2, 1.2, 0.2))
                plt.ylim([-1.1, 1.1])
                plt.savefig('fake_waveform_plot.png')

    def fit_model(self, models_dir):
        print("Preparing to train model...")
        # Split the spectrogram datasets for training and validation
        train, val = self._split_data()
        # Build the folder for the model
        model_id = uuid.uuid4()
        print("Model ID:", str(model_id))
        model_folder = Model_Utils.build_model_path(models_dir, model_id)
        print("Model and metrics will be saved in", model_folder)
        # Path where all of the model history and metrics will be saved
        metrics_path = os.path.join(model_folder, Model_Utils.metrics_file_name)
        csv_logger = CSVLogger(metrics_path, separator=',', append=False)
        print("Creating metrics file", metrics_path)
        # Create and fit the model
        print("Creating and fitting model...")
        model = Model_Utils.create_default_model()
        history = model.fit(
            train,
            validation_data=val,
            epochs=1,
            callbacks=[csv_logger],
        )
        # Save the model to file
        model_path= os.path.join(model_folder, Model_Utils.model_file_name)
        print("Saving model to path", model_path)
        model.save(model_path)
        print("Printing model summary")
        model.summary()
        print('Model accuracy:', history.history['accuracy'])
        # Clear the keras session and cleanup
        print("Clearing model session")
        kerasbackend.clear_session()
        del model
        return str(model_id)
    
    def _split_data(self):
        # Not sure if this is actually how we want to split the data
        print("Splitting dataset for training and validation...")
        data = self.real_spectrograms.concatenate(self.fake_spectrograms)
        data.shuffle(1000)
        data.batch(64)
        size = sum(1 for _ in data.unbatch())
        print("Total Dataset Size:", str(size))
        train_size = np.int64(size * 0.8)
        print("Training Set Size:", str(train_size))
        val_size = np.int64(size * 0.2)
        print("Validation Set Size:", str(val_size))
        train = data.take(train_size)
        val = data.skip(train_size).take(val_size)
        return train, val