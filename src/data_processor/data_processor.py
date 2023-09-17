############################################
# Put this at the top of each file before 
# importing TensorFlow to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################################

import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from keras.callbacks import CSVLogger
from model_utilities.model_utils import Model_Utils
from keras import backend as kerasbackend

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

class Data_Processor:
    def __init__(self):
        self.train_ds = []
        self.val_ds = []
        self.test_ds = []
        self.train_spectrograms = []
        self.val_spectrograms = []
        self.test_spectrograms = []
        self.class_names = []

    def load_datasets(self, data_file_path):
        SEQUENCE_LENGTH = 9 * 16000 # 9s
        print("Loading data sets from", data_file_path)
        data_dir = pathlib.Path(data_file_path)
        self.train_ds,val_and_test_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=data_dir,
            batch_size=64,
            validation_split=0.2,
            shuffle=True,
            seed=0,
            output_sequence_length=SEQUENCE_LENGTH,
            subset='both')
        self.val_ds = val_and_test_ds.shard(num_shards=2, index=0)
        self.test_ds = val_and_test_ds.shard(num_shards=2, index=1)
        self.class_names = np.array(self.train_ds.class_names)

    def make_spectrogram_datasets(self):
        train_ds = self.train_ds.map(squeeze, tf.data.AUTOTUNE)
        val_ds = self.val_ds.map(squeeze, tf.data.AUTOTUNE)
        test_ds = self.test_ds.map(squeeze, tf.data.AUTOTUNE)
        self.train_spectrograms = train_ds.map(
            map_func=lambda audio, label: (get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
        self.val_spectrograms = val_ds.map(
            map_func=lambda audio, label: (get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
        self.test_spectrograms = test_ds.map(
            map_func=lambda audio, label: (get_spectrogram(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    
    def get_test_dataset(self):
        return self.test_spectrograms
    
    def get_class_names(self):
        return self.class_names

    def _get_input_shape(self):
        for example_spectrograms, example_spect_labels in self.train_spectrograms.take(1):
            break
        input_shape = example_spectrograms.shape[1:]
        return input_shape

    def fit_model(self, models_dir):
        # Cleanup or created model folder
        Model_Utils.build_model_path(models_dir)

        # Path where all of the model history and metrics will be saved
        metrics_path = os.path.join(models_dir, Model_Utils.metrics_file_name)
        csv_logger = CSVLogger(metrics_path, separator=',', append=False)

        # Train model
        input_shape = self._get_input_shape()
        model = Model_Utils.create_default_model(input_shape)

        # TODO: Look into early stop callback. I have seen examples that 
        # use this and it is supposed to prevent overtraining.
        history = model.fit(
            self.train_spectrograms,
            validation_data=self.val_spectrograms,
            epochs=10,
            callbacks=[csv_logger],
        )

        # Save the model and history plots to file
        model_path= os.path.join(models_dir, Model_Utils.model_file_name)
        print("Saving model to path", model_path)
        model.save(model_path, save_format='tf')
        Model_Utils.save_model_plots(history, models_dir)

        # Clear model and session
        kerasbackend.clear_session()
        del model
