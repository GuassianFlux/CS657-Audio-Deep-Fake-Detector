import os
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
import pandas as pd

class Model_Utils:

    metrics_file_name= "metric.log"
    model_file_name = "model.keras"

    @staticmethod
    def create_default_model():
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Accuracy()])
        return model

    @staticmethod
    def build_model_path(models_dir, model_id):
        model_folder = os.path.join(models_dir, str(model_id))
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        return model_folder

    @staticmethod
    def load_model(models_dir, model_id):
        model_folder = Model_Utils.build_model_path(models_dir, model_id)
        metrics_path = os.path.join(model_folder, Model_Utils.metrics_file_name)
        metrics_data = pd.read_csv(metrics_path, sep=',', engine='python')
        print("Metrics:", metrics_data)
        model_path= os.path.join(model_folder, Model_Utils.model_file_name)
        loaded_model = tf.keras.saving.load_model(model_path)
        loaded_model.summary()
        return loaded_model
