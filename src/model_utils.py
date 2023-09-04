import os
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from keras.callbacks import CSVLogger
import pandas as pd
import uuid

class Model_Utils:

    trained_models_dir = "/workspaces/trained_models"
    metrics_file_name= "metrics.log"
    model_file_name = "model.keras"

    @staticmethod
    def fit_model(train, val, epochs):
        model = Model_Utils._create_default_model()
        model_id = uuid.uuid5()
        model_folder = Model_Utils._build_model_path(model_id)
        metrics_path = os.path.join(model_folder, Model_Utils.metrics_file_name)
        csv_logger = CSVLogger(metrics_path, separator=',', append=False)
        model.fit(train, epochs=epochs, validation_data=val, callbacks=[csv_logger])
        model_path= os.path.join(model_folder, Model_Utils.model_file_name)
        model.save(model_path)
        return model

    @staticmethod
    def _build_model_path(model_id):
        model_folder = os.path.join(Model_Utils.trained_models_dir, str(model_id))
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        return model_folder

    @staticmethod
    def _create_default_model():
        model = Sequential()
        model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))
        model.add(Conv2D(16, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),tf.keras.metrics.Accuracy()])
        return model

    @staticmethod
    def load_model(model_id):
        model_folder = Model_Utils._build_model_path(model_id)
        model_path= os.path.join(model_folder, Model_Utils.model_file_name)
        loaded_model = tf.keras.saving.load_model(model_path)
        metrics_path = os.path.join(model_folder, Model_Utils.metrics_file_name)
        metrics_data = pd.read_csv(metrics_path, sep=',', engine='python')
        print("Metrics:", metrics_data)
        loaded_model.summary()
