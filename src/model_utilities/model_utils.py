import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 
import pandas as pd

class Model_Utils:

    metrics_file_name= "metric.log"
    model_file_name = "model.keras"

    @staticmethod
    def create_default_model(input_shape_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_size))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def build_model_path(models_dir, model_id):
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_folder = os.path.join(models_dir, str(model_id))
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        return model_folder
    
    @staticmethod
    def load_model(models_dir, model_id):
        model_folder = Model_Utils.build_model_path(models_dir, model_id)
        metrics_path = os.path.join(model_folder, Model_Utils.metrics_file_name)
        metrics_data = pd.read_csv(metrics_path, sep=',', engine='python')
        print(metrics_data)
        model_path= os.path.join(model_folder, Model_Utils.model_file_name)
        loaded_model = tf.keras.saving.load_model(model_path)
        loaded_model.summary()
        return loaded_model