############################################
# Put this at the top of each file before 
# importing TensorFlow to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################################

import tensorflow as tf 
import pandas as pd
import shutil
import matplotlib.pyplot as plt

class Model_Utils:

    metrics_file_name= "history.log"
    model_file_name = "model.keras"

    # Create a Convolution Neural Network (CNN)
    # TODO: Look into adding another Conv2D layer
    @staticmethod
    def create_default_model(input_shape_size):
        # Sequential Model: plain stack of layers where each layer has one input and one output
        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Masking(mask_value=0.,input_shape=input_shape_size))
        # Conv2D: Converts spectrogram image to a 2D 3x3 matrix of pixels
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_size))
        # Max Pooling: Converts the 3x3 matrix to a 2x2, finds the most significant features of the matrix
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # Flatten: converts the matrix into a 1D array
        model.add(tf.keras.layers.Flatten())
        # Dense: Network layers
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    # Cleans the model directory
    @staticmethod
    def build_model_path(models_dir):
        if os.path.exists(models_dir):
            try:
                shutil.rmtree(models_dir) 
            except OSError as e:
                print(f"Error in build_model_path: {e}")
        os.makedirs(models_dir)
        
    
    @staticmethod
    def load_model(models_dir):
        metrics_path = os.path.join(models_dir, Model_Utils.metrics_file_name)
        metrics_data = pd.read_csv(metrics_path, sep=',', engine='python')
        print(metrics_data)
        model_path= os.path.join(models_dir, Model_Utils.model_file_name)
        loaded_model = tf.keras.saving.load_model(model_path)
        loaded_model.summary()
        return loaded_model
    
    @staticmethod
    def save_model_plots(history, models_dir):
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        accuracy_path = os.path.join(models_dir, "history_accuracy.png")
        plt.savefig(accuracy_path)
        plt.cla()
        plt.clf()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        loss_path = os.path.join(models_dir, "history_loss.png")
        plt.savefig(loss_path)
        plt.cla()
        plt.clf()
