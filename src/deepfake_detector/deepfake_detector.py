############################################
# Put this at the top of each file before 
# importing TensorFlow to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################################

from model_utilities.model_utils import Model_Utils
from termcolor import colored
import numpy as np

class DeepFake_Detector:
    def __init__(self):
        self.loaded_model = None

    def load_model_from_file(self, models_dir):
        self.loaded_model = Model_Utils.load_model(models_dir)

    def set_model(self, model):
        self.loaded_model = model

    def predict_dataset(self, test_spectrogram_ds, class_names):
        print("Making predictions for test dataset...")
        file_num = 1
        correct = 0
        batch_size = 0
        for batch_num, (X, Y) in enumerate(test_spectrogram_ds):
            batch_size = len(Y)
            pred = self.loaded_model.predict(X)
            for i in range(batch_size):
                output_arr = pred[i]
                output = output_arr[0]
                confidence_str = format(output, ".4%") if output >= 0.5 else format((1-output), ".4%")
                predicted = 1 if output >= 0.5 else 0
                actual = Y[i]
                print_color = 'red'
                if predicted == actual:
                    print_color = 'green'
                    correct += 1
                print(colored(f'WAV {file_num} Prediction: {class_names[predicted]}, Actual: {class_names[actual]}, Confidence: {confidence_str} ', print_color))
                file_num += 1
            break

        print(f'Number correct: {correct} out of {batch_size}')
        print(f'Accuracy: {format(correct / batch_size, ".4%")}')