############################################
# Put this at the top of each file before 
# importing TensorFlow to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################################

from model_utilities.model_utils import Model_Utils
from termcolor import colored
import tensorflow as tf 
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

class DeepFake_Detector:
    def __init__(self):
        self.loaded_model = None

    def load_model_from_file(self, models_dir):
        self.loaded_model = Model_Utils.load_model(models_dir)

    def set_model(self, model):
        self.loaded_model = model

    def _log_outputs(self, sample):
        inp = self.loaded_model.input
        outputs = [layer.output for layer in self.loaded_model.layers]
        functors = [tf.keras.backend.function([inp], [out]) for out in outputs]
        layer_outs = [func([sample]) for func in functors]
        file_name = "output.txt"
        output_file = os.path.join("trained_model", file_name)
        with open(output_file, 'w') as file:
            for idx, layer in enumerate(self.loaded_model.layers):
                print("layer name {} \noutputs: {}". format(layer.name, numpy.array(layer_outs[idx])), file=file)

    def predict_dataset(self, test_spectrogram_ds, class_names):
        print("Making predictions for test dataset...")
        file_num = 1
        correct = 0
        batch_size = 0
        for batch_num, (X, Y) in enumerate(test_spectrogram_ds):
            batch_size = len(Y)
            pred = self.loaded_model.predict(X)
            for i in range(batch_size):
                predicted_arr = [1 if prediction > 0.5 else 0 for prediction in pred[i]]
                predicted = predicted_arr[0]
                actual = Y[i]
                print_color = 'red'
                if predicted == actual:
                    print_color = 'green'
                    correct += 1
                print(colored(f'WAV {file_num} Prediction: {class_names[predicted]}, Actual: {class_names[actual]}', print_color))
                file_num += 1
            #break
            #self._log_outputs(X)

        print(f'Number correct: {correct} out of {batch_size}')
        print(f'Accuracy: {format(correct / batch_size, ".0%")}')