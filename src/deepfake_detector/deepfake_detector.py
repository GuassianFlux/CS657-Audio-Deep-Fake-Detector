############################################
# Put this at the top of each file before 
# importing TensorFlow to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################################

from model_utilities.model_utils import Model_Utils
from termcolor import colored
from keract import get_activations, display_activations
import numpy as np
import shutil

class DeepFake_Detector:
    def __init__(self):
        self.loaded_model = None

    def load_model_from_file(self, models_dir):
        self.loaded_model = Model_Utils.load_model(models_dir)

    def set_model(self, model):
        self.loaded_model = model

    def _log_outputs(self, sample, prediction_dir):
        inp = self.loaded_model.input
        outputs = [layer.output for layer in self.loaded_model.layers]
        functors = [tf.keras.backend.function([inp], [out]) for out in outputs]
        layer_outs = [func([sample]) for func in functors]
        file_name = "output.txt"
        output_file = os.path.join(prediction_dir, file_name)
        with open(output_file, 'w') as file:
            for idx, layer in enumerate(self.loaded_model.layers):
                print("layer name {} \noutputs: {}". format(layer.name, numpy.array(layer_outs[idx])), file=file)

    def _show_ouputs(self, input, prediction_dir):
        activations = get_activations(self.loaded_model, input) #, auto_compile=True)
        display_activations(activations, cmap='YlGnBu', save=True, directory=prediction_dir)

    def build_prediction_path(self, prediction_dir):
        if os.path.exists(prediction_dir):
            try:
                shutil.rmtree(prediction_dir) 
            except OSError as e:
                print(f"Error in build_model_path: {e}")
        os.makedirs(prediction_dir)

    def predict_dataset(self, test_spectrogram_ds, class_names, prediction_dir):
        print("Making dataset predictions...")
        self.build_prediction_path(prediction_dir)
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

            #self._show_ouputs(X[1:2], prediction_dir)

        print(f'Number correct: {correct} out of {batch_size}')
        print(f'Accuracy: {format(correct / batch_size, ".4%")}')