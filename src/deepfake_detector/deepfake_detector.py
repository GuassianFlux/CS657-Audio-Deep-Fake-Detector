from model_utilities.model_utils import Model_Utils
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 

class DeepFake_Detector:
    def __init__(self):
        self.loaded_model = None

    def load_model(self, models_dir, model_id):
        self.loaded_model = Model_Utils.load_model(models_dir, model_id)

    def predict(self, test_spectrogram_ds, label_names):
        self.loaded_model.evaluate(test_spectrogram_ds, return_dict=True)
        y_pred = self.loaded_model.predict(test_spectrogram_ds)
        y_pred = [1 if prediction > 0.5 else 0 for prediction in y_pred]
        file_num = 1
        for prediction in y_pred:
            prediction_text = "Real"
            if prediction == 1:
                prediction_text = "Fake"
            print("Test File", file_num, "Prediction:", prediction_text)
            file_num = file_num + 1
