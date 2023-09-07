from model_utilities.model_utils import Model_Utils
import tensorflow as tf 

class DeepFake_Detector:
    def __init__(self):
        self.loaded_model = None

    def load_model(self, models_dir, model_id):
        self.loaded_model = Model_Utils.load_model(models_dir, model_id)

    def predict(self, test_spectrogram_ds):
        self.loaded_model.evaluate(test_spectrogram_ds, return_dict=True)
        y_hat = self.loaded_model.predict(test_spectrogram_ds)
        print(y_hat)
    
