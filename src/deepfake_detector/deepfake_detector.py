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
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()
    
