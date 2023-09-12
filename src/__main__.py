import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data_processor.data_processor import Data_Processor
from deepfake_detector.deepfake_detector import DeepFake_Detector
import tensorflow as tf 
import warnings


DATASET_PATH = '/workspaces/small_data_sets'
TRAINED_MODELS = "/workspaces/trained_models"
PLOT_PATH = 'plot'

if __name__ == "__main__": 
    # Ignore Python and TensorFlow Warnings   
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel('ERROR')

    processor = Data_Processor()
    processor.load_datasets(DATASET_PATH)
    processor.make_spectrogram_datasets()
    model_id = processor.fit_model(TRAINED_MODELS)
    test_ds = processor.get_test_dataset()

    detector = DeepFake_Detector()
    detector.load_model_from_file(TRAINED_MODELS, model_id)
    detector.predict(test_ds)