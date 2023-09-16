############################################
# Put this at the top of each file before 
# importing TensorFlow to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################################

from data_processor.data_processor import Data_Processor
from deepfake_detector.deepfake_detector import DeepFake_Detector
import tensorflow as tf 
import warnings
import shutil

# Path of the dataset that will be used to train, validate, and test the model. Classes will be 
# constrcuted based on the folders in this directory(Ex. fake, real). The small dataset can be 
# used for demo. Make sure the dataset only has a real and fake folder as children.
DATASET_PATH = '/workspaces/small_data_sets'
# DATASET_PATH = '/workspaces/data_sets'

# Path where the trained model will be saved along with metrics
TRAINED_MODEL = "/workspaces/trained_model"

# Legacy model folder structure. This will be deleted if it exists.
OLD_TRAINED_MODELS = "/workspaces/trained_models"
def delete_old_models():
    if os.path.exists(OLD_TRAINED_MODELS):
        try:
            shutil.rmtree(OLD_TRAINED_MODELS)  # Remove the directory and its contents recursively
        except OSError as e:
            print(f"Error deleting old trained models: {e}")

if __name__ == "__main__": 
    # Ignore Python and TensorFlow Warning and Debug Info 
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel('ERROR')

    # Delete the old trained models folder structure
    delete_old_models();

    # Create initialize data processor and load datasets
    processor = Data_Processor()
    processor.load_datasets(DATASET_PATH)
    processor.make_spectrogram_datasets()

    # Trains a new model with the loaded datasets.
    # This can be commented out if there is already a trained model saved
    # and you don't want to train a new one. 
    processor.fit_model(TRAINED_MODEL)
    
    # Load the trained model and make predictions on the 
    # on the test dataset
    detector = DeepFake_Detector()
    detector.load_model_from_file(TRAINED_MODEL)
    test_ds = processor.get_test_dataset()
    class_names = processor.get_class_names()
    detector.predict_dataset(test_ds, class_names)