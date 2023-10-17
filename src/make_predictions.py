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
import argparse

if __name__ == '__main__':
    # Ignore Python and TensorFlow Warning and Debug Info 
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel('ERROR')

    # Arguments and help tips
    parser = argparse.ArgumentParser(description='Script for making predictions with a trained model')
    parser.add_argument('dataset_path', help='Path to the dataset. The dataset directory should contain a folder name fake and a folder name real.')
    parser.add_argument('--trained_model_path', '-tmp', help='Path where the trained model and metrics will be saved.')
    parser.add_argument('--prediction_data_path', '-pdp', help='Path where prediction data, heatmaps, and activation maps will be saved.')
    args = parser.parse_args()

    # Dataset path
    dataset_path = args.dataset_path

    # Trained model path
    trained_model_path = args.trained_model_path

    #Prediction data path
    prediction_data_path = args.prediction_data_path

    # Create initialize data processor and load datasets
    processor = Data_Processor()
    processor.load_datasets(dataset_path)
    processor.make_spectrogram_datasets()
    
    # Load the trained model and make predictions on the 
    # on the test dataset
    detector = DeepFake_Detector()
    detector.load_model_from_file(trained_model_path)
    test_ds = processor.get_test_dataset()
    class_names = processor.get_class_names()
    detector.predict_dataset(test_ds, class_names, prediction_data_path)