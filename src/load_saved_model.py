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
    parser.add_argument('trained_model_path', help='Path where the trained model and metrics will be saved.')
    args = parser.parse_args()

    # Trained model path
    trained_model_path = args.trained_model_path
    
    # Load the trained model and make predictions on the 
    # on the test dataset
    detector = DeepFake_Detector()
    detector.load_model_from_file(trained_model_path)