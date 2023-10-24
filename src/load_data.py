############################################
# Put this at the top of each file before 
# importing TensorFlow to suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################################

from data_processor.data_processor import Data_Processor
import tensorflow as tf 
import warnings
import argparse

if __name__ == '__main__':
    # Ignore Python and TensorFlow Warning and Debug Info 
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel('ERROR')

    # Arguments and help tips
    parser = argparse.ArgumentParser(description='Script for loading a dataset')
    parser.add_argument('dataset_path', help='Path to the dataset. The dataset directory should contain a folder named fake and a folder name real.')
    args = parser.parse_args()

    # Dataset path
    dataset_path = args.dataset_path

    # Create initialize data processor and load datasets
    processor = Data_Processor()
    processor.load_datasets(dataset_path)
    processor.make_spectrogram_datasets()