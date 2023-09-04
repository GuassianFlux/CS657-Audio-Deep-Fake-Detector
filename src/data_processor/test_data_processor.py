import unittest

from data_processor.data_processor import Data_Processor

DATASET_PATH_FAKE = '/workspaces/src/data_processor/sample_data/fake_audio'
DATASET_PATH_REAL = '/workspaces/src/data_processor/sample_data/real_audio'

class TestDataProcessor(unittest.TestCase):
    def test_load_real_dataset(self):
        processor = Data_Processor()
        init_ds_size = len(processor.get_real_dataset())
        processor.load_real_dataset(DATASET_PATH_REAL)
        post_ds_size = len(processor.get_real_dataset())
        self.assertGreater(post_ds_size, init_ds_size)

    def test_load_fake_dataset(self):
        processor = Data_Processor()
        init_ds_size = len(processor.get_fake_dataset())
        processor.load_fake_dataset(DATASET_PATH_FAKE)
        post_ds_size = len(processor.get_fake_dataset())
        self.assertGreater(post_ds_size, init_ds_size)

    def test_make_spectrograms(self):
        processor = Data_Processor()
        init_real_spec_size = len(processor.get_real_spectrograms())
        init_fake_spec_size = len(processor.get_fake_spectrograms())
        processor.load_real_dataset(DATASET_PATH_REAL)
        processor.load_fake_dataset(DATASET_PATH_FAKE)
        processor.make_spectrogram_datasets()
        post_real_spec_size = len(processor.get_real_spectrograms())
        post_fake_spec_size = len(processor.get_fake_spectrograms())
        self.assertGreater(post_real_spec_size, init_real_spec_size)
        self.assertGreater(post_fake_spec_size, init_fake_spec_size)
