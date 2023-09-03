from data_processor import Data_Processor

DATASET_PATH_FAKE = '/workspaces/data_sets/fake_audio/generated_audio'
DATASET_PATH_REAL = '/workspaces/data_sets/real_audio/LJSpeech-1.1'
PLOT_PATH = 'plot'

if __name__ == "__main__":   
    processor = Data_Processor(DATASET_PATH_REAL)
    processor.loadDataset()
    processor.makeSpectrogram(PLOT_PATH)