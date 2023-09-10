from data_processor.data_processor import Data_Processor

DATASET_PATH_FAKE = '/workspaces/data_sets/fake_audio/generated_audio'
DATASET_PATH_REAL = '/workspaces/data_sets/real_audio/LJSpeech-1.1'
PLOT_PATH = 'plot'

if __name__ == "__main__":   
    processor = Data_Processor()
    processor.load_real_dataset(DATASET_PATH_REAL)
    processor.load_fake_dataset(DATASET_PATH_FAKE)
    processor.make_spectrogram_datasets()
    processor.plot_first_spectrogram()
    processor.plot_first_waveform()
    processor.plot_dual_wave_spec()