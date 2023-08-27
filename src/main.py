from data_processor import Data_Processor

DATASET_PATH = '/workspaces/data_sets/fake_audio/generated_audio'
PLOT_PATH = 'plot.png'

if __name__ == "__main__":   
    processor = Data_Processor(DATASET_PATH)
    processor.loadDataset()
    processor.plotFirstSample(PLOT_PATH)