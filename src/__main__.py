from data_processor.data_processor import Data_Processor
from deepfake_detector.deepfake_detector import DeepFake_Detector

DATASET_PATH = '/workspaces/data_sets'
TRAINED_MODELS = "/workspaces/trained_models"
PLOT_PATH = 'plot'

if __name__ == "__main__":   
    processor = Data_Processor()
    processor.load_datasets(DATASET_PATH)
    processor.make_spectrogram_datasets()
    # processor.plot_first_spectrogram()
    # processor.plot_first_waveform()
    # processor.plot_dual_wave_spec()
    model_id = processor.fit_model(TRAINED_MODELS)
    test_ds = processor.get_test_dataset()
    label_names = processor.get_label_names()

    detector = DeepFake_Detector()
    detector.load_model(TRAINED_MODELS, model_id)
    detector.predict(test_ds, label_names)

