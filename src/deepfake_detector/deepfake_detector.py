from model_utilities.model_utils import Model_Utils

class DeepFake_Detector:
    def __init__(self):
        self.loaded_model = None

    def load_model_from_file(self, models_dir, model_id):
        self.loaded_model = Model_Utils.load_model(models_dir, model_id)

    def set_model(self, model):
        self.loaded_model = model

    def predict(self, test_spectrogram_ds):
        correct = 0
        batch_size = 0
        for batch_num, (X, Y) in enumerate(test_spectrogram_ds):
            batch_size = len(Y)
            pred = self.loaded_model.predict(X)
            for i in range(batch_size):
                predicted = [1 if prediction > 0.5 else 0 for prediction in pred[i]]
                actual = Y[i]
                print(f'predicted {predicted}, actual {actual}')
                if predicted == actual:
                    correct += 1
            break

        print(f'Number correct: {correct} out of {batch_size}')
        print(f'Accuracy: {correct / batch_size}')
