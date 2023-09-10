import unittest
import tensorflow as tf 
from  model_utilities.model_utils import Model_Utils
from tensorflow.python.keras.models import Sequential

def load_and_process_wav(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav

def preprocess(file_path, label): 
    wav = load_and_process_wav(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

class ModelTests(unittest.TestCase):

    def setUp(self):
        print("\nRunning setUp method...")

    def tearDown(self):
        print("Running tearDown method...")

    def test_loading_model(self):
         loaded_model = Model_Utils.load_model('78d8553b-6fec-4924-a0ed-ee571ccfd104')
         print("Finished loading model")

    # def test_training_model(self):
    #     print("Test Training Model")
    #     current_directory = os.getcwd()
    #     print("Current Working Directory:", current_directory)
    #     POS = os.path.join(current_directory, 'data_sets', 'real_audio')
    #     if os.path.exists(POS):
    #         print(POS, "exists.")
    #     NEG = os.path.join(current_directory, 'data_sets', 'fake_audio', 'generated_audio', 'common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech')
    #     if os.path.exists(NEG):
    #         print(NEG, "exists.")
    #     pos = tf.data.Dataset.list_files(POS+'/*.wav')
    #     neg = tf.data.Dataset.list_files(NEG+'/*.wav')
    #     positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    #     negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    #     data = positives.concatenate(negatives)
    #     data = data.map(preprocess)
    #     data = data.cache()
    #     data = data.shuffle(buffer_size=1000)
    #     data = data.batch(16)
    #     data = data.prefetch(8)
    #     train = data.take(50)
    #     val = data.skip(15).take(25)
    #     Model_Utils.fit_model(train, val, 1)


if __name__=='__main__':
	unittest.main()

