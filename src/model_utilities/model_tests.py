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


if __name__=='__main__':
	unittest.main()

