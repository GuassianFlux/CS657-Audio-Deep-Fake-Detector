import numpy as np
import noisereduce as nr

class Noise_Filter:
    @staticmethod
    def filter(waveform, samplerate):
        return nr.reduce_noise(y=waveform, sr=samplerate)