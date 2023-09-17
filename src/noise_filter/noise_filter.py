import numpy as np
import noisereduce as nr

INT16_MIN=-32768
INT16_MAX=32767

class Noise_Filter:
    @staticmethod
    def filter(waveform, samplerate):
        return nr.reduce_noise(y=waveform, sr=samplerate)