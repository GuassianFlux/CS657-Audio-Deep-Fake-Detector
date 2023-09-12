import numpy as np

INT16_MIN=-32768
INT16_MAX=32767

class Noise_Generator:
    @staticmethod
    def add_agwn(waveform, snr):
        # Compute average amplitude
        wav_abs = np.absolute(waveform)
        wav_mean = np.mean(wav_abs);

        # Produce noise
        noise_amp = wav_mean / snr
        noise = np.random.normal(0, noise_amp, len(waveform)).astype(np.int16)

        # Combine original wavefile with noise
        noise_wav = np.add(noise, waveform)
        noise_wav = np.clip(noise_wav, INT16_MIN, INT16_MAX)

        return noise_wav