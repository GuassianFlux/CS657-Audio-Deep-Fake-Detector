import numpy as np
from scipy import signal

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
    
    @staticmethod
    def add_burst(waveform):
        print(waveform)
        wav_abs = np.absolute(waveform)
        wav_max = np.max(wav_abs);

        noise_amp = wav_max / 15
        noise_wav = []
        toggle = False
        for sample in waveform:
            if(toggle):
                noise_sample = (sample+noise_amp).astype(np.int16)
                noise_wav.append(noise_sample)
            else:
                noise_wav.append(sample)

            rand = np.random.random()
            if(rand > 0.998):
                toggle = not toggle

        noise_wav = np.clip(noise_wav, INT16_MIN, INT16_MAX)
        print(noise_wav)
        return noise_wav
        
