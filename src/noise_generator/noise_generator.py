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
        noise_amp = np.divide(float(wav_mean), float(snr))
        noise = np.random.normal(0, noise_amp, len(waveform)).astype(np.int16)

        # Combine original wavefile with noise
        noise_wav = np.add(noise, waveform)
        noise_wav = np.clip(noise_wav, INT16_MIN, INT16_MAX)

        return noise_wav
    
    @staticmethod
    def add_burst(waveform, snr):
        TOGGLE_RATE = 1000 # Toggles on average every 1000 samples

        # Compute max amplitude
        wav_abs = np.absolute(waveform)
        wav_max = np.max(wav_abs);

        # Initialize noise variables
        noise_wav = []
        noise_toggle = False
        noise_amp = np.divide(float(wav_max), float(snr))

        # For each sample in the waveform, evaluate a boolean toggle state and randomly update it
        # If true, add the burst noise to the sample
        # If false, do not alter the signal
        for sample in waveform:
            if(noise_toggle):
                noise_sample = (sample+noise_amp).astype(np.int16)
                noise_wav.append(noise_sample)
            else:
                noise_wav.append(sample)

            rand = np.random.random()
            if(rand < 1/TOGGLE_RATE):
                noise_toggle = not noise_toggle

        # Clip noise waveform so that values are within int16 range
        noise_wav = np.clip(noise_wav, INT16_MIN, INT16_MAX)
        return noise_wav
        
