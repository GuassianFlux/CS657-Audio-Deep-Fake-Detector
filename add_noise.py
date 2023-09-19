import sys, getopt
import os
from scipy.io import wavfile
import numpy as np

INT16_MIN=-32768
INT16_MAX=32767

def add_agwn(waveform, snr):
    # Compute average amplitude
    wav_abs = np.absolute(waveform)
    wav_mean = np.mean(wav_abs)

    # Produce noise
    noise_amp = wav_mean / snr
    noise = np.random.normal(0, noise_amp, len(waveform)).astype(np.int16)

    # Combine original wavefile with noise
    noise_wav = np.add(noise, waveform)
    noise_wav = np.clip(noise_wav, INT16_MIN, INT16_MAX)

    return noise_wav

input_dir = 'test_data_set/real'
output_dir = 'noisey_test_data_set/real'
for filename in os.listdir(input_dir):
    print(filename)
    inputfile = os.path.join(input_dir,filename)
    outputfile = os.path.join(output_dir,filename)
    if os.path.isfile(inputfile):
        samplerate, wav = wavfile.read(inputfile)
        wav = add_agwn(wav,15)
        wavfile.write(outputfile,samplerate,wav)