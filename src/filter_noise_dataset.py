import sys, getopt
import os
from scipy.io import wavfile
import numpy as np
from noise_filter.noise_filter import Noise_Filter

def main(argv):
    opts, args = getopt.getopt(argv,"h")
    for opt, arg in opts:
        if opt == '-h':
            print('make_noise_dataset.py <input_dir> <output_dir>\n-r, --snr   Signal to Noise ratio')
            sys.exit()
            
    input_dir = argv[-2]
    output_dir = argv[-1]
    for filename in os.listdir(input_dir):
        inputfile = os.path.join(input_dir,filename)
        outputfile = os.path.join(output_dir,filename)
        if os.path.isfile(inputfile):
            samplerate, wav = wavfile.read(inputfile)
            wav = Noise_Filter.filter(wav,samplerate)
            wavfile.write(outputfile,samplerate,wav)

if __name__ == "__main__":
    main(sys.argv[1:])