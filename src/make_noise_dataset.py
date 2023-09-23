import sys, getopt
import os
from scipy.io import wavfile
import numpy as np
from noise_generator.noise_generator import Noise_Generator

# Add function reference to noise option here.
# Main will check against this data structure to determine if valid options are passed
noise_dict = {
     'white': Noise_Generator.add_agwn,
     'burst': Noise_Generator.add_burst
}

def main(argv):
    snr = 15
    type = 'white'
    opts, args = getopt.getopt(argv,'hr:t:',['snr=,type='])
    for opt, arg in opts:
        if opt == '-h':
            print('make_noise_dataset.py <input_dir> <output_dir>\n-r, --snr   Signal to Noise ratio')
            sys.exit()
        elif opt in ('-r', '--snr'):
                snr=np.int16(arg)
        elif opt in ('-t', '--type'):
            if arg in noise_dict.keys():
                type = arg
            else:
                print('Invalid noise option.')
                print('Valid options are:')
                print('[' + ", ".join(list(noise_dict.keys())) + ']')
                print('Defaulting to white noise.')

            
    input_dir = argv[-2]
    output_dir = argv[-1]
    for filename in os.listdir(input_dir):
        inputfile = os.path.join(input_dir,filename)
        outputfile = os.path.join(output_dir,filename)
        if os.path.isfile(inputfile):
            samplerate, wav = wavfile.read(inputfile)
            wav = noise_dict[type](wav, snr)
            wavfile.write(outputfile,samplerate,wav)

if __name__ == '__main__':
    main(sys.argv[1:])