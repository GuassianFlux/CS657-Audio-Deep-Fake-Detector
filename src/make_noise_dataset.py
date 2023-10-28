import sys, getopt
import os
from scipy.io import wavfile
import numpy as np
from noise_generator.noise_generator import Noise_Generator
import argparse
import shutil

# Add function reference to noise option here.
# Main will check against this data structure to determine if valid options are passed
noise_dict = {
     'white': Noise_Generator.add_agwn,
     'burst': Noise_Generator.add_burst
}

def main(argv):
    # snr = 15
    # type = 'white'
    # opts, args = getopt.getopt(argv,'hr:t:',['snr=,type='])
    # for opt, arg in opts:
    #     if opt == '-h':
    #         print('make_noise_dataset.py <input_dir> <output_dir>\n-r, --snr   Signal to Noise ratio')
    #         sys.exit()
    #     elif opt in ('-r', '--snr'):
    #             snr=np.int16(arg)
    #     elif opt in ('-t', '--type'):
    #         if arg in noise_dict.keys():
    #             type = arg
    #         else:
    #             print('Invalid noise option.')
    #             print('Valid options are:')
    #             print('[' + ", ".join(list(noise_dict.keys())) + ']')
    #             print('Defaulting to white noise.')

    # Arguments and help tips
    parser = argparse.ArgumentParser(description='Script for making white noise or burst noise dataset')
    parser.add_argument('input_dir', help='Path to the input directory')
    parser.add_argument('output_dir', help='Path where the trained model and metrics will be saved.')
    parser.add_argument('--type', '-t', help='Noise type - White noise or bust noise')
    parser.add_argument('--snr', '-r', help='Sinal to noise ratio')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    type = args.type
    snr = args.snr

    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir) 
        except OSError as e:
            print(e)
    os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        inputfile = os.path.join(input_dir,filename)
        outputfile = os.path.join(output_dir,filename)
        if os.path.isfile(inputfile):
            samplerate, wav = wavfile.read(inputfile)
            wav = noise_dict[type](wav, snr)
            wavfile.write(outputfile,samplerate,wav)

if __name__ == '__main__':
    main(sys.argv[1:])