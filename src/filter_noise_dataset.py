import sys, getopt
import os
from scipy.io import wavfile
import argparse
from noise_filter.noise_filter import Noise_Filter

def main(argv):
    # opts, args = getopt.getopt(argv,"h")
    # for opt, arg in opts:
    #     if opt == '-h':
    #         print('make_noise_dataset.py <input_dir> <output_dir>\n-r, --snr   Signal to Noise ratio')
    #         sys.exit()


    # Arguments and help tips
    parser = argparse.ArgumentParser(description='Script for making white noise or burst noise dataset')
    parser.add_argument('input_dir', help='Path to the input directory')
    parser.add_argument('output_dir', help='Path where the trained model and metrics will be saved.')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
            
    for filename in os.listdir(input_dir):
        inputfile = os.path.join(input_dir,filename)
        outputfile = os.path.join(output_dir,filename)
        if os.path.isfile(inputfile):
            samplerate, wav = wavfile.read(inputfile)
            wav = Noise_Filter.filter(wav,samplerate)
            wavfile.write(outputfile,samplerate,wav)

if __name__ == "__main__":
    main(sys.argv[1:])