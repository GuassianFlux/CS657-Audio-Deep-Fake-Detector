import sys
from scipy.io import wavfile
from plotter.plotter import Plotter

def get_label(filepath):
    filename = filepath.split('/')[-1]
    return filename.split('.')[0]

def main(argv):
    input_filepath = argv[-1]
    label = get_label(input_filepath)

    samplerate, wav = wavfile.read(input_filepath)

    Plotter.plot_waveform(waveform=wav, label=label)
    Plotter.plot_spectrogram(waveform=wav, label=label)
    Plotter.plot_dual_wave_spec(waveform=wav, label=label)

if __name__ == "__main__":
    main(sys.argv[1:])