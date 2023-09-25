mkdir noise
mkdir filter

python src/make_noise_dataset.py /workspaces/src/data_processor/sample_data/real_audio/audio/ /workspaces/noise
python src/filter_noise_dataset.py /workspaces/noise/ /workspaces/filter/
mv /workspaces/noise/LJ001-0001.wav /workspaces/noise/noise_LJ001-0001.wav
mv /workspaces/filter/LJ001-0001.wav /workspaces/filter/filter_LJ001-0001.wav

python src/generate_plots.py /workspaces/small_data_sets/real/LJ001-0001.wav
python src/generate_plots.py /workspaces/noise/noise_LJ001-0001.wav
python src/generate_plots.py /workspaces/filter/filter_LJ001-0001.wav