#!/bin/bash

# Workflow 6 - Adding White Noise: Create white noise datasets
echo "=== Workflow 6 - Adding White Noise: Create white noise datasets ==="

root_noise_dir='/workspaces/white_noise_data_sets'
if [ -d $root_noise_dir ]; then
    rm -r $root_noise_dir
fi
mkdir $root_noise_dir

training_dataset_path='/workspaces/data_sets'
real_dataset_path='/workspaces/data_sets/real'
real_white_noise_dataset_path='/workspaces/white_noise_data_sets/real'
fake_dataset_path='/workspaces/data_sets/fake'
fake_white_noise_dataset_path='/workspaces/white_noise_data_sets/fake'

# Add white noise to real data
python /workspaces/src/make_noise_dataset.py $real_dataset_path $real_white_noise_dataset_path --snr 5 --type burst

# Add white noise to fake data
python /workspaces/src/make_noise_dataset.py $fake_dataset_path $fake_white_noise_dataset_path --snr 5 --type burst