#!/bin/bash

# Workflow 7 - Adding Burst Noise: Create burst noise datasets
echo "=== Workflow 7 - Adding Burst Noise: Create burst noise datasets ==="

root_noise_dir='/workspaces/burst_noise_data_sets'
if [ -d $root_noise_dir ]; then
    rm -r $root_noise_dir
fi
mkdir $root_noise_dir

training_dataset_path='/workspaces/data_sets'
real_dataset_path='/workspaces/data_sets/real'
real_burst_noise_dataset_path='/workspaces/burst_noise_data_sets/real'
fake_dataset_path='/workspaces/data_sets/fake'
fake_burst_noise_dataset_path='/workspaces/burst_noise_data_sets/fake'

# Add burst noise to real data
python /workspaces/src/make_noise_dataset.py $real_dataset_path $real_burst_noise_dataset_path --snr 15 --type burst

# Add burst noise to fake data
python /workspaces/src/make_noise_dataset.py $fake_dataset_path $fake_burst_noise_dataset_path --snr 15 --type burst