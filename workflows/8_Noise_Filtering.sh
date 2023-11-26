#!/bin/bash

# Workflow 8 - Noise Filtering: Filtering the white noise dataset
echo "=== Workflow 8 - Noise Filtering: Filtering the white and burst noise datasets ==="


real_white_noise_dataset_path='/workspaces/white_noise_data_sets/real'
fake_white_noise_dataset_path='/workspaces/white_noise_data_sets/fake'


root_filtered_white_noise_dir='/workspaces/filtered_white_noise_data_sets'
if [ -d $root_filtered_white_noise_dir ]; then
    rm -r $root_filtered_white_noise_dir
fi
mkdir $root_filtered_white_noise_dir

real_filtered_white_noise_dir='/workspaces/filtered_white_noise_data_sets/real'
if [ -d $real_filtered_white_noise_dir ]; then
    rm -r $real_filtered_white_noise_dir
fi
mkdir $real_filtered_white_noise_dir

fake_filtered_white_noise_dir='/workspaces/filtered_white_noise_data_sets/fake'
if [ -d $fake_filtered_white_noise_dir ]; then
    rm -r $fake_filtered_white_noise_dir
fi
mkdir $fake_filtered_white_noise_dir

# Filter noise for real data
python /workspaces/src/filter_noise_dataset.py $real_white_noise_dataset_path $real_filtered_white_noise_dir 

# Filter noise for fake data
python /workspaces/src/filter_noise_dataset.py $fake_white_noise_dataset_path $fake_filtered_white_noise_dir 

real_burst_noise_dataset_path='/workspaces/burst_noise_data_sets/real'
fake_burst_noise_dataset_path='/workspaces/burst_noise_data_sets/fake'


root_filtered_burst_noise_dir='/workspaces/filtered_burst_noise_data_sets'
if [ -d $root_filtered_burst_noise_dir ]; then
    rm -r $root_filtered_burst_noise_dir
fi
mkdir $root_filtered_burst_noise_dir

real_filtered_burst_noise_dir='/workspaces/filtered_burst_noise_data_sets/real'
if [ -d $real_filtered_burst_noise_dir ]; then
    rm -r $real_filtered_burst_noise_dir
fi
mkdir $real_filtered_burst_noise_dir

fake_filtered_burst_noise_dir='/workspaces/filtered_burst_noise_data_sets/fake'
if [ -d $fake_filtered_burst_noise_dir ]; then
    rm -r $fake_filtered_burst_noise_dir
fi
mkdir $fake_filtered_burst_noise_dir

# Filter noise for real data
python /workspaces/src/filter_noise_dataset.py $real_burst_noise_dataset_path $real_filtered_burst_noise_dir 

# Filter noise for fake data
python /workspaces/src/filter_noise_dataset.py $fake_burst_noise_dataset_path $fake_filtered_burst_noise_dir 