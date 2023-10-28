#!/bin/bash

# Workflow 1 - Loading Data: Test for loading a dataset from a directory structure
echo "=== Workflow 1 - Loading Data: Test for loading a dataset from a directory structure ==="

dataset_path='/workspaces/data_sets'

# Fit a new model 
python /workspaces/src/load_data.py $dataset_path