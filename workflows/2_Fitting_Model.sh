#!/bin/bash

# Workflow 2 - Fitting Model: Normal training of the model with the default dataset
echo "=== Workflow 2 - Fitting Model: Normal training of the model with the default dataset ==="

dataset_path='/workspaces/data_sets'
trained_model_path='/workspaces/trained_model'

# Fit a new model 
python /workspaces/src/fit_model.py $dataset_path -tmp $trained_model_path