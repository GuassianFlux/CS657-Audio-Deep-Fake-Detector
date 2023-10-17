#!/bin/bash

# Workflow 1: Normal training and prediction of the dataset with any modifications
echo "=== Workflow 1: Normal training and prediction of the dataset with any modifications ==="

dataset_path='/workspaces/small_data_sets'
trained_model_path='/workspaces/trained_model'
prediction_data_path='/workspaces/prediction_data'

# Fit a new model 
python /workspaces/src/fit_model.py $dataset_path -tmp $trained_model_path

# Make predictions
python /workspaces/src/make_predictions.py $dataset_path -tmp $trained_model_path -pdp $prediction_data_path