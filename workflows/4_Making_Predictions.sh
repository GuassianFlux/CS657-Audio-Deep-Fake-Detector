#!/bin/bash

# Workflow 2: Prediction of the dataset with existing model
echo "=== Workflow 4: Prediction of the dataset with existing model ==="

dataset_path='/workspaces/data_sets'
trained_model_path='/workspaces/trained_model'
prediction_data_path='/workspaces/prediction_data'

# Make predictions
python /workspaces/src/make_predictions.py $dataset_path -tmp $trained_model_path -pdp $prediction_data_path