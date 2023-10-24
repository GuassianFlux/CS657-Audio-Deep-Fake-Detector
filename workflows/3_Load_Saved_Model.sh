#!/bin/bash

# Workflow 3 - Load Saved Model: Loading a model that has been saved to file
echo "=== Workflow 3 - Load Saved Model: Loading a model that has been saved to file ==="

trained_model_path='/workspaces/trained_model'

# Load saved model
python /workspaces/src/load_saved_model.py $trained_model_path