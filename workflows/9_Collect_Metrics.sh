#!/bin/bash

# Workflow 9 - Collect Metrics: Evaluate the model against datasets
echo "=== Workflow 9 - Collect Metrics: Evaluate the model against datasets ==="
default=1
count=${1:-$default}

dataset_path='/workspaces/data_sets'
white_noise_dataset_path='/workspaces/white_noise_data_sets'
burst_noise_dataset_path='/workspaces/burst_noise_data_sets'
filtered_white_noise_dataset_path='/workspaces/filtered_white_noise_data_sets'
filtered_burst_noise_dataset_path='/workspaces/filtered_burst_noise_data_sets'

trained_model_path='/workspaces/trained_model'
prediction_data_path='/workspaces/prediction_data'

metrics_root_path='/workspaces/metrics'

if [ -d $metrics_root_path ]; then
    rm -r $metrics_root_path
fi
mkdir $metrics_root_path
mkdir $metrics_root_path/baseline
mkdir $metrics_root_path/white_noise_train
mkdir $metrics_root_path/burst_noise_train
mkdir $metrics_root_path/white_noise_no_train
mkdir $metrics_root_path/burst_noise_no_train
mkdir $metrics_root_path/filtered_white_noise
mkdir $metrics_root_path/filtered_burst_noise


for i in $(seq 1 $count)
do
    python /workspaces/src/fit_model.py $dataset_path -tmp $trained_model_path
    python /workspaces/src/make_predictions.py $dataset_path -tmp $trained_model_path -pdp $prediction_data_path > $metrics_root_path/baseline/metric_$i.txt
    python /workspaces/src/make_predictions.py $white_noise_dataset_path -tmp $trained_model_path -pdp $prediction_data_path > $metrics_root_path/white_noise_no_train/metric_$i.txt
    python /workspaces/src/make_predictions.py $burst_noise_dataset_path -tmp $trained_model_path -pdp $prediction_data_path > $metrics_root_path/burst_noise_no_train/metric_$i.txt
    python /workspaces/src/make_predictions.py $filtered_white_noise_dataset_path -tmp $trained_model_path -pdp $prediction_data_path > $metrics_root_path/filtered_white_noise/metric_$i.txt
    python /workspaces/src/make_predictions.py $filtered_burst_noise_dataset_path -tmp $trained_model_path -pdp $prediction_data_path > $metrics_root_path/filtered_burst_noise/metric_$i.txt
    python /workspaces/src/fit_model.py $white_noise_dataset_path -tmp $trained_model_path
    python /workspaces/src/make_predictions.py $white_noise_dataset_path -tmp $trained_model_path -pdp $prediction_data_path > $metrics_root_path/white_noise_train/metric_$i.txt
    python /workspaces/src/fit_model.py $burst_noise_dataset_path -tmp $trained_model_path
    python /workspaces/src/make_predictions.py $burst_noise_dataset_path -tmp $trained_model_path -pdp $prediction_data_path > $metrics_root_path/burst_noise_train/metric_$i.txt
done

echo 'Baseline Training and Prediction:'
python /workspaces/src/parse_metrics.py $metrics_root_path/baseline
echo '\nBaseline Training and White Noise Prediction:'
python /workspaces/src/parse_metrics.py $metrics_root_path/white_noise_no_train
echo '\nWhite Noise Training and Prediction:'
python /workspaces/src/parse_metrics.py $metrics_root_path/white_noise_train
echo '\nBaseline Training and Burst Noise Prediction:'
python /workspaces/src/parse_metrics.py $metrics_root_path/filtered_white_noise
echo '\nBaseline Training and Filtered Burst Noise Prediction:'
python /workspaces/src/parse_metrics.py $metrics_root_path/burst_noise_no_train
echo '\nBurst Noise Training and Prediction:'
python /workspaces/src/parse_metrics.py $metrics_root_path/burst_noise_train
echo '\nBaseline Training and Filtered White Noise Prediction:'
python /workspaces/src/parse_metrics.py $metrics_root_path/filtered_burst_noise

