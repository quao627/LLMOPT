#!/bin/bash

# List of all datasets
datasets=(
    "complexor.json"
    "industryor_test.json"
    "mamo_complex_test.json"
    "mamo_easy.json"
    "nl4opt_test.json"
    "nlp4lp_test.json"
    "optibench.json"
)


# Run evaluation for all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running evaluation for $dataset with $model"
        python eval.py --data_dir "../baseline_test_data" --dataset_file "$dataset"
    done
done

echo "All evaluations completed!"
