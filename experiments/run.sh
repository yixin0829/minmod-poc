#!/bin/bash

echo "Running shell script to run experiments with different methods and group size"

# define array of method and group size
models=("llama3-custom")
methods=("batch-qa" "batch-structured" "batch-qa-structured")
batch_size=(2 4 8)

# run qa_vs_structured.py with different method and group size
for model in "${models[@]}"; do
    for method in "${methods[@]}"; do
        for size in "${batch_size[@]}"; do
            echo "Running model: $model with method: $method and batch size: $size"
            python3 experiments/main.py --model $model --method $method --batch_size $size
        done
    done
done
