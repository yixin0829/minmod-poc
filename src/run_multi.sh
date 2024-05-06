#!/bin/bash

echo "Running shell script to run qa_vs_structured.py with different method and group size"

# define array of method and group size
methods=("1shot_multi_field_qa")
group_size=(2 4 8)

# run qa_vs_structured.py with different method and group size
for m in "${methods[@]}"; do
    for size in "${group_size[@]}"; do
        echo "Running method: $m with group size: $size"
        python3 src/qa_vs_structure.py --model llama3-custom --fixed_samples 1 --method $m --group_size $size
    done
done
