#!/bin/bash

echo "Running shell script to run qa_vs_structured.py with different method and group size"

# define array of method and group size
# methods=("1shot_multi_field_qa", "1shot_multi_field_structured")
# methods=("1shot_multi_field_qa")
methods=("1shot_multi_field_structured")
fixed_samples=(0 1)
group_size=(2 4 8)

# run qa_vs_structured.py with different method and group size
for m in "${methods[@]}"; do
    for fixed in "${fixed_samples[@]}"; do
        for size in "${group_size[@]}"; do
            echo "Running method: $m with group size: $size and fixed samples: $fixed"
            python3 src/qa_vs_structure.py --model llama3-custom --fixed_samples $fixed --method $m --group_size $size
        done
    done
done
