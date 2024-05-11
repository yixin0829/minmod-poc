
#!/bin/bash

echo $SHELL
echo "Running shell script to run qa_vs_structured.py with single QA and structured group"

# define array of method and group size
methods=("2shot_qa" "2shot_structured_w_retry")
group_size=(1)

# run qa_vs_structured.py with different method and group size
for m in "${methods[@]}"; do
    for size in "${group_size[@]}"; do
        echo "Running method: $m with group size: $size"
        python3 src/qa_vs_structure.py --model llama3-custom --method $m --group_size $size
    done
done
