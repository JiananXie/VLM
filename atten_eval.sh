#! /bin/bash


model_path="models/llava-phi2"
image_file="view_336.jpg"
query="What are the things I should be cautious about when I visit this place? Are there any dangerous areas or activities I should avoid? Or any other important information I should know?"


if [[ $model_path == *"llava"* ]]; then 
    CUDA_VISIBLE_DEVICES=0 python run/run_llava.py \
        --model-path $model_path \
        --image-file $image_file \
        --query "$query"
elif [[ $model_path == *"bunny"* ]]; then 
    CUDA_VISIBLE_DEVICES=0 python run/run_bunny.py \
        --model-path $model_path \
        --model-type "stablelm-2" \
        --image-file $image_file \
        --query "$query" \
        --conv-mode "bunny"
elif [[ $model_path == *"idefics"* ]]; then 
    # conda activate mllm
    CUDA_VISIBLE_DEVICES=0 python run/run_idefics.py \
        --model-path $model_path \
        --image-file $image_file \
        --query "$query"
elif [[ $model_path == *"qwen2"* ]]; then 
    # conda activate mllm
    CUDA_VISIBLE_DEVICES=0 python run/run_qwen2vl.py \
        --model-path $model_path \
        --image-file $image_file \
        --query "$query"
fi