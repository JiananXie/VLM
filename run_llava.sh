#! /bin/bash
CUDA_VISIBLE_DEVICES=0 python llava_atten.py \
    --model-path models/llava-v1.6-mistral-7b \
    --image-file "view_336.jpg" \
    --query "What are the things I should be cautious about when I visit this place? Are there any dangerous areas or activities I should avoid? Or any other important information I should know?"
