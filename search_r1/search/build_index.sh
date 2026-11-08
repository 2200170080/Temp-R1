#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
cd ~/Temp-R1/search_r1/search

corpus_file=~/Temp-R1/corpus/full_kg.jsonl
save_dir=~/Temp-R1/corpus/kg_index
retriever_name=e5
retriever_model=intfloat/e5-base-v2

mkdir -p "$save_dir"

CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/retriever/bin/python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
