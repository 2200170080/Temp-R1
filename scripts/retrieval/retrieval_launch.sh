#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
cd ~/Temp-R1
export CUDA_VISIBLE_DEVICES=0

~/miniconda3/envs/retriever/bin/python search_r1/search/retrieval_server.py \
    --index_path corpus/kg_index/e5_Flat.index \
    --corpus_path corpus/full_kg.jsonl \
    --topk 30 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu \
    --port 8000
