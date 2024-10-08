#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export task=$1
export model=$2  
export exp=$3  # no-rag-cxt rag-cxt

root_dir=`realpath ..`

declare -A MODEL_ZOO
MODEL_ZOO["llama-2-7b-chat"]="NousResearch/Llama-2-7b-chat-hf"
MODEL_ZOO["mistral-7B-instruct"]="mistralai/Mistral-7B-Instruct-v0.1"
model_name=${MODEL_ZOO["$model"]}

# replace these two lines with your local directories
model_cache=/local2/diwu/selfrag_model_cache/ 
output_dir=/local2/diwu/rag_analysis_logs/contrastive_logits/${task}/
output_file=${output_dir}/202410_logits_${model}_${exp}.npy
mkdir -p $output_dir

python save_dist_w_wo_ctx.py \
       --model_name ${model_name} \
       --mode ${exp} \
       --root_dir ${root_dir} \
       --out_file ${output_file} \
       --task ${task} \
       --download_dir ${model_cache} 
       