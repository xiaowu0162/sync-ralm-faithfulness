#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
export PYTHONPATH=${PYTHONPATH}:`realpath ../..`

export task=$1                  # bio famous-100 famous-100-anti-v2 QA Summary Data2txt
export model=$2                 # llama-2-7b-chat mistral-7B-instruct
export split=${3:-"test"}       # test train (only for ragtruth)

root_dir=`realpath ..`

declare -A MODEL_ZOO
MODEL_ZOO["llama-2-7b-chat"]="NousResearch/Llama-2-7b-chat-hf"
MODEL_ZOO["mistral-7B-instruct"]="mistralai/Mistral-7B-Instruct-v0.1"
model_name=${MODEL_ZOO["$model"]}

# if necessary, replace these two lines with other local directories
model_cache=${root_dir}/model_cache/
output_dir=${root_dir}/rag_analysis_logs/activations_for_lid/${task}/
output_file=${output_dir}/202410_layer_activations_${model}_${split}.npy
mkdir -p $output_dir

if [[ $task == 'bio' || $task == 'famous-100' || $task == 'famous-100-anti' || $task == 'famous-100-anti-v2' ]]; then
    data_dir="${root_dir}/data/sentence_level/${task}/"
elif [[ $task == 'QA' || $task == 'Summary' || $task == 'Data2txt' ]]; then
    data_dir="${root_dir}/data/sentence_level/ragtruth/${task}/${split}/"
fi
sentence_file=${data_dir}/${model}.jsonl

python save_sent_last_tok_activation.py \
       --model ${model_name} \
       --sentence_file ${sentence_file} \
       --out_file ${output_file} \
       --layers 15 16 17 \
       --cache_dir ${model_cache}
       
       

