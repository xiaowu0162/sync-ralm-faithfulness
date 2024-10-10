#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export task=$1     # bio famous-100 famous-100-anti-v2 QA Summary Data2txt
export model=$2    # llama-2-7b-chat mistral-7B-instruct
export exp=$3      # no-rag-cxt rag-cxt

root_dir=`realpath ..`

declare -A MODEL_ZOO
MODEL_ZOO["llama-2-7b-chat"]="NousResearch/Llama-2-7b-chat-hf"
MODEL_ZOO["mistral-7B-instruct"]="mistralai/Mistral-7B-Instruct-v0.1"
model_name=${MODEL_ZOO["$model"]}

# if necessary, replace these two lines with other local directories
model_cache=${root_dir}/model_cache/ 
output_dir=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/
output_file=${output_dir}/202410_logits_${model}_${exp}.npy
mkdir -p $output_dir

if [[ $task == 'bio' || $task == 'famous-100' || $task == 'famous-100-anti-v2' ]]; then
       script="save_dist_w_wo_ctx_bio.py"
elif [[ $task == 'QA' || $task == 'Summary' || $task == 'Data2txt' ]]; then
       script="save_dist_w_wo_ctx_ragtruth.py"
else
       echo "Unrecognized task"
       exit
fi

python $script \
       --model_name ${model_name} \
       --mode ${exp} \
       --root_dir ${root_dir} \
       --out_file ${output_file} \
       --task ${task} \
       --download_dir ${model_cache} 
       