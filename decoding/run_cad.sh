#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
export HOME_DIR=`realpath ..`
export PYTHONPATH=${PYTHONPATH}:${HOME_DIR}

task=$1         # bio, famous-100, famous-100-anti-v2, QA, Summary, Data2txt
model=$2        # llama-2-7b-chat, mistral-7B-instruct
search_type=cad
cad_alpha=$3     # 1.0
cad_max_length=1000

log_dir=logs/${task}/
mkdir -p ${log_dir}

out_file=${log_dir}/${model}_${search_type}_alpha${cad_alpha}_maxlen${cad_max_length}.jsonl
out_log=${out_file}.log


declare -A MODEL_ZOO
MODEL_ZOO["llama-2-7b-chat"]="NousResearch/Llama-2-7b-chat-hf"
MODEL_ZOO["llama-2-13b-chat"]="NousResearch/Llama-2-13b-chat-hf"
MODEL_ZOO["llama-2-70b-chat"]="TheBloke/Llama-2-70B-Chat-AWQ"
MODEL_ZOO["mistral-7B-instruct"]="mistralai/Mistral-7B-Instruct-v0.1"
model_name=${MODEL_ZOO["$model"]}


# data configs
if [[ $task == 'bio' ]]; then
    input_file=${HOME_DIR}/data/instance_level/factscore_unlabeled_alpaca_13b_retrieval.jsonl 
    source_file=${input_file}
    training_source='famous-100'
    syncheck_checkpoint=${HOME_DIR}/syncheck_checkpoints/syncheck_${model}_${training_source}.pkl
elif [[ $task == 'famous-100' ]]; then
    input_file=${HOME_DIR}/data/instance_level/famous_people_100_with_wiki.jsonl
    source_file=${input_file}
    training_source='bio'
    syncheck_checkpoint=${HOME_DIR}/syncheck_checkpoints/syncheck_${model}_${training_source}.pkl
elif [[ $task == 'famous-100-anti-v2' ]]; then
    input_file=${HOME_DIR}/data/instance_level/famous_people_100_with_anti_wiki_v2.jsonl
    source_file=${input_file}
    training_source='bio'
    syncheck_checkpoint=${HOME_DIR}/syncheck_checkpoints/syncheck_${model}_${training_source}.pkl
elif [[ $task == 'QA' || $task == 'Summary' || $task == 'Data2txt' ]]; then
    input_file=/home/diwu/ralm/when-to-retrieve/ragtruth/data/all_split/${task}/test/${model}.jsonl
    source_file=/home/diwu/ralm/when-to-retrieve/ragtruth/data/all_split/${task}/source_info.jsonl
    training_source='self'
    syncheck_checkpoint=${HOME_DIR}/syncheck_checkpoints/syncheck_${model}_${task}.pkl
fi

# activation files
if [[ $training_source == 'self' ]]; then
    train_activation_file=${HOME_DIR}/rag_analysis_logs/activations_for_lid/${task}/202410_layer_activations_${model}_train.npy
elif [[ $training_source == 'bio' || $training_source == 'famous-100' || $training_source == 'famous-100-anti' ]]; then
    train_activation_file=${HOME_DIR}/rag_analysis_logs/activations_for_lid/${training_source}/202410_layer_activations_${model}_test.npy
else
    train_activation_file=${HOME_DIR}/rag_analysis_logs/activations_for_lid/${training_source}/202410_layer_activations_${model}_train.npy
fi


python decode.py \
    --task ${task} \
    --search_type ${search_type} \
    --cad_alpha ${cad_alpha} \
    --cad_max_length ${cad_max_length} \
    --model ${MODEL_ZOO["$model"]} \
    --download_dir ${HOME_DIR}/model_cache/ \
    --input_file ${input_file} \
    --source_file ${source_file} \
    --train_activation_file ${train_activation_file} \
    --syncheck_checkpoint ${syncheck_checkpoint} \
    --out_file ${out_file} \
    --out_log ${out_log}
