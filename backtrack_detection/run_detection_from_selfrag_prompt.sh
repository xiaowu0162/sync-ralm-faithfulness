#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

root_dir=`realpath ..`
export PYTHONPATH=${PYTHONPATH}:${root_dir}


task=$1     # bio
model=$2    # llama-2-7b-chat
metric_model=selfrag/selfrag_llama2_7b
criterion=$3   # selfrag_sample   # selfrag_sentence selfrag_sample

out_dir=logs/${task}/${criterion}/
mkdir -p ${out_dir}

data_dir="${root_dir}/data/sentence_level/${task}/"
sentence_file=${data_dir}/${model}.jsonl
if [[ $task == 'QA' || $task == 'Summary' || $task == 'Data2txt' ]]; then
    data_dir=`realpath data/ragtruth/${task}/test/`
    sentence_file=${data_dir}/${model}.jsonl
fi

python run_detection_from_selfrag_prompt.py \
       --task ${task} \
       --model_name ${metric_model} \
       --sentence_file ${sentence_file} \
       --criterion ${criterion} \
       --out_file ${out_dir}/${model}.jsonl        
