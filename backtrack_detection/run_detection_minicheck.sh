#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

root_dir=`realpath ..`
export PYTHONPATH=${PYTHONPATH}:${root_dir}

task=$1     # bio
model=$2    # llama-2-7b-chat

metric_dir=/home/diwu/eval/MiniCheck
export PYTHONPATH=${PYTHONPATH}:${metric_dir}

out_dir=logs/${task}/minicheck/
mkdir -p ${out_dir}


data_dir="${root_dir}/data/sentence_level/${task}/"
sentence_file=${data_dir}/${model}.jsonl
if [[ $task == 'QA' || $task == 'Summary' || $task == 'Data2txt' ]]; then
    data_dir="${root_dir}/data/sentence_level/ragtruth/${task}/test/"
    sentence_file=${data_dir}/${model}.jsonl
fi

python run_detection_minicheck.py \
       --task ${task} \
       --sentence_file ${sentence_file} \
       --out_file ${out_dir}/${model}.jsonl        
