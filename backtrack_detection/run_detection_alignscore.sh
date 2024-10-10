#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

root_dir=`realpath ..`
export PYTHONPATH=${PYTHONPATH}:${root_dir}

task=$1                     # bio famous-100 famous-100-anti-v2 QA Summary Data2txt
model=$2                    # llama-2-7b-chat mistral-7B-instruct
split=${3:-"test"}          # test train (only for ragtruth)

metric_model=${root_dir}/AlignScore/checkpoints/AlignScore-base.ckpt

out_dir=logs/${task}_${split}/alignscore/
mkdir -p ${out_dir}

data_dir="${root_dir}/data/sentence_level/${task}/"
sentence_file=${data_dir}/${model}.jsonl
if [[ $task == 'QA' || $task == 'Summary' || $task == 'Data2txt' ]]; then
    data_dir="${root_dir}/data/sentence_level/ragtruth/${task}/${split}/"
    sentence_file=${data_dir}/${model}.jsonl
fi

python run_detection_alignscore.py \
       --task ${task} \
       --sentence_file ${sentence_file} \
       --alignscore_checkpoint ${metric_model} \
       --out_file ${out_dir}/${model}.jsonl        
