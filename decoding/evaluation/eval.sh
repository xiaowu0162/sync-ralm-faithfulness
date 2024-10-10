#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
export HOME_DIR=`realpath ../..`
export PYTHONPATH=${PYTHONPATH}:${HOME_DIR}


task=$1
pred_file=$2


if [[ $task == 'bio' ]]; then
    support_file=${HOME_DIR}/data/instance_level/factscore_unlabeled_alpaca_13b_retrieval.jsonl 
elif [[ $task == 'famous-100' ]]; then
    support_file=${HOME_DIR}/data/instance_level/famous_people_100_with_wiki.jsonl
elif [[ $task == 'famous-100-anti-v2' ]]; then
    support_file=${HOME_DIR}/data/instance_level/famous_people_100_with_anti_wiki_v2.jsonl
elif [[ $task == 'QA' || $task == 'Summary' || $task == 'Data2txt' ]]; then
    support_file=${HOME_DIR}/data/instance_level/ragtruth/data/all_split/${task}/source_info.jsonl
fi


python run_faithfulness_eval.py \
       --task ${task} \
       --pred_file ${pred_file} \
       --support_file ${support_file} \
       --log_file ${pred_file}.eval.faithfulness.log \
       --decomp_backend densexretrieval \
       --openai_key openai.key \
       --cache_dir ${HOME_DIR}/model_cache/factscore/

