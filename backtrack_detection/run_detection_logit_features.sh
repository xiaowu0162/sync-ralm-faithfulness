#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

root_dir=`realpath ..`
export PYTHONPATH=${PYTHONPATH}:${root_dir}

task=$1     # bio
model=$2    # llama-2-7b-chat
criterion=${3:-"active_rag"}  # active_rag mean_entropy
threshold=${4:-"3.0"}  # active_rag mean_entropy


out_dir=logs/${task}/${criterion}/
if [[ $criterion == 'neg_large_kl_count' ]]; then 
    out_dir=${out_dir}/threshold${threshold}/
fi
mkdir -p ${out_dir}


declare -A MODEL_ZOO
MODEL_ZOO["llama-2-7b-chat"]="NousResearch/Llama-2-7b-chat-hf"
MODEL_ZOO["llama-2-13b-chat"]="NousResearch/Llama-2-13b-chat-hf"
MODEL_ZOO["llama-2-7b"]="NousResearch/Llama-2-7b-hf"
MODEL_ZOO["llama-2-70b-chat"]="TheBloke/Llama-2-70B-Chat-AWQ"
MODEL_ZOO["mistral-7B-instruct"]="mistralai/Mistral-7B-Instruct-v0.1"
MODEL_ZOO["alpaca-7b"]="${root_dir}/selfrag_model_cache/Llama-2-7b-alpaca-cleaned/"
MODEL_ZOO["selfrag-7b"]="selfrag/selfrag_llama2_7b"
MODEL_ZOO["selfrag-13b"]="selfrag/selfrag_llama2_13b"


if [[ $task == 'bio' ]]; then
    data_dir="${root_dir}/data/sentence_level/${task}/
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'famous-100' ]]; then
    data_dir="${root_dir}/data/sentence_level/${task}/
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'famous-100-anti' ]]; then
    data_dir="${root_dir}/data/sentence_level/${task}/
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'famous-100-anti-v2' ]]; then
    data_dir="${root_dir}/data/sentence_level/${task}/
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'QA' ]]; then
    data_dir="${root_dir}/data/sentence_level/ragtruth/${task}/test/
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'Summary' ]]; then
    data_dir="${root_dir}/data/sentence_level/ragtruth/${task}/test/
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_no-rag-cxt.npy 
elif [[ $task == 'Data2txt' ]]; then
    data_dir="${root_dir}/data/sentence_level/ragtruth/${task}/test/
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_no-rag-cxt.npy
fi


python run_detection_logit_features.py \
       --criterion ${criterion} \
       --kl_threshold ${threshold} \
       --sentence_file $sentence_file \
       --logit_file $logit_file \
       --ref_logit_file $ref_logit_file \
       --out_file ${out_dir}/${model}.jsonl \
       --tokenizer_name ${MODEL_ZOO["$model"]}
       
