#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

root_dir=`realpath ..`
export PYTHONPATH=${PYTHONPATH}:${root_dir}

task=$1                     # bio famous-100 famous-100-anti-v2 QA Summary Data2txt
model=$2                    # llama-2-7b-chat mistral-7B-instruct
split=${3:-"test"}          # test train (only for ragtruth)
threshold=3.0 

out_dir=${root_dir}/rag_analysis_logs/syncheck_aggregated_features/${task}/
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
    training_source='famous-100'
    data_dir=`realpath data/${task}/`
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'famous-100' ]]; then
    training_source='bio'
    data_dir=`realpath data/${task}/`
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'famous-100-anti' ]]; then
    training_source='bio'
    data_dir=`realpath data/${task}/`
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'famous-100-anti-v2' ]]; then
    training_source='bio'
    data_dir=`realpath data/${task}/`
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_${task}_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'QA' ]]; then
    training_source='self'
    data_dir=`realpath data/ragtruth/${task}/${split}/`
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_no-rag-cxt.npy
elif [[ $task == 'Summary' ]]; then
    training_source='self'
    data_dir=`realpath data/ragtruth/${task}/${split}/`
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_no-rag-cxt.npy 
elif [[ $task == 'Data2txt' ]]; then
    training_source='self'
    data_dir=`realpath data/ragtruth/${task}/${split}/`
    sentence_file=${data_dir}/${model}.jsonl
    logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_rag-cxt.npy
    ref_logit_file=${root_dir}/rag_analysis_logs/contrastive_logits/${task}/202410_logits_${model}_no-rag-cxt.npy
fi

test_activation_file=${root_dir}/rag_analysis_logs/activations_for_lid/${task}/202410_layer_activations_${model}_${split}.npy
if [[ $training_source == 'self' ]]; then
    train_activation_file=${root_dir}/rag_analysis_logs/activations_for_lid/${task}/202410_layer_activations_${model}_train.npy
elif [[ $training_source == 'bio' || $training_source == 'famous-100' || $training_source == 'famous-100-anti' ]]; then
    train_activation_file=${root_dir}/rag_analysis_logs/activations_for_lid/${training_source}/202410_layer_activations_${model}_test.npy
else
    train_activation_file=${root_dir}/rag_analysis_logs/activations_for_lid/${training_source}/202410_layer_activations_${model}_train.npy
fi

if [[ $split == 'train' ]]; then   
    alignscore_file=logs/${task}_train/alignscore/${model}.jsonl
else
    alignscore_file=logs/${task}_test/alignscore/${model}.jsonl
fi

python aggregate_features.py \
       --kl_threshold ${threshold} \
       --sentence_file $sentence_file \
       --train_activation_file $train_activation_file \
       --test_activation_file $test_activation_file \
       --logit_file $logit_file \
       --ref_logit_file $ref_logit_file \
       --alignscore_preds_file $alignscore_file \
       --out_file ${out_dir}/${model}_${split}.npy \
       --tokenizer_name ${MODEL_ZOO["$model"]}
