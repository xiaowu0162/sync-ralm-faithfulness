import os
import json
import argparse
import torch
from torch.distributions import Categorical
from torch.nn.functional import kl_div
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from backtrack_detection.utils.split_sentences import split_sentences


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--sentence_file', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--criterion', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    return parser.parse_args()   


selfrag_prompt = "### Instruction:\n{}\n\n### Response:\n[Retrieval]<paragraph>{}</paragraph>[Relevant]{}"


def get_preds_selfrag(task, sentence_entries, criterion, model_name):
    model = LLM(model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=10, logprobs=len(tokenizer))
    
    tok_id_sup, tok_id_part_sup, tok_id_no_sup = tokenizer.encode('[Fully supported][Partially supported][No support / Contradictory]', add_special_tokens=False)
    
    outputs = []
    for sent_entry in tqdm(sentence_entries):
        if criterion == 'selfrag_sentence':
            output = (sent_entry['output_prefix'] + ' ' + sent_entry['sentence']).strip()
        elif criterion == 'selfrag_sample':
            output = sent_entry['output_full']
        else:
            raise NotImplementedError
        
        if task in ['QA']:
            instruction, passage = sent_entry['prompt'].split('passage 1:')
            passage = 'passage 1:' + passage
        elif task in ['Summary']:
            instruction, passage = sent_entry['prompt'].split('words:')[0], sent_entry['prompt'].split('words:')[1:]
            passage = 'words:' + 'words:'.join(passage)
        elif task in ['Data2txt']:
            instruction, passage = sent_entry['prompt'].split('Structured data:\n')
            passage = 'Structured data:\n' + passage
        elif task in ['bio', 'famous-100', 'famous-100-anti', 'famous-100-anti-v2']:
            passage, instruction = sent_entry['prompt'].split('### Instruction:\n')
            instruction = '### Instruction:\n' + instruction
            passage = passage.split('### Paragraph:\n')[-1]
        else:
            raise NotImplementedError
        
        # collect support token value
        prompt = selfrag_prompt.format(instruction, passage, output)
        out = model.generate([prompt], sampling_params, use_tqdm=False)
        cur_logprob_dict = out[0].outputs[0].logprobs[0]
        cur_logprobs = []
        for tok_id in range(len(tokenizer)):
            cur_logprobs.append(cur_logprob_dict[tok_id])
        cur_logprobs = torch.tensor(cur_logprobs)
        cur_logprobs = torch.softmax(cur_logprobs, dim=-1)
        
        # strategy 1
        score = cur_logprobs[tok_id_sup] / (cur_logprobs[tok_id_part_sup] + cur_logprobs[tok_id_no_sup])
        sent_entry['pred_val'] = - score.cpu().numpy().item()   # large = more likely to hallucinate
        
        # strategy 2
        # score = cur_logprobs[tok_id_no_sup] / cur_logprobs[tok_id_sup]
        # sent_entry['pred_val'] = score.cpu().numpy().item()   # large = more likely to hallucinate
        
        print(cur_logprobs[tok_id_sup], cur_logprobs[tok_id_part_sup], cur_logprobs[tok_id_no_sup])
        print(score)        
        
        outputs.append(sent_entry)
        
    return outputs

if __name__ == '__main__':
    args = parse_args()
    
    sentence_data = [json.loads(line) for line in open(args.sentence_file).readlines()]

    all_outputs = get_preds_selfrag(args.task, sentence_data, args.criterion, args.model_name)
    out_f = open(args.out_file, 'w')
    for out_entry in all_outputs:
        print(json.dumps(out_entry), file=out_f)
        
        
