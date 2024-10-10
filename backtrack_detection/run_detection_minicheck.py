import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from backtrack_detection.utils.split_sentences import split_sentences
from minicheck.minicheck import MiniCheck


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--sentence_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    return parser.parse_args()   


def get_preds_minicheck(task, sentence_entries):
    scorer = MiniCheck(model_name='flan-t5-large', device=f'cuda:0', cache_dir='/local2/diwu/hf_cache/')
    
    outputs = []
    for sent_entry in tqdm(sentence_entries):
        output = sent_entry['sentence'].strip()
        
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
        pred_label, raw_prob, _, _ = scorer.score(docs=[passage], claims=[output]) 
        sent_entry['pred_val'] = -1 * raw_prob[0]
        print(raw_prob)
        
        outputs.append(sent_entry)
        
    return outputs

if __name__ == '__main__':
    args = parse_args()
    
    sentence_data = [json.loads(line) for line in open(args.sentence_file).readlines()]

    all_outputs = get_preds_minicheck(args.task, sentence_data)
    out_f = open(args.out_file, 'w')
    for out_entry in all_outputs:
        print(json.dumps(out_entry), file=out_f)
        
        
