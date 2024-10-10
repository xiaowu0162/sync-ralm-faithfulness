import os
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--surrogate_model', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['no-rag-cxt', 'rag-cxt'], required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    return parser.parse_args()
    

def main():
    args = parse_args()

    model2short = {
        'NousResearch/Llama-2-7b-chat-hf': 'llama-2-7b-chat',
        'NousResearch/Llama-2-13b-chat-hf': 'llama-2-13b-chat',
        'mistralai/Mistral-7B-Instruct-v0.1': 'mistral-7B-instruct',
        'TheBloke/Llama-2-70B-chat-AWQ': 'llama-2-70b-chat'
    }
    model_short = model2short[args.model_name]
        
    pred_file_no_rag = f'{args.root_dir}/data/rag_outputs/ragtruth/{args.task}/no-rag/{model_short}.json'
    no_rag_prompts = ['[INST] ' + json.loads(line)['input'] + ' [/INST]' for line in open(pred_file_no_rag).readlines()]

    pred_file_rag = f'{args.root_dir}/data/instance_level/ragtruth/task_model_split/{args.task}/output_only/{model_short}.jsonl'
    pred_data = [json.loads(line) for line in open(pred_file_rag).readlines()]
    rag_prompts = ['[INST] ' + json.loads(line)['input'] + ' [/INST]' for line in open(pred_file_rag).readlines()]
    
    # load model
    if args.surrogate_model:
        print('Surrogate model specified:', args.surrogate_model)
        tokenizer = AutoTokenizer.from_pretrained(args.surrogate_model)
        model = LLM(model=args.surrogate_model, download_dir=args.download_dir, tensor_parallel_size=1,
                    quantization='awq' if 'awq' in args.surrogate_model.lower() else None)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = LLM(model=args.model_name, download_dir=args.download_dir, tensor_parallel_size=1,
                    quantization='awq' if 'awq' in args.model_name.lower() else None)
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1, logprobs=32000)

    # load input and pred trace for logit calc
    out_entries = []
    for i in tqdm(range(len(pred_data))):
        entry = pred_data[i]
        out_entry = {}
        
        if args.mode == 'rag-cxt':
            cur_prompt = rag_prompts[i]
        elif args.mode == 'no-rag-cxt':
            cur_prompt = no_rag_prompts[i]
        else:
            raise NotImplementedError
        
        out_entry['prompt'] = cur_prompt
        out_entry['prompt_tokens'] = tokenizer.encode(out_entry['prompt'])
        # hack - models usually generate a blank space between [/INST] and the response
        if entry['response'][0] != ' ':
            entry['response'] = ' ' + entry['response']
        out_entry['output_text'] = entry['response']        
        out_entry['output_tokens'] = tokenizer.encode(out_entry['output_text'], add_special_tokens=False)
        assert tokenizer.decode(out_entry['output_tokens']) == entry['response']
        
        # calculate logits
        prefixes_to_validate = [out_entry['prompt_tokens'] + out_entry['output_tokens'][:i] for i in range(len(out_entry['output_tokens'])+1)]
        prefixes_to_validate = [tokenizer.decode(x, skip_special_tokens=True) for x in prefixes_to_validate]
        assert prefixes_to_validate[0] == out_entry['prompt']
        assert prefixes_to_validate[-1] == tokenizer.decode(tokenizer.encode(out_entry['prompt']) 
                                                            + tokenizer.encode(out_entry['output_text'], add_special_tokens=False), skip_special_tokens=True)
        outputs = model.generate(prefixes_to_validate, sampling_params)
        assert len(outputs) == len(prefixes_to_validate)
        seq_logits = []
        for out in outputs:
            cur_logprob_dict = out.outputs[0].logprobs[0]
            cur_logprobs = []
            for tok_id in range(32000):
                cur_logprobs.append(cur_logprob_dict[tok_id])
            cur_logprobs = np.array(cur_logprobs)
            seq_logits.append(cur_logprobs)
        seq_logits = np.stack(seq_logits, axis=0)
        assert seq_logits.shape[0] == len(prefixes_to_validate)
        out_entry['logits'] = seq_logits
                
        out_entries.append(out_entry)

    # save
    np.save(args.out_file, out_entries)
    print(args.out_file)
    
    
if __name__ == '__main__':
    main()
