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
    parser.add_argument('--mode', type=str, choices=['no-rag-cxt', 'rag-cxt'], required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    return parser.parse_args()
    

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_no_input_retrieval": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def main(args): #task, cxt_mode, tokenizer, model, sampling_params, pred_data, out_file):
    model2short = {
        'NousResearch/Llama-2-7b-chat-hf': 'llama-2-7b-chat',
        'NousResearch/Llama-2-13b-chat-hf': 'llama-2-13b-chat',
        'mistralai/Mistral-7B-Instruct-v0.1': 'mistral-7B-instruct',
        'TheBloke/Llama-2-70B-chat-AWQ': 'llama-2-70b-chat'
    }
    model_short = model2short[args.model_name]
    
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

    pred_file_rag = f'{args.root_dir}/data/rag_outputs/{args.task}/{model_short}.json'
    pred_data = [json.loads(line) for line in open(pred_file_rag).readlines()]
    
    # load input and pred trace for logit calc
    out_entries = []
    for entry in tqdm(pred_data):
        out_entry = {}
        
        input_data_for_formatting = {'instruction': entry['question']}
        if args.mode == 'rag-cxt':
            retrieval_result = entry["ctxs"][:10]  # top 10 contexts
            evidences = ["[{}] ".format(i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
            input_data_for_formatting["paragraph"] = "\n".join(evidences)
            prompt_template = PROMPT_DICT['prompt_no_input_retrieval']
        elif args.mode == 'no-rag-cxt':
            prompt_template = PROMPT_DICT['prompt_no_input']
        else:
            raise NotImplementedError
        
        out_entry['prompt'] = prompt_template.format_map(input_data_for_formatting)
        out_entry['prompt_tokens'] = tokenizer.encode(out_entry['prompt'])
        out_entry['output_text'] = entry['output']
        out_entry['output_tokens'] = tokenizer.encode(out_entry['output_text'], add_special_tokens=False)
        assert tokenizer.decode(out_entry['output_tokens']) == entry['output']
        
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
            for tok_id in range(len(tokenizer) - 1):
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
    args = parse_args()
    main(args)
    