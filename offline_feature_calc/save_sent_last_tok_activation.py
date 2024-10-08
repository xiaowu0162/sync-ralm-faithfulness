import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layers', nargs='+', type=int, help='layers to list', required=True)
    parser.add_argument('--sentence_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--remove_prefix', type=bool, default=False)
    parser.add_argument('--cache_dir', type=str, default=None)
    return parser.parse_args()   


class ActivationHook:
    def __init__(self):
        self._stored = None

    def __call__(self, module, input, output):
        self._stored = output[0]  # hidden states at output[0]

    @property
    def out(self):
        return self._stored

    def remove(self):
        pass

    
def get_acts(example, tokenizer, model, layers, hook_handles, device, prefix=None):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    # get activations
    acts = {layer: [] for layer in layers}
    # tokenized_input = tokenizer(example, return_tensors="pt", padding=True)
    tokenized_input = tokenizer(example, return_tensors="pt", padding=False)
    input_ids = tokenized_input['input_ids'].to(device)
    attention_masks = tokenized_input['attention_mask'].to(device)
    prefix_length = 0
    if prefix:
        tokenized_input_prefix = tokenizer(prefix, return_tensors="pt", padding=False)
        prefix_length = tokenized_input_prefix['input_ids'][0].shape[0]
    length = attention_masks.sum(dim=-1) - prefix_length

    labels = input_ids.clone()
    labels = attention_masks.int() * labels + (1 - attention_masks.int()) * -100
    model_output = model(input_ids, attention_mask=attention_masks, labels=labels, output_hidden_states=True)
    
    for layer, hook in zip(layers, hook_handles):
        acts[layer].append(hook.out[:, prefix_length:])
        
    for layer, act in acts.items():
        acts[layer] = torch.stack(act)[0, :].detach().cpu().float().squeeze().numpy()

    return acts, length

    
def get_sent_to_token_range(sent_original, out_tokens, tokenizer):
    seq_tokenized = out_tokens
    sent = sent_original.lstrip()
    start = 0
    while True:
        if sent in tokenizer.decode(seq_tokenized[start+1:]):
            start += 1
        else:
            break
    end = len(seq_tokenized)
    while True:
        if sent in tokenizer.decode(seq_tokenized[:end-1]):
            end -= 1
        else:
            break
    if start > end:
        return (-1, -1)
    return (start, end)


if __name__ == '__main__':
    args = parse_args()

    # tokenizer and hooked model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").cuda()
    hooks, handles = [], []
    for layer in args.layers:
        hook = ActivationHook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    model.eval()

    sentence_data = [json.loads(line) for line in open(args.sentence_file).readlines()]

    doc_activation_cache = {}

    all_outputs = []
    for sent_entry in tqdm(sentence_data):
        if args.remove_prefix:
            prefix = ""
            prefix_len = 0
        else:
            prefix = sent_entry['prompt']
            prefix_len = tokenizer(prefix, return_tensors="pt", padding=False)['input_ids'][0].shape[0]
        full_sequence = prefix + sent_entry['output_full']
        tokens = tokenizer.encode(full_sequence)[len(tokenizer.encode(prefix)):]
        if sent_entry['id'] not in doc_activation_cache:
            layer2allactivations, cur_length = get_acts(full_sequence, tokenizer, model, args.layers,
                                                        hooks, model.device, prefix=prefix)
            doc_activation_cache[sent_entry['id']] = layer2allactivations
        else:
            print('1 cache hit:', sent_entry['id'])
            layer2allactivations = doc_activation_cache[sent_entry['id']]

        # extract needed activation
        start_idx, end_idx = get_sent_to_token_range(sent_entry['sentence'], tokens, tokenizer)
        print({k: v.shape for k, v in layer2allactivations.items()})
        print((start_idx, end_idx))
        sent_entry['layer_activations'] = {}
        for k, v in layer2allactivations.items():
            try:
                # use end-1 since the span is [start, end)
                sent_entry['layer_activations'][k] = v[end_idx-1]
            except:
                sent_entry['layer_activations'][k] = v[-1]
                print('Warning: sentence out of bounds, use -1 instead')
            
        all_outputs.append(sent_entry)
        
    # remove hooks
    for handle in handles:
        handle.remove()

    np.save(args.out_file, all_outputs)
