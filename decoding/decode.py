import re
import json
import argparse
import torch
from torch.distributions import Categorical
from torch.nn.functional import kl_div
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from nltk import sent_tokenize
from alignscore import AlignScore
import pickle
from backtrack_detection.lid import compute_lid
from transformers import AutoTokenizer, AutoModelForCausalLM
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--source_file', type=str, required=True)
    parser.add_argument('--train_activation_file', type=str, required=True)
    parser.add_argument('--syncheck_checkpoint', type=str, required=True)
    parser.add_argument('--align_score_model_path', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--out_log', type=str, default=None)
    
    parser.add_argument('--search_type', type=str, required=True, choices=['beam', 'bnb', 'cad'])
    
    # parameters specific to the CAD baseline
    parser.add_argument('--cad_alpha', type=float, default=1)
    parser.add_argument('--cad_max_length', type=int, default=1000)
    
    # parameters specific to beam search
    parser.add_argument('--max_sentences', type=int, default=8)
    parser.add_argument('--beam_size', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--sample_size_per_round', type=int, default=6)
    parser.add_argument('--start_beam_search_syncheck_threshold', type=float, default=0.7)
    parser.add_argument('--stop_beam_threshold', type=float, default=0.7)
    
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    
    return parser.parse_args()
    

def format_no_rag(entry, task):
    if task == 'QA':
        prompt_template = 'Briefly answer the following question:\n{}\noutput:'
        return prompt_template.format(entry['source_info']['question'])
    elif task == 'Summary':
        prompt_template = entry['prompt'].split('\n')[0] + '\n{}\noutput:'
        return prompt_template.format(sent_tokenize(entry['source_info'].strip())[0])
    elif task == 'Data2txt':
        prompt_template = 'Instruction:\nWrite an objective overview about the following local business based only on the provided structured data in the JSON format.\n{}\noutput:'
        structure_pruned = {k: v for k, v in entry['source_info'].items() if k != 'review_info'}
        return prompt_template.format(structure_pruned)
    else:
        raise NotImplementedError
    
    
def decode_a_sentence(model, tokenizer, rag_prompt, previous_pred_sents, logprob_num_offset=0,
                      greedy=True, nongreedy_temperature=0.9):
    """
    Decode a sentence as well as return the token-wise contrastive logits
    """
    # todo: add temperature and sample size
    n_logprobs = len(tokenizer)-logprob_num_offset
    assert n_logprobs == 32000
    
    if greedy:
        sampling_params = SamplingParams(temperature=0, top_p=1, logprobs=n_logprobs, 
                                     max_tokens=65)   # max_tokens could be changed based on the dataset
    else:
        sampling_params = SamplingParams(temperature=nongreedy_temperature, top_p=0.95, logprobs=n_logprobs, 
                                     max_tokens=65)   # max_tokens could be changed based on the dataset
    
    text_rag = ''.join(previous_pred_sents)
    next_step_prompt_rag = rag_prompt + text_rag
    next_segment = model.generate([next_step_prompt_rag], sampling_params, use_tqdm=False)[0]
    
    if next_segment.outputs[0].text.strip() == "":
        return '', [], []
        
    # get next sent & tokens
    cur_pred_sent = sent_tokenize(next_segment.outputs[0].text)[0]
    cur_pred_tokens = tokenizer.encode(cur_pred_sent, add_special_tokens=False) # next_tok.token_ids
    if cur_pred_tokens[0] in [29871, 28705]:    # hack for llama and mistral tokenizers
        cur_pred_tokens = cur_pred_tokens[1:]
        
    # get RAG logits
    logits_rag = []  
    try:
        for i_token in range(len(cur_pred_tokens)):
            cur_logprob_dict = next_segment.outputs[0].logprobs[i_token]
            cur_logprobs = []
            for tok_id in range(n_logprobs):
                cur_logprobs.append(cur_logprob_dict[tok_id])
            cur_logprobs = np.array(cur_logprobs)
            logits_rag.append(cur_logprobs)
        logits_rag = np.stack(logits_rag, axis=0)
    except:
        # very rare edge cases
        return '', [], []
    
    return cur_pred_sent, cur_pred_tokens, logits_rag


def get_no_rag_logits(model, tokenizer, no_rag_prompt, previous_pred_sents, cur_pred_sent, cur_pred_tokens, 
                      logprob_num_offset=0):
    """
    Decode a sentence as well as return the token-wise contrastive logits
    """
    n_logprobs = len(tokenizer)-logprob_num_offset
    assert n_logprobs == 32000
    
    text_rag = ''.join(previous_pred_sents)
    
    # get no RAG logits - implementation 1
    # print('Getting no rag logits...', end='')
    logits_no_rag = []
    next_step_prompt_no_rag = no_rag_prompt + text_rag + cur_pred_sent
    sampling_params_no_rag = SamplingParams(temperature=0, top_p=1, prompt_logprobs=n_logprobs, max_tokens=1)
    dummy_segment = model.generate([next_step_prompt_no_rag], sampling_params_no_rag, use_tqdm=False)[0]
    cared_no_rag_logits = dummy_segment.prompt_logprobs[-len(cur_pred_tokens):]
    for i_token in range(len(cur_pred_tokens)):
        cur_logprob_dict = cared_no_rag_logits[i_token]
        cur_logprobs = []
        for tok_id in range(n_logprobs):
            cur_logprobs.append(cur_logprob_dict[tok_id])
        cur_logprobs = np.array(cur_logprobs)
        logits_no_rag.append(cur_logprobs)
    logits_no_rag = np.stack(logits_no_rag, axis=0)
    
    return logits_no_rag


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


def _get_acts(example, tokenizer, model, layers, hook_handles, keep_length=1):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    acts = {layer: [] for layer in layers}
    tokenized_input = tokenizer(example, return_tensors="pt", padding=False)
    input_ids = tokenized_input['input_ids'].to(model.device)
    attention_masks = tokenized_input['attention_mask'].to(model.device)
    length = keep_length # attention_masks.sum(dim=-1) - prefix_length

    labels = input_ids.clone()
    labels = attention_masks.int() * labels + (1 - attention_masks.int()) * -100
    model_output = model(input_ids, attention_mask=attention_masks, labels=labels, output_hidden_states=True)
    
    for layer, hook in zip(layers, hook_handles):
        # acts[layer].append(hook.out[:, prefix_length:])
        acts[layer].append(hook.out[:, -keep_length:])
        
    for layer, act in acts.items():
        acts[layer] = torch.stack(act)[0, :].detach().cpu().float().squeeze().numpy()

    return acts, length


def get_last_tok_activations(tokenizer, model_for_activation, hook_handles, processed_prompt, cur_output, pred_sent, pred_sent_tokens):
    example = processed_prompt + ''.join(processed_prompt) + pred_sent
    acts, length = _get_acts(example, tokenizer, model_for_activation, [15, 16, 17], hook_handles, keep_length=1)
    return acts


def get_lid(pred_sent_last_tok_acts, train_activations, layer_ids):
    ret = {}
    for l in layer_ids:
        lids = compute_lid(pred_sent_last_tok_acts[l].reshape(1, -1),
                           train_activations[l], sample_size=-1, k_list=[train_activations[l].shape[0]-1], metric='l2', block=50000)
        ret[l] = lids[0]
    return ret


def get_alignscore(align_scorer, context_string, pred_sent):
    score = align_scorer.score(contexts=[context_string], claims=[pred_sent.strip()])[0]
    return score


def syncheck(syncheck_model, task, pred_sent_tokens, pred_sent_logits, pred_sent_ref_logits, pred_sent_lids, pred_sent_alignscore):
    # construct features 
    sent_logits = torch.softmax(torch.tensor(pred_sent_logits), axis=-1).numpy()
    sent_ref_logits = torch.softmax(torch.tensor(pred_sent_ref_logits), axis=-1).numpy()
    entropy_with_ctx = [Categorical(probs = torch.tensor(sent_logits[i])).entropy().item()
                        for i in range(sent_logits.shape[0])]
    
    # feat 1
    try:
        max_entropy = np.max(entropy_with_ctx).item()
        if np.isnan(max_entropy):
            max_entropy = 0
    except:
        max_entropy = 0
        
    # feat 2
    mean_entropy = np.mean(entropy_with_ctx).item()
    if np.isnan(mean_entropy):
        mean_entropy = 0
        
    # feat 3, 4, 5
    lid_layer_15 = pred_sent_lids[15]
    lid_layer_16 = pred_sent_lids[16]
    lid_layer_17 = pred_sent_lids[17]
    
    # feat 6, 7
    token_probs = [sent_logits[i, y_i].item() for i, y_i in enumerate(pred_sent_tokens)]
    if len(token_probs) == 0:
        min_prob = 0
        mean_prob = 0
    else:
        min_prob = np.array(token_probs).min().item()            
        mean_prob = np.mean(token_probs).item()   
        
    # feature 8
    kl_divs = [kl_div(torch.tensor(sent_ref_logits[i]).log(), torch.tensor(sent_logits[i]), size_average=False).item()
               for i in range(sent_logits.shape[0])]
    mean_contrastive_kl = np.mean(kl_divs).item()
    if np.isnan(mean_contrastive_kl):
        mean_contrastive_kl = 0
        
    # feature 9
    large_kl_count = 0
    for cur_kl in kl_divs:
        if cur_kl > 3.0:
            large_kl_count += 1
    large_kl_count_threshold = large_kl_count
        
    # get preds
    if task in ['bio', 'famous-100', 'famous-100-anti-v2', 'QA']:
        features = [
            max_entropy, mean_entropy,
            lid_layer_15, lid_layer_16, lid_layer_17,
            min_prob, mean_prob,
            mean_contrastive_kl, large_kl_count_threshold,
            pred_sent_alignscore
        ]
    elif task in ['Summary', 'Data2txt']:
        features = [
            max_entropy, mean_entropy,
            min_prob, mean_prob,
            mean_contrastive_kl, large_kl_count_threshold,
            pred_sent_alignscore
        ]
    else:
        raise NotImplementedError
    
    syncheck_pred = syncheck_model.predict_proba([features])[:,0].item() 
    print('Syncheck (larger the better):', syncheck_pred)
        
    # Note: syncheck is the larger the better
    return syncheck_pred

    
def simple_moving_average(cur_prefix_score, prefix_n_sents, new_score):
    return (new_score + cur_prefix_score * prefix_n_sents) / (prefix_n_sents + 1)
    
    
def beam_search(args, tokenizer, model, model_for_activation, hooks, train_activations, align_scorer, input_entry, syncheck_model, out_log=None):
    # configs
    beam_size = args.beam_size
    sample_size_per_round = args.sample_size_per_round
    # assert sample_size_per_round % beam_size == 0
    max_sentences = args.max_sentences
    syncheck_backtrack_threshold = args.start_beam_search_syncheck_threshold
    stop_beam_threshold = args.stop_beam_threshold
    
    logprob_num_offset = 1 if 'llama' in args.model.lower() else 0
    
    out_entry = input_entry
    processed_prompt = '[INST] ' + out_entry['prompt'] + ' [/INST]'
    processed_prompt_no_rag = '[INST] ' + out_entry['prompt'] + ' [/INST]'
    
    outputs = [([], 1.0)]
    
    # Step 1: greedy search
    print('Entering greedy stage...')
    if out_log:
        print('Entering greedy stage...', file=out_log)
    while len(outputs[0][0]) <= max_sentences:
        cur_output = outputs[0][0]
        pred_sent, pred_sent_tokens, pred_sent_logits = decode_a_sentence(model, tokenizer, processed_prompt, cur_output,
                                                                          logprob_num_offset=logprob_num_offset, greedy=True)
        if pred_sent.strip() == '':
            break
        
        # feature 1 - contrastive logits
        pred_sent_ref_logits = get_no_rag_logits(model, tokenizer, processed_prompt_no_rag, cur_output, 
                                                 pred_sent, pred_sent_tokens, logprob_num_offset=logprob_num_offset)
        
        # feature 2 - activation
        pred_sent_last_tok_acts = get_last_tok_activations(tokenizer, model_for_activation, hooks, processed_prompt, cur_output, pred_sent, pred_sent_tokens)
        
        # feature 3 - lid from activation
        pred_sent_lids = get_lid(pred_sent_last_tok_acts, train_activations, [15, 16, 17])
        
        # feature 4 - alignscore
        pred_sent_alignscore = get_alignscore(align_scorer, input_entry['context_string'], pred_sent)
        
        # Syncheck inference      
        pred_sent_score = syncheck(syncheck_model, args.task, pred_sent_tokens, pred_sent_logits, pred_sent_ref_logits, pred_sent_lids, pred_sent_alignscore)
        
        cur_output_score = simple_moving_average(outputs[0][1], len(outputs[0][0]), pred_sent_score)
        # print(pred_sent, pred_sent_score)
        if pred_sent_score > syncheck_backtrack_threshold:
            outputs = [(cur_output + [pred_sent], cur_output_score)]
        else:
            break
    
    # Step 2: beam search
    print('Backtrack prefix:', outputs[0])
    print('Entering beam stage...')
    if out_log:
        print('Backtrack prefix:', outputs[0], file=out_log)
        print('Entering beam stage...', file=out_log)
    while len(outputs[0][0]) <= max_sentences:
        cur_outputs = []
        
        stop = False
        
        for (cur_output, cur_prefix_score) in outputs:
            cur_round_checked_samples = set()
            for _ in range(sample_size_per_round // beam_size):
                if stop:
                    break
                
                pred_sent, pred_sent_tokens, pred_sent_logits = decode_a_sentence(model, tokenizer, processed_prompt, cur_output, 
                                                                                  logprob_num_offset=logprob_num_offset, greedy=False, nongreedy_temperature=args.temperature)
                
                if pred_sent.strip() == '':
                    stop = True
                    break
                
                if pred_sent in cur_round_checked_samples:
                    continue
                else:
                    cur_round_checked_samples.add(pred_sent)
                
                # feature 1 - contrastive logits
                pred_sent_ref_logits = get_no_rag_logits(model, tokenizer, processed_prompt_no_rag, cur_output, 
                                                         pred_sent, pred_sent_tokens, logprob_num_offset=logprob_num_offset)
                
                # feature 2 - activation
                pred_sent_last_tok_acts = get_last_tok_activations(tokenizer, model_for_activation, hooks, processed_prompt, cur_output, pred_sent, pred_sent_tokens)
                
                # feature 3 - lid from activation
                pred_sent_lids = get_lid(pred_sent_last_tok_acts, train_activations, [15, 16, 17])
                
                # feature 4 - alignscore
                pred_sent_alignscore = get_alignscore(align_scorer, input_entry['context_string'], pred_sent)
                
                # Syncheck inference  
                pred_sent_score = syncheck(syncheck_model, args.task, pred_sent_tokens, pred_sent_logits, pred_sent_ref_logits,
                                           pred_sent_lids, pred_sent_alignscore)
                
                # if score is less than a threshold, skip the sample
                if pred_sent_score < stop_beam_threshold:
                    continue
                
                cur_output_score = simple_moving_average(cur_prefix_score, len(cur_output), pred_sent_score)
                cur_outputs.append((cur_output + [pred_sent], cur_output_score))
            
        if stop:
            break
        
        # if all the beams fail to have syncheck scores above a threshold, break
        if not cur_outputs:
            break
        
        cur_outputs.sort(key=lambda x: x[-1], reverse=True)
        print('\nOne step result:')
        print(json.dumps(cur_outputs, indent=4))
        if out_log:
            print('\nOne step result:', file=out_log)
            print(json.dumps(cur_outputs, indent=4), file=out_log)
        outputs = cur_outputs[:beam_size]
        
    # outputs are already sorted
    cur_output = outputs[0]
    print('\nFinal prediction:')
    print(json.dumps(cur_output, indent=4))
    print('\n===================================\n')
    if out_log:
        print('\nFinal prediction:', file=out_log)
        print(json.dumps(cur_output, indent=4), file=out_log)
        print('\n===================================\n', file=out_log, flush=True)
    
    cur_output = ''.join(cur_output[0])
    if args.task in ['QA', 'Summary', 'Data2txt']:
        out_entry['response'] = cur_output
    else:
        out_entry['output'] = cur_output
        
    return out_entry


def cad_decoding(args, tokenizer, model, input_entry, out_log=None):
    logprob_num_offset = 1 if 'llama' in args.model.lower() else 0
    
    out_entry = input_entry
    processed_prompt = '[INST] ' + out_entry['prompt'] + ' [/INST]'
    processed_prompt_no_rag = '[INST] ' + out_entry['prompt'] + ' [/INST]'
    
    # token-level contrastive search
    n_logprobs = 32000
    output_tokens = []
    sampling_params = SamplingParams(temperature=0, top_p=1, logprobs=n_logprobs, max_tokens=1)
    while len(output_tokens) < args.cad_max_length:
            
        #n_logprobs = len(tokenizer)-logprob_num_offset
        #assert n_logprobs == 32000
        
        text_rag = tokenizer.decode(output_tokens, skip_special_tokens=True)
        next_step_prompt_rag = processed_prompt + text_rag
        next_segment = model.generate([next_step_prompt_rag], sampling_params, use_tqdm=False)[0]
        
        # get RAG logits
        logits_rag = []  
        cur_logprob_dict = next_segment.outputs[0].logprobs[0]
        for tok_id in range(n_logprobs):
            logits_rag.append(cur_logprob_dict[tok_id])
        logits_rag = np.array(logits_rag)
        
        # get no RAG logits
        logits_no_rag = []
        next_step_prompt_no_rag = processed_prompt_no_rag + text_rag
        norag_next_segment = model.generate([next_step_prompt_no_rag], sampling_params, use_tqdm=False)[0]
        cur_logprob_dict = norag_next_segment.outputs[0].logprobs[0]
        for tok_id in range(n_logprobs):
            logits_no_rag.append(cur_logprob_dict[tok_id])
        logits_no_rag = np.array(logits_no_rag)
        
        # merge 
        logits_all = (1 + args.cad_alpha) * logits_rag - args.cad_alpha * logits_no_rag
        next_token = logits_all.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        else:
            output_tokens.append(next_token)
    
    outputs = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print('\nFinal prediction:')
    print(outputs)
    print('\n===================================\n')
    if out_log:
        print('\nFinal prediction:', file=out_log)
        print(outputs, file=out_log)
        print('\n===================================\n', file=out_log, flush=True)
    
    if args.task in ['QA', 'Summary', 'Data2txt']:
        out_entry['response'] = outputs
    else:
        out_entry['output'] = outputs

    return out_entry


def main(args):
    # load data
    input_data = [json.loads(line) for line in open(args.input_file).readlines()]
    id_to_source_data = {}
    if args.task in ['QA', 'Summary', 'Data2txt']:
        for line in open(args.source_file).readlines():
            entry = json.loads(line)
            id_to_source_data[entry['source_id']] = entry
            
    # prepare prompt
    for i in range(len(input_data)):
        if args.task in ['QA', 'Summary', 'Data2txt']:
            source_entry = id_to_source_data[input_data[i]['source_id']]
            input_data[i]['prompt'] = source_entry['prompt']
            input_data[i]['no_rag_prompt'] = format_no_rag(source_entry, args.task)
            
            if args.task == 'QA':
                instruction, passage = source_entry['prompt'].split('passage 1:')
                input_data[i]['context_string'] = 'passage 1:' + passage
            elif args.task == 'Summary':
                instruction, passage = source_entry['prompt'].split('words:')[0], source_entry['prompt'].split('words:')[1:]
                input_data[i]['context_string'] = 'words:' + 'words:'.join(passage)
            elif args.task == 'Data2txt':
                instruction, passage = source_entry['prompt'].split('Structured data:\n')
                input_data[i]['context_string'] = 'Structured data:\n' + passage
            else:
                raise NotImplementedError
        else:
            # biography
            assert args.task in ['bio', 'famous-100', 'famous-100-anti-v2']
            retrieval_result = input_data[i]['ctxs']
            evidences = ["[{}] ".format(j+1) + ctx["title"]+"\n" + ctx["text"] for j, ctx in enumerate(retrieval_result[:10])]
            evidence_string = "\n".join(evidences)
            input_data[i]['prompt'] =  "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Paragraph:\n{}\n\n### Instruction:\n{}\n\n### Response:".format(evidence_string, input_data[i]['input'])
            input_data[i]['no_rag_prompt'] =  "### Instruction:\n{}\n\n### Response:\n".format(input_data[i]['input'])
            input_data[i]['context_string'] = evidence_string
    
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = LLM(model=args.model, download_dir=args.download_dir, gpu_memory_utilization=0.8,
                quantization='awq' if 'awq' in args.model.lower() else None)
    
    # load model for activation
    if args.search_type in ['beam']:
        assert torch.cuda.device_count() == 2
        model_for_activation = AutoModelForCausalLM.from_pretrained(args.model, cache_dir='/local2/diwu/hf_cache/',
                                                                    torch_dtype=torch.bfloat16,
                                                                    attn_implementation="flash_attention_2").to('cuda:1')
        hooks, handles = [], []
        for layer in [15, 16, 17]:
            hook = ActivationHook()
            handle = model_for_activation.model.layers[layer].register_forward_hook(hook)
            hooks.append(hook), handles.append(handle)
        model_for_activation.eval()
        
        # load train activations for LID
        train_activations_all = np.load(args.train_activation_file, allow_pickle=True)
        try:
            train_activations_filtered = [x for x in train_activations_all if (not x['is_conflict']) and (not x['is_baseless'])]
        except:
            train_activations_filtered = [x for x in train_activations_all if not x['is_baseless']]
        train_activations_filtered = train_activations_filtered[:2000]
        train_activations_final = {}
        for layer in [15, 16, 17]:
            train_activations_final[layer] = [x['layer_activations'][layer] for x in train_activations_filtered]
            train_activations_final[layer] = np.array(train_activations_final[layer]).astype('float32')
        print('LID:', len(train_activations_all), 'pos+neg instances ->', len(train_activations_filtered), 'training samples')
        
        # load model for alignscore
        align_scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', 
                                  ckpt_path=args.align_score_model_path, # '/home/diwu/eval/AlignScore/checkpoints/AlignScore-base.ckpt', 
                                  evaluation_mode='nli_sp', verbose=False)
        
        # load syncheck model
        syncheck_model = pickle.load(open(args.syncheck_checkpoint, 'rb'))
        
    else:
        model_for_activation = None
        train_activations_final = {}
        align_scorer = None
        syncheck_model = None
        
    # decoding
    with open(args.out_file, 'w') as out_f:
        out_log = open(args.out_log, 'w') if args.out_log else None
        for in_entry in tqdm(input_data):
            try:
                if args.search_type == 'beam':
                    out_entry = beam_search(args, tokenizer, model, model_for_activation, hooks, train_activations_final, align_scorer, in_entry, syncheck_model, out_log)
                elif args.search_type == 'cad':
                    out_entry = cad_decoding(args, tokenizer, model, in_entry, out_log)
                else:
                    raise NotImplementedError
                print(json.dumps(out_entry), file=out_f, flush=True)
            except:
                print('**One Exception**', in_entry, file=out_log)
                print(json.dumps(in_entry), file=out_f, flush=True)
        out_log.close()
        

if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    main(args)
