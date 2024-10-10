import os
import json
import argparse
import torch
from torch.distributions import Categorical
from torch.nn.functional import kl_div
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from backtrack_detection.utils.split_sentences import split_sentences
from backtrack_detection.lid import compute_lid
import random 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kl_threshold', type=float, default=None)
    parser.add_argument('--sentence_file', type=str, required=True)
    parser.add_argument('--train_activation_file', type=str, required=True)
    parser.add_argument('--test_activation_file', type=str, required=True)
    parser.add_argument('--logit_file', type=str, required=True)
    parser.add_argument('--ref_logit_file', type=str, required=True)
    parser.add_argument('--alignscore_preds_file', type=str, default=None)
    parser.add_argument('--tokenizer_name', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    return parser.parse_args()   


def get_sent_to_token_range(out_sents, out_tokens):
    seq_tokenized = out_tokens
    out_data = {}
    for sent_original in out_sents:
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
        out_data[sent_original] = (start, end)
    return out_data


def get_logit_features(sentence_entries, tokens, logits, ref_logits, threshold):
    sent2span = get_sent_to_token_range([x['sentence'] for x in sentence_entries], tokens)
    outputs = []
    for sent_entry in sentence_entries:
        sent_logits = logits[sent2span[sent_entry['sentence']][0]: sent2span[sent_entry['sentence']][1]]
        sent_logits = torch.softmax(torch.tensor(sent_logits), axis=-1).numpy()
        sent_ref_logits = ref_logits[sent2span[sent_entry['sentence']][0]: sent2span[sent_entry['sentence']][1]]
        sent_ref_logits = torch.softmax(torch.tensor(sent_ref_logits), axis=-1).numpy()
        sent_tokens = tokens[sent2span[sent_entry['sentence']][0]: sent2span[sent_entry['sentence']][1]]
        sent_entry['sent_tokens'] = sent_tokens
        
        
        # feature 1: max entropy across all tokens
        entropy_with_ctx = [Categorical(probs = torch.tensor(sent_logits[i])).entropy().item()
                            for i in range(sent_logits.shape[0])]
        try:
            sent_entry['max_entropy'] = np.max(entropy_with_ctx).item()
            if np.isnan(sent_entry['max_entropy']):
                sent_entry['max_entropy'] = 0
        except:
            sent_entry['max_entropy'] = 0
         
        # feature 2: averaged entropy over the sequence
        sent_entry['mean_entropy'] = np.mean(entropy_with_ctx).item()
        if np.isnan(sent_entry['mean_entropy']):
            sent_entry['mean_entropy'] = 0
            
            
        # feature 3, 4: minimum/mean max_prob across all tokens (active rag)
        token_probs = [sent_logits[i, y_i].item() for i, y_i in enumerate(sent_tokens)]
        if len(token_probs) == 0:
            print(sent_entry['id'], sent_entry['sentence_id'], sent_entry['sentence'])
            print(sent2span[sent_entry['sentence']], sent_tokens)
            print('Warning: localization failed for a sentence', flush=True)
            sent_entry['min_prob'] = 0
            sent_entry['mean_prob'] = 0
        else:
            sent_entry['min_prob'] = np.array(token_probs).min().item()            
            sent_entry['mean_prob'] = np.mean(token_probs).item()            

        
        # feature 5: mean KL
        # sent_ref_logits is the logits of the model without retrieval
        kl_divs = [kl_div(torch.tensor(sent_ref_logits[i]).log(), torch.tensor(sent_logits[i]), size_average=False).item()
                    for i in range(sent_logits.shape[0])]
        sent_entry['mean_contrastive_kl'] = np.mean(kl_divs).item()
        if np.isnan(sent_entry['mean_contrastive_kl']):
            sent_entry['mean_contrastive_kl'] = 0
            
            
        # feature 6: count of large KL
        assert threshold is not None
        large_kl_count = 0
        for cur_kl in kl_divs:
            if cur_kl > threshold:
                large_kl_count += 1
        sent_entry['large_kl_count_threshold{}'.format(threshold)] = large_kl_count
        
        outputs.append(sent_entry)
        
    return outputs


def gather_sample(train_activations_filtered, layer, n_training_sample=1500):
    # further subsample training samples
    if len(train_activations_filtered) <= n_training_sample:
        print('Warning: max number of training sample is', len(train_activations_filtered))
    else:
        print('Further down-sampling training example to', n_training_sample)
        train_activations_filtered = random.sample(train_activations_filtered, n_training_sample)

    train_activations_filtered = [x['layer_activations'][layer] for x in train_activations_filtered]
    train_activations_filtered = np.array(train_activations_filtered).astype('float32')

    return train_activations_filtered


def get_activation_features(test_activations, train_activations_filtered):

    # bootstrap sampling the supporting instances
    layers = [15, 16, 17]
    n_bootstrap = 5
    training_activations_final = {}
    for layer in layers:
        training_activations_final[layer] = [gather_sample(train_activations_filtered, layer=layer) for _ in range(n_bootstrap)]
    
    outputs = []    
    for sent_entry in tqdm(test_activations):
        for layer in layers:
            lid_final_preds = []
            for t in training_activations_final[layer]:
                lids = compute_lid(sent_entry['layer_activations'][layer].reshape(1, -1),
                                    t, sample_size=-1, k_list=[t.shape[0]-1], metric='l2', block=50000)
                lid_final_preds.append(lids[0])
            sent_entry['lid_layer_{}'.format(layer)] = np.mean(lid_final_preds).item()
        outputs.append(sent_entry)
    return outputs


if __name__ == '__main__':
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    sentence_data = [json.loads(line) for line in open(args.sentence_file).readlines()]
    prompts = set([x['prompt'].strip() for x in sentence_data])
    doc2sentences = {}
    for sentence_entry in sentence_data:
        if sentence_entry['id'] not in doc2sentences:
            doc2sentences[sentence_entry['id']] = []
        doc2sentences[sentence_entry['id']].append(sentence_entry)
    for k in doc2sentences:
        doc2sentences[k].sort(key=lambda x: x['sentence_id'])
    
    logit_data = np.load(args.logit_file, allow_pickle=True)
    prompt2logit = {x['prompt'].replace('[INST]', '').replace('[/INST]', '').strip(): x
                    for x in logit_data}    
    prompt2logit = {k: v for k, v in prompt2logit.items() if k.strip() in prompts}
    assert len(prompts) == len(prompt2logit)
    
    ref_logit_data = np.load(args.ref_logit_file, allow_pickle=True)
    output2reflogit = {x['output_text']: x for x in ref_logit_data}    
    
    test_activations = np.load(args.test_activation_file, allow_pickle=True)
    assert len(sentence_data) == len(test_activations)
    
    train_activations_all = np.load(args.train_activation_file, allow_pickle=True)
    try:
        train_activations_filtered = [x for x in train_activations_all if (not x['is_conflict']) and (not x['is_baseless'])]
    except:
        train_activations_filtered = [x for x in train_activations_all if not x['is_baseless']]
    print(len(train_activations_all), 'pos+neg instances ->', len(train_activations_filtered), 'activation anchor samples')

    alignscore_preds = None
    if args.alignscore_preds_file:
        alignscore_preds = [json.loads(line) for line in open(args.alignscore_preds_file).readlines()]
        assert len(alignscore_preds) == len(sentence_data)
        
    all_outputs = []
    for docid in tqdm(doc2sentences):
        sent_entries = doc2sentences[docid]
        sents = split_sentences(prompt2logit[sent_entries[0]['prompt']]['output_text'])
        
        # logit features
        logits = prompt2logit[sent_entries[0]['prompt']]['logits']
        ref_logits = output2reflogit[prompt2logit[sent_entries[0]['prompt']]['output_text']]['logits']
        tokens = tokenizer.encode(prompt2logit[sent_entries[0]['prompt']]['output_text'],
                                  add_special_tokens=False)

        try:
            assert all([x['sentence'].strip() in [y.strip() for y in sents] for x in sent_entries]) and all([x.strip() in [y['sentence'].strip() for y in sent_entries] for x in sents])
        except:
            print('Sentence mismatch found')

        assert len(tokens) == logits.shape[0] - 1
        
        sent_entries_with_features = get_logit_features(sent_entries, tokens, logits, ref_logits, args.kl_threshold)
        
        all_outputs += sent_entries_with_features
        
    for i in range(len(all_outputs)):
        assert all_outputs[i]['sentence'] == test_activations[i]['sentence']
        all_outputs[i]['layer_activations'] = test_activations[i]['layer_activations']
    all_outputs = get_activation_features(all_outputs, train_activations_filtered)
    for entry in all_outputs:
        entry.pop('layer_activations')


    for i in range(len(all_outputs)):
        assert all_outputs[i]['sentence'] == alignscore_preds[i]['sentence']
        all_outputs[i]['alignscore'] = -1 * alignscore_preds[i]['pred_val']
    
    np.save(args.out_file, all_outputs)
    print(args.out_file)
    
