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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--criterion', type=str, required=True)
    parser.add_argument('--kl_threshold', type=float, default=None)
    parser.add_argument('--sentence_file', type=str, required=True)
    parser.add_argument('--logit_file', type=str, required=True)
    parser.add_argument('--ref_logit_file', type=str, required=True)
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


def get_preds_logit_features(sentence_entries, tokens, logits, ref_logits, criterion, threshold=None):
    sent2span = get_sent_to_token_range([x['sentence'] for x in sentence_entries], tokens)
    outputs = []
    for sent_entry in sentence_entries:
        sent_logits = logits[sent2span[sent_entry['sentence']][0]: sent2span[sent_entry['sentence']][1]]
        sent_logits = torch.softmax(torch.tensor(sent_logits), axis=-1).numpy()
        sent_ref_logits = ref_logits[sent2span[sent_entry['sentence']][0]: sent2span[sent_entry['sentence']][1]]
        sent_ref_logits = torch.softmax(torch.tensor(sent_ref_logits), axis=-1).numpy()
        sent_tokens = tokens[sent2span[sent_entry['sentence']][0]: sent2span[sent_entry['sentence']][1]]
        sent_entry['sent_tokens'] = sent_tokens
        
        if criterion == 'active_rag':
            # Active RAG's strategy: the min likelihood of the per-token maxprob in the sentence.
            # We subtract 1 by it to make it the larger the more likely to hallucinate
            cared_probs = [sent_logits[i, y_i].item() for i, y_i in enumerate(sent_tokens)]
            if len(cared_probs) == 0:
                print(sent_entry['id'], sent_entry['sentence_id'], sent_entry['sentence'])
                print(sent2span[sent_entry['sentence']], sent_tokens)
                print('Warning: localization failed for a sentence', flush=True)
                sent_entry['pred_val'] = 0
            else:
                sent_entry['pred_val'] = 1 - np.array(cared_probs).min().item()            
            sent_entry['cared_probs'] = cared_probs
            
        elif criterion == 'mean_entropy_with_ctx':
            # return the averaged entropy over the sequence
            entropy_with_ctx = [Categorical(probs = torch.tensor(sent_logits[i])).entropy().item()
                                for i in range(sent_logits.shape[0])]
            sent_entry['pred_val'] = np.mean(entropy_with_ctx).item()
            if np.isnan(sent_entry['pred_val']):
                sent_entry['pred_val'] = 0
            sent_entry['entropy_with_ctx'] = entropy_with_ctx
        
        elif criterion == 'max_entropy_with_ctx':
            # return the max entropy over the sequence
            entropy_with_ctx = [Categorical(probs = torch.tensor(sent_logits[i])).entropy().item()
                                for i in range(sent_logits.shape[0])]
            try:
                sent_entry['pred_val'] = np.max(entropy_with_ctx).item()
            except:
                sent_entry['pred_val'] = 0
            if np.isnan(sent_entry['pred_val']):
                sent_entry['pred_val'] = 0
            sent_entry['entropy_with_ctx'] = entropy_with_ctx

        elif criterion == 'avg_neg_contrastive_kl':
            # average token-wise KL, negated
            # sent_ref_logits is the logits of the model without retrieval
            kl_divs = [kl_div(torch.tensor(sent_ref_logits[i]).log(), torch.tensor(sent_logits[i]), size_average=False).item()
                       for i in range(sent_logits.shape[0])]
            sent_entry['pred_val'] = -1 * np.mean(kl_divs).item()  # negative because we want large values for potential hallucination
            if np.isnan(sent_entry['pred_val']):
                sent_entry['pred_val'] = 0
            sent_entry['kl_divs'] = kl_divs
        
        elif criterion == 'neg_large_kl_count':
            # number of tokens that have KL greater than threshold
            # sent_ref_logits is the logits of the model without retrieval
            assert threshold is not None
            kl_divs = [kl_div(torch.tensor(sent_ref_logits[i]).log(), torch.tensor(sent_logits[i]), size_average=False).item()
                       for i in range(sent_logits.shape[0])]
            large_kl_count = 0
            for cur_kl in kl_divs:
                if cur_kl > threshold:
                    large_kl_count += 1
            sent_entry['pred_val'] = -1 * large_kl_count  # negative because we want large values for potential hallucination
            sent_entry['kl_divs'] = kl_divs
        
        else:
            raise NotImplementedError
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
    
    out_f = open(args.out_file, 'w')

    all_outputs = []
    for docid in tqdm(doc2sentences):
        sent_entries = doc2sentences[docid]
        sents = split_sentences(prompt2logit[sent_entries[0]['prompt']]['output_text'])
        logits = prompt2logit[sent_entries[0]['prompt']]['logits']
        ref_logits = output2reflogit[prompt2logit[sent_entries[0]['prompt']]['output_text']]['logits']
        tokens = tokenizer.encode(prompt2logit[sent_entries[0]['prompt']]['output_text'],
                                  add_special_tokens=False)
        
        assert all([x['sentence'].strip() in [y.strip() for y in sents] for x in sent_entries]) and all([x.strip() in [y['sentence'].strip() for y in sent_entries] for x in sents])
        assert len(tokens) == logits.shape[0] - 1
        
        #prompt2logit[sent_entries[0]['prompt']]['output_text'],
        all_outputs += get_preds_logit_features(sent_entries, tokens, logits, ref_logits, args.criterion, args.kl_threshold)

    for out_entry in all_outputs:
        print(json.dumps(out_entry), file=out_f)
        
        
