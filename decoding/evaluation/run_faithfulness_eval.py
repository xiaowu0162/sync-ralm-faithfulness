import os
import argparse
import numpy as np
import json
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pickle
import time
import string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--support_file', type=str, required=True)
    parser.add_argument('--log_file', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--decomp_backend', type=str, required=True, choices=['factscore', 'densexretrieval'])
    parser.add_argument('--openai_key', type=str, required=True)
    return parser.parse_args()   


def load_cache(cache_file):
    if os.path.exists(cache_file):
        while True:
            try:
                with open(cache_file, "rb") as f:
                    cache = pickle.load(f)
                break
            except Exception:
                print ("Pickle Error: Retry in 5sec...")
                time.sleep(5)        
    else:
        cache = {}
    return cache


def save_cache(cache_dict, cache_file):
    # if there were other processes running in parallel, cache might have been updated
    for k, v in load_cache(cache_file).items():
        cache_dict[k] = v
    with open(cache_file, "wb") as f:
        pickle.dump(cache_dict, f)
        

# Propositionization
def run_text2prop_densexretrieval(entry, prop_tokenizer, prop_model, cache_dict):
    input_text = f"Title: {entry['title']}. Section: {entry['section']}. Content: {entry['content']}"
    if input_text in cache_dict:
        return cache_dict[input_text]
    input_ids = prop_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = prop_model.generate(input_ids.to(prop_model.device), max_new_tokens=512).cpu()
    
    output_text = prop_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        prop_list = json.loads(output_text)
    except:
        prop_list = []
        print("[ERROR] Failed to parse output text as JSON.")
    cache_dict[input_text] = prop_list
    return prop_list


def main(args):
    print(args)
    
    # read data
    outputs = [json.loads(line) for line in open(args.pred_file).readlines()]
    if 'id' not in outputs[0]:
        for i in range(len(outputs)):
            outputs[i]['id'] = i
            if args.task in ['bio', 'famous-100', 'famous-100-anti', 'famous-100-anti-v2']:
                outputs[i]['id'] = outputs[i]['input']
    supports = [json.loads(line) for line in open(args.support_file).readlines()]
    if args.task in ['QA', 'Summary', 'Data2txt']:
        id2support = {x['source_id']: x for x in supports}
        assert all([x['source_id'] in id2support for x in outputs])
    elif args.task in ['bio', 'famous-100', 'famous-100-anti', 'famous-100-anti-v2']:
        id2support = {x['input']: x for x in supports}
        assert all([x['id'] in id2support for x in outputs])
        
        
    # step 1: fact decomposition
    if args.decomp_backend == 'factscore':
        af_generator = AtomicFactGenerator(key_path=args.openai_key, 
                                           demon_dir=os.path.join(args.cache_dir, "demos"), 
                                           gpt3_cache_file=os.path.join(args.cache_dir, "InstructGPT.pkl"))
    else:
        decomp_cache_file = os.path.join(args.cache_dir, "densexretrieval.pkl")
        decomp_cache = load_cache(decomp_cache_file)
        prop_model_name = "chentong00/propositionizer-wiki-flan-t5-large"
        prop_tokenizer = AutoTokenizer.from_pretrained(prop_model_name)
        prop_model = AutoModelForSeq2SeqLM.from_pretrained(prop_model_name).cuda()
    
    for i in tqdm(range(len(outputs)), desc='Atom Fact Generation'):
        if args.task in ['QA', 'Summary', 'Data2txt']:
            gen = outputs[i]['response']
        else:
            gen = outputs[i]['output']
        if gen.strip() == '':
            curr_afs = []
        else:
            if args.decomp_backend == 'factscore':
                curr_afs, _ = af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
            else:
                temp_entry = {'title': '', 'section': '', 'content': gen}
                curr_afs = run_text2prop_densexretrieval(temp_entry, prop_tokenizer, prop_model, decomp_cache)
        outputs[i]['atomic_facts'] = curr_afs
        print(gen)
        print(curr_afs)
        print('\n================================\n')
        
    if args.decomp_backend == 'densexretrieval':
        save_cache(decomp_cache, decomp_cache_file)
        
        
    # step 2: knowledge formatting
    atomfact2passage = {}
    for i in tqdm(range(len(outputs)), desc='Formatting'):
        if args.task in ['QA', 'Summary', 'Data2txt']:
            support_entry = id2support[outputs[i]['source_id']]
            if args.task == 'QA':
                outputs[i]['context_text'] = support_entry['source_info']['passages']
            elif args.task == 'Summary':
                outputs[i]['context_text'] = support_entry['source_info']
            elif args.task == 'Data2txt':
                outputs[i]['context_text'] = str(support_entry['source_info'])
            else:
                raise NotImplementedError
        else:
            support_entry = id2support[outputs[i]['id']]
            context = ""
            for psg_idx, psg in enumerate(support_entry['ctxs']):
                context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
            outputs[i]['context_text'] = context
            
        for atom_fact in outputs[i]['atomic_facts']:
            atomfact2passage[atom_fact] = outputs[i]['context_text']
        
    # step 3: scoring based on llama and npm
    atom_fact_per_resp = []
    accuracy = []
    eval_lm_llama = CLM("inst-llama-7B", 
                        model_dir=os.path.join(args.cache_dir, "inst-llama-7B"),
                        cache_file=os.path.join(args.cache_dir, "inst-llama-7B.pkl"))

    class PseudoRetrieverforNPM:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/npm-single')
        def get_passages(self, topic, question, k):
            passage_text = atomfact2passage[question]
            passage_text = self.tokenizer.decode(self.tokenizer(passage_text, truncation=True, max_length=400, add_special_tokens=False)['input_ids'])
            return [{'text': passage_text}]
    eval_lm_npm = NPM(PseudoRetrieverforNPM(),
                      "npm-single",
                      cache_file=os.path.join(args.cache_dir, f"npm-enwiki-20230401.pkl"))
    
    for i in tqdm(range(len(outputs)), desc='Scoring with llama'):
        for atom_fact in outputs[i]['atomic_facts']:
            # clm
            definition = "Answer the question based on the given context.\n\n"
            context = outputs[i]['context_text']
            definition += context.strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom_fact.strip())
            output = eval_lm_llama.generate(prompt)
            generated_answer = output[0].lower()
            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index("true") > generated_answer.index("false")
            else:
                is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            # npm
            if is_supported:
                npprob = eval_lm_npm.get_probabilty('', atom_fact)
                is_supported = npprob > 0.3
                print([is_supported, atom_fact, npprob])    
            else:
                print([is_supported, atom_fact])
            accuracy.append(1 if is_supported else 0)
            
        if outputs[i]['atomic_facts']:
            atom_fact_per_resp.append(len(outputs[i]['atomic_facts']))
        print('\n================================\n')
    
    print('Accuracy', np.mean(accuracy))
    print('Atom fact per response', np.mean(atom_fact_per_resp))
    with open(args.log_file, 'w') as f:
        print('Accuracy', np.mean(accuracy), file=f)
        print('Atom fact per response', np.mean(atom_fact_per_resp), file=f)
        print('# of responses:', len(atom_fact_per_resp), file=f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
