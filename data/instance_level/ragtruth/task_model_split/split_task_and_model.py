import os
import json


in_dir = os.path.dirname(os.getcwd()) + '/raw'

all_source_info = [json.loads(line) for line in open(in_dir + '/source_info.jsonl').readlines()]
all_response = [json.loads(line) for line in open(in_dir + '/response.jsonl').readlines()]


for task in ['QA', 'Data2txt', 'Summary']:
    out_dir = task
    os.makedirs(out_dir + '/output_only', exist_ok=True)
    os.makedirs(out_dir + '/output_labeled', exist_ok=True)
    
    task_sources = [x for x in all_source_info if x['task_type'] == task]
    task_id_to_source = {}
    with open(f'{task}/source_info.jsonl', 'w') as out_f:
        for entry in task_sources:
            print(json.dumps(entry), file=out_f)
            task_id_to_source[entry['source_id']] = entry
    
    task_ids = set([x['source_id'] for x in task_sources])
    task_responses = [x for x in all_response if x['source_id'] in task_ids]
    for model in ['gpt-3.5-turbo-0613', 'llama-2-13b-chat', 'mistral-7B-instruct', 'llama-2-70b-chat', 'llama-2-7b-chat', 'gpt-4-0613']:
        out_full_file = f'{out_dir}/output_labeled/{model}.jsonl'
        out_output_only_file = f'{out_dir}/output_only/{model}.jsonl'
        with open(out_full_file, 'w') as out_full_f, open(out_output_only_file, 'w') as out_f:
            for entry in task_responses:
                if entry['model'] != model:
                    continue
                print(json.dumps(entry), file=out_full_f)
                print(json.dumps({
                    'input': task_id_to_source[entry['source_id']]['prompt'],
                    'response': entry['response']
                }), file=out_f)
        #print(len(task_responses))
        #print(list(set([x['model'] for x in task_responses])))
