import json
import sys
from sklearn.metrics import roc_auc_score


in_file = sys.argv[1]


first_sentence_only = False
if first_sentence_only:
    print('Note, only evaluating on the first sentence with hallucination label.')


traces = [json.loads(line) for line in open(in_file).readlines()]
if first_sentence_only:
    traces = [x for x in traces if 'n_preceding_baseless' in x and x['n_preceding_baseless'] == 0]
    traces = [x for x in traces if 'n_preceding_conflict' in x and x['n_preceding_conflict'] == 0]
pred_vals = [x['pred_val'] for x in traces]


# is_conflict only
try:
    labels = [1 if entry['is_conflict'] else 0 for entry in traces]
    print('Label = 1 only when is_conflict = True')
    print('AUROC:', roc_auc_score(labels, pred_vals))
except:
    print('Skipping is_conflict as it is not in data')

# is_baseless only
try:
    print('Label = 1 only when is_baseless = True')
    labels = [1 if entry['is_baseless'] else 0 for entry in traces]
    print('AUROC:', roc_auc_score(labels, pred_vals))
except:
    print('Skipping is_baseless as it is not in data')

# is_conflict or is_baseless
try:
    labels = [1 if entry['is_baseless'] or entry['is_conflict']  else 0 for entry in traces]
    print('Label = 1 when is_conflict or is_baseless')
    print('AUROC:', roc_auc_score(labels, pred_vals))
except:
    pass
