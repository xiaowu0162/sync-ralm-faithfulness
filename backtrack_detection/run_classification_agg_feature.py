import json
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--train_task', type=str, default=None)
    parser.add_argument('--checked_model', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    return parser.parse_args()   


def main(args):
    # modeling parameters
    # detection_model = LogisticRegression(max_iter=1000)
    # detection_model = XGBClassifier(objective="binary:logistic")
    detection_model = MLPClassifier(hidden_layer_sizes=100, max_iter=200)#, early_stopping=True)
    # detection_model = MLPClassifier(hidden_layer_sizes=100, max_iter=200, learning_rate_init=0.0002 if args.task in ['QA', 'Summary', 'Data2txt'] else 0.001)#, early_stopping=True)

    train_task = args.task if args.train_task is None else args.train_task 
    if train_task == 'bio':
        train_data = f'{args.root_dir}/rag_analysis_logs/syncheck_aggregated_features/famous-100/{args.checked_model}_test.npy'
    elif train_task in ['famous-100', 'famous-100-anti', 'famous-100-anti-v2']:
        train_data = f'{args.root_dir}/rag_analysis_logs/syncheck_aggregated_features/bio/{args.checked_model}_test.npy'
    else:
        train_data = f'{args.root_dir}/rag_analysis_logs/syncheck_aggregated_features/{train_task}/{args.checked_model}_train.npy'

    test_data = f'{args.root_dir}/rag_analysis_logs/syncheck_aggregated_features/{args.task}/{args.checked_model}_test.npy'
    print('Train:', train_data)
    print('Test:', test_data)
    train_data = np.load(train_data, allow_pickle=True)
    test_data = np.load(test_data, allow_pickle=True)

    if args.task in ['famous-100', 'famous-100-anti', 'famous-100-anti-v2', 'bio', 'QA']:
        use_features = [
            'max_entropy', 'mean_entropy',
            'lid_layer_15', 'lid_layer_16', 'lid_layer_17',
            'min_prob', 'mean_prob',
            'mean_contrastive_kl', 'large_kl_count_threshold3.0',
            'alignscore'
        ]
    else:
        use_features = [
            'max_entropy', 'mean_entropy',
            # 'lid_layer_15', 'lid_layer_16', 'lid_layer_17',
            'min_prob', 'mean_prob',
            'mean_contrastive_kl', 'large_kl_count_threshold3.0',
            'alignscore'
        ]

    train_X, train_y = [], []
    for sent_entry in train_data:
        if sent_entry['is_baseless'] or ('is_conflict' in sent_entry and sent_entry['is_conflict']):
            is_hallucination = True
        else:
            is_hallucination = False
        train_X.append([sent_entry[feat] for feat in use_features])
        train_y.append(1 if is_hallucination else 0)

    test_X, test_y = [], []
    for sent_entry in test_data:
        if sent_entry['is_baseless'] or ('is_conflict' in sent_entry and sent_entry['is_conflict']):
            is_hallucination = True
        else:
            is_hallucination = False
        test_X.append([sent_entry[feat] for feat in use_features])
        test_y.append(1 if is_hallucination else 0)

        
    print('Train:', len(train_X))
    print('Test:', len(test_X))

    detection_model.fit(train_X, train_y)

    train_preds = detection_model.predict(train_X)
    train_preds_prob = detection_model.predict_proba(train_X)[:, 1]
    print('train:', {
        "Accuracy": accuracy_score(train_y, train_preds),
        "Precision": precision_score(train_y, train_preds),
        "Recall": recall_score(train_y, train_preds),
        "F1": f1_score(train_y, train_preds),
        "AUROC": roc_auc_score(train_y, train_preds_prob)
    })

    test_preds = detection_model.predict(test_X)
    test_preds_prob = detection_model.predict_proba(test_X)[:, 1]
    print('test:', {
        "Accuracy": accuracy_score(test_y, test_preds),
        "Precision": precision_score(test_y, test_preds),
        "Recall": recall_score(test_y, test_preds),
        "F1": f1_score(test_y, test_preds),
        "AUROC": roc_auc_score(test_y, test_preds_prob)
    })
    
    checkpoint_file = f'syncheck_{args.checked_model}_{args.train_task}.pkl'
    pickle.dump(detection_model, open(checkpoint_file, 'wb')) 


if __name__ == '__main__':
    args = parse_args()
    main(args)
