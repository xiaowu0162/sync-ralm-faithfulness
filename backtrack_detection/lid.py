import os
import random
import torch
import faiss
import skdim
import ast
import numpy as np
import pandas as pd
import torch.nn as nn
import plotly.express as px
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, normalize
from torch.utils.data import Dataset, DataLoader
from skdim.id import TwoNN, ESS, MOM, KNN, DANCo, MiND_ML, MLE, lPCA
from sklearn import metrics


def compute_lid(y, sampled_feats, sample_size=-1, k_list=[200], metric='l2', block=50000):
    if metric == 'cos':
        cpu_index = faiss.IndexFlatIP(sampled_feats.shape[1])
        y = normalize(y)
        sampled_feats = normalize(sampled_feats)
    if metric == 'l2':
        cpu_index = faiss.IndexFlatL2(sampled_feats.shape[1])

    # print('cpu_index')
    # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    cpu_index.add(np.ascontiguousarray(sampled_feats))

    avg_lids = []

    for k in k_list:
        i = 0
        D = []
        while i < y.shape[0]:
           tmp = y[i:min(i + block, y.shape[0])]
           i += block
           b, nid = cpu_index.search(tmp, k)
           b = np.sqrt(b)
           D.append(b)

        D = np.vstack(D)
        # print("query finish")
        if metric == 'cos':
          D = 1 - D  # cosine dist = 1 - cosine
          D[D <= 0] = 1e-8
        rk = np.max(D, axis=1)
        rk[rk == 0] = 1e-8
        lids = D / rk[:, None]
        lids = -1 / np.mean(np.log(lids), axis=1)
        lids[np.isinf(lids)] = y.shape[1]  # if inf, set as space dimension
        lids = lids[~np.isnan(lids)]  # filter nan
        avg_lids.append(lids.tolist())
        # print('filter nan/inf shape', lids.shape)
        # print('k', k - 1, 'lid_mean', np.mean(lids), 'lid_std', np.std(lids))
    avg_lids = np.array(avg_lids).mean(axis=0)
    return avg_lids


def roc(corrects, scores):
    auroc = metrics.roc_auc_score(corrects, scores)
    return auroc

def pvalue_score(scores_null, scores_obs, log_transform=False, bootstrap=True, n_bootstrap=5):
    """
    Calculate the empirical p-values of the observed scores `scores_obs` with respect to the scores from the
    null distribution `scores_null`.
    """
    eps = 1e-16
    n_samp = scores_null.shape[0]
    n_obs = scores_obs.shape[0]
    p = np.zeros(n_obs)
    for i in range(n_obs):
        for j in range(n_samp):
            if scores_null[j] >= scores_obs[i]:
                p[i] += 1.

        p[i] = p[i] / n_samp

    if bootstrap:
        ind_null_repl = np.random.choice(np.arange(n_samp), size=(n_bootstrap, n_samp), replace=True)
        p_sum = p
        for b in range(n_bootstrap):
            print(b)
            p_curr = np.zeros(n_obs)
            for i in range(n_obs):
                for j in ind_null_repl[b, :]:
                    if scores_null[j] >= scores_obs[i]:
                        p_curr[i] += 1.

                p_curr[i] = p_curr[i] / n_samp

            p_sum += p_curr

        # Average p-value from the bootstrap replications
        p = p_sum / (n_bootstrap + 1.)

    p[p < eps] = eps
    if log_transform:
        return -np.log(p)
    else:
        return p

def other_ids():
    IDE = MLE(K=100, unbiased=True, neighborhood_based=False)
    IDE = KNN(k=100)
    # IDE = TwoNN(discard_fraction=0.1)
    # IDE = ESS(ver='a')
    # IDE = lPCA(ver='FO')
    true_id = IDE.fit_transform(X=true_pds)
    false_id = IDE.fit_transform(X=wrong_pds)

def visualization():
    pca = PCA(n_components=2)
    components = pca.fit_transform(train_pd)
    # total_var = pca.explained_variance_ratio_.sum() * 100
    # print(total_var)
    fig = px.scatter(components, x=0, y=1, color=train_labels)
    fig.write_image(f'{i}th_pcaplot.png')
