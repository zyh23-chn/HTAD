import torch
from torch_geometric.data import HeteroData
import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def extract_data(dataset, name, eidict=None) -> HeteroData:
    '''identify name to tackle multi-label'''
    data = HeteroData()
    for i, node_type in enumerate(dataset.node_types):
        data[node_type].x = dataset.features[i]
    for i, link_type in enumerate(dataset.link_types):
        if eidict is None:
            data[link_type].edge_index = dataset.edge_index_dict[i]
        elif i in eidict:
            data[link_type].edge_index = eidict[i]

    type_t = dataset.type_t
    if name == 'IMDB':
        data[type_t].y = torch.from_numpy(dataset.labels).float()  # multi-label target
    else:
        data[type_t].y = torch.from_numpy(dataset.labels).long()  # only masked positions have true label
    return data


def div_bucket_discrete(deg_t, indices_test, n_buckets):
    n_test_t = indices_test.shape[0]
    n_nodes_t = deg_t.shape[0]
    deg_test = deg_t[indices_test]
    x_l, x_r = min(deg_test), max(deg_test)
    diff = x_r - x_l + 1
    n_buckets_deg = min(n_buckets, diff)
    interval_len = diff // n_buckets_deg
    rem = diff % n_buckets_deg
    bucket_deg = np.zeros(n_nodes_t)
    id_deg = np.stack((deg_test, indices_test), axis=-1)
    id_deg = id_deg[np.lexsort(id_deg.T[::-1])]
    y_l, j = x_l, 0
    for i in range(n_buckets_deg):
        y_r = y_l + interval_len + (1 if i < rem else 0)
        while j < n_test_t and id_deg[j][0] < y_r:
            bucket_deg[round(id_deg[j][1])] = i
            j += 1
        y_l = y_r
    assert y_l == x_r + 1 and j == n_test_t
    return bucket_deg[indices_test].astype(int)


def div_bucket_conti(hlid, indices_test, n_buckets):
    n_test_t = indices_test.shape[0]
    n_nodes = hlid.shape[0]
    n_buckets_hlid = min(n_buckets, n_test_t)
    interval_len = n_test_t // n_buckets_hlid
    rem = n_test_t % n_buckets_hlid
    id_hlid = np.stack((hlid[indices_test], indices_test), axis=-1)  # convert indices into float here!
    id_hlid = id_hlid[np.lexsort(id_hlid.T[::-1])]
    bucket_hlid = np.zeros(n_nodes)
    y_l = 0
    for i in range(n_buckets_hlid):
        y_r = y_l + interval_len + (1 if i < rem else 0)
        for j in range(y_l, y_r):
            bucket_hlid[round(id_hlid[j][1])] = i
        y_l = y_r
    assert y_l == n_test_t
    return bucket_hlid[indices_test].astype(int)


def sim_arr(X):
    mat = cosine_similarity(X, X)
    rows, cols = np.triu_indices(n=mat.shape[0], k=1)  # k=1 表示从主对角线往上偏移1（不包含对角线）
    return mat[rows, cols]


def multilabel_dist(y_true, y_pred):
    '''Return:
    L1, L2 distance'''
    arr1 = sim_arr(y_true)
    arr2 = sim_arr(y_pred)
    n_pairs = arr1.shape[0]
    return np.sum(np.abs(arr1 - arr2)) / n_pairs, np.sqrt(np.sum((arr1 - arr2) ** 2) / n_pairs)
