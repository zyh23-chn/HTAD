import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import dropout_adj


def get_contrastive_g(dataset, hlid, p_inf, dens_scale):
    dl = dataset.dl
    links = dl.links
    nodes = dl.nodes
    n_node_types = len(nodes['count'])
    edge_index_dict = defaultdict(lambda: [[] for _ in range(2)])
    sft = nodes['shift']
    cnt = nodes['count']
    for i in links['count']:
        tp0, tp1 = links['meta'][i]
        adj = links['data'][i]
        # adj=links['data'][i].tocoo()
        # adj_indices=torch.tensor([adj.row,adj.col])
        # adj_vals=torch.tensor(adj.data)
        # adj_size=torch.Size(adj.shape)
        # adj=torch.sparse_coo_tensor(adj_indices,adj_vals,adj_size).to(device)
        for nd0 in range(cnt[tp0]):
            sft1, cnt1 = sft[tp1], cnt[tp1]
            adj1 = np.array(adj[nd0 + sft[tp0], sft1 : sft1 + cnt1].todense()).squeeze()
            rnd = np.random.rand(cnt1)
            pr0 = 1.0 - (1.0 - p_inf) * np.exp(-dens_scale * np.abs(hlid[nd0] - hlid[sft1 : sft1 + cnt1]))
            pr1 = 1.0 - np.exp(-dens_scale * np.abs(hlid[nd0] - hlid[sft1 : sft1 + cnt1]))
            pr = np.where(adj1 != 0, pr0, pr1)
            mask = rnd < pr
            edge_index_dict[i][0].extend(nd0 for _ in range(mask.sum().item()))
            edge_index_dict[i][1].extend(nd1 for nd1 in range(cnt1) if mask[nd1])

        '''slow, need to be optimized'''
        # row_i, col_i = links['data'][i].nonzero()
        # pos = set((r_i, c_i) for r_i, c_i in zip(row_i, col_i))
        # for nd0 in tqdm(range(sft[tp0], sft[tp0] + cnt[tp0])):
        #     for nd1 in range(sft[tp1], sft[tp1] + cnt[tp1]):
        #         pr = 0.0
        #         if (nd0, nd1) in pos:
        #             pr = 1.0 - (1.0 - p_inf) * math.exp(-dens_scale * abs(hlid[nd0] - hlid[nd1]))
        #         else:
        #             pr = 1.0 - math.exp(-dens_scale * abs(hlid[nd0] - hlid[nd1]))
        #         rnd = random.random()
        #         if rnd < pr:
        #             edge_index_dict[i][0].append(nd0)
        #             edge_index_dict[i][1].append(nd1)
    return {edge_type: torch.tensor(edge_index) for edge_type, edge_index in edge_index_dict.items()}


def inner_prod(h1, h2):
    '''
    h1:(n_features,)
    h2:(n_samples,n_features)
    '''
    return torch.matmul(h2, h1)


def cos_sim(h1, h2):
    prod = torch.matmul(h2, h1)
    a_norm = torch.norm(h1)
    b_norms = torch.norm(h2, dim=1)  # Shape (m,)
    sim = prod / (a_norm * b_norms)  # Shape (m,)
    return sim
