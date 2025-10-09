import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import numpy as np
import torch_geometric
import os.path as osp
from tqdm import tqdm
import math
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import hdbscan
import skfuzzy as fuzz
from copy import deepcopy

from new_geodata import IMDB, DBLP, ACM, Freebase
from model import get_contrastive_g, get_grace_g, get_gca_g, inner_prod, cos_sim, HAN, HGT, GAT, SAGE
from utils import extract_data, div_bucket_discrete, div_bucket_conti, multilabel_dist


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="IMDB")
    parser.add_argument('--model', type=str, default='HAN')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_pretrain", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--ext", type=str, default='hn', choices=['mp', 'hn'])
    parser.add_argument("--p_inf", type=float, default=0.5, help='lower bound of p_uv, (u,v) in E')
    parser.add_argument("--dens_scale", type=float, default=0.01, help='density scale parameter')
    parser.add_argument("--simf", type=str, default='cos', choices=['cos', 'prod'], help='similarity function')
    parser.add_argument("--temp", type=float, default=1.0, help='temperature parameter')
    parser.add_argument("--label_rate", type=float, default=0.05)
    parser.add_argument("--rel_reg", type=float, default=2000.0)
    parser.add_argument("--self_reg", type=float, default=3.0)
    parser.add_argument("--lambda_u", type=float, default=0.0, help='unsupervised loss regulation')
    parser.add_argument("--lambda_t", type=float, default=0.0, help='supervised loss regulation')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch_geometric.is_xpu_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    if args.dataset == "IMDB":
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../data/IMDB")
        dataset = IMDB(path, args)
    elif args.dataset == "DBLP":
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../data/DBLP")
        dataset = DBLP(path, args)
    elif args.dataset == "ACM":
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../data/ACM")
        dataset = ACM(path, args)
    elif args.dataset == "Freebase":
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../data/Freebase")
        dataset = Freebase(path, args)
    else:
        raise NotImplementedError
    n_classes = dataset.n_classes
    type_t = dataset.type_t
    data = extract_data(dataset, args.dataset)
    if args.model == 'HAN':
        model = HAN(in_channels={node_type: dataset.features[i].shape[1] for i, node_type in enumerate(dataset.node_types)}, out_channels=n_classes, metadata=data.metadata(), device=device)
    elif args.model == 'HGT':
        model = HGT(hidden_channels=64, out_channels=n_classes, metadata=data.metadata(), device=device)
    elif args.model == 'GAT':
        model = GAT(hidden_channels=64, out_channels=n_classes, metadata=data.metadata(), device=device)
    elif args.model == 'GCN':
        model = SAGE(hidden_channels=64, out_channels=n_classes, metadata=data.metadata(), device=device)
    else:
        raise NotImplementedError
    data, model = data.to(device), model.to(device)
    print('Constructing contrastive graph...')
    hlid = dataset.get_hlid(args.rel_reg or args.self_reg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    data_t = data[dataset.type_t]
    eidict_ctr = get_contrastive_g(dataset, hlid, args.p_inf, args.dens_scale)
    data_ctr = extract_data(dataset, args.dataset, eidict_ctr).to(device)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)
        out_ctr = model(data_ctr.x_dict, data_ctr.edge_index_dict)

    temp = args.temp
    nd_cnt = dataset.dl.nodes['count']
    n_tot = dataset.dl.nodes['total']

    def train(pret: bool) -> float:
        model.train()
        optimizer.zero_grad()
        train_i = dataset.train_i
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[dataset.type_t][train_i], data_t.y[train_i])

        if pret:
            if args.simf == 'cos':
                sim_f = cos_sim
            elif args.simf == 'prod':
                sim_f = inner_prod
            else:
                raise NotImplementedError
            '''calculate contrastive loss here
            eidict_ctr is given'''
            loss_u, loss_t = 0.0, 0.0
            cnt_u = 0
            n_trains = len(train_i)
            rec = defaultdict(lambda: [False] * n_trains)  # record indices of the same label
            for i, index in enumerate(train_i):
                yi = data_t.y[index]
                rec[yi.item() if yi.ndim == 0 else tuple(yi.tolist())][i] = True

            def calc_loss(out, out_ctr, type_ids, hn=False):
                nonlocal loss_u, loss_t,cnt_u
                for tp_id in type_ids:
                    node_type = dataset.node_types[tp_id]
                    fea = out[node_type]
                    fea_ctr = out_ctr[node_type]
                    n_nodes = fea.size(0)
                    if node_type == type_t:
                        for index in train_i:
                            yi = data_t.y[index]
                            mask = torch.tensor(rec[yi.item() if yi.ndim == 0 else tuple(yi.tolist())]).to(device)
                            sim_v = torch.exp(sim_f(fea[index], fea_ctr[train_i]) / temp)
                            x_pos = torch.where(mask, sim_v, 0.0).sum()
                            x_neg = sim_v.sum() - x_pos
                            loss_t -= torch.log(x_pos / x_neg)
                    for index in range(n_nodes):
                        sim_v = torch.exp(sim_f(fea[index], fea_ctr) / temp)
                        x_pos = sim_v[index]
                        x_neg = sim_v.sum() - x_pos
                        if hn:
                            betas = torch.from_numpy(np.random.beta(1, 1, n_nodes)).float().to(device)
                            fea_hn = betas.unsqueeze(1) @ fea[index].unsqueeze(0) + (1 - betas.unsqueeze(1)) * fea_ctr
                            sim_v = torch.exp(sim_f(fea[index], fea_hn) / temp)
                            x_neg += sim_v.sum()
                        loss_u -= torch.log(x_pos / x_neg)
                    cnt_u += n_nodes

            print(f'Cardinality of the original edge set: {sum([ei.shape[1] for ei in data.edge_index_dict.values()])}')
            print(f'Cardinality of edge set: {sum([ei.shape[1] for ei in eidict_ctr.values()])}')
            if args.ext == 'mp' or args.ext == 'hn':
                meta = dataset.dl.links['meta']
                mps = dataset.mps
                for mp in mps:
                    tuples = [meta[et] for et in mp]
                    node_type_ids = list({x for tpl in tuples for x in tpl})
                    eidict_mp = {et: eidict_ctr[et] for et in mp}
                    data_mp = extract_data(dataset, args.dataset, eidict_mp).to(device)
                    out_ctr = model(data_mp.x_dict, data_mp.edge_index_dict)  # meta-path only
                    calc_loss(out, out_ctr, node_type_ids, args.ext == 'hn')
            else:
                data_ctr = extract_data(dataset, args.dataset, eidict_ctr).to(device)
                out_ctr = model(data_ctr.x_dict, data_ctr.edge_index_dict)
                calc_loss(out, out_ctr, list(nd_cnt.keys()))
            loss_u /= cnt_u  # normalization
            loss_t /= len(train_i)
            loss += args.lambda_u * loss_u + args.lambda_t * loss_t

        loss.backward()

        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(n_buckets=7):
        model.eval()
        mask = dataset.test_mask
        test_idx = dataset.test_idx
        pred = model(data.x_dict, data.edge_index_dict)[dataset.type_t][mask]
        # average of 0,1=0.5
        pred = (pred > 0.5).int()
        id_bucket = div_bucket_conti(hlid, test_idx, n_buckets)
        return dataset.dl.evaluate(pred.cpu(), id_bucket, n_buckets)

    @torch.no_grad()
    def node_clus():
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)

        labels_test = dataset.dl.labels_test
        X, y = out[dataset.type_t][labels_test["mask"][: nd_cnt[0]]].cpu().numpy(), labels_test["data"][labels_test["mask"]]

        '''handle IMDB dataset'''
        if args.dataset == 'IMDB':
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X.T, c=dataset.n_classes, m=2, error=0.005, maxiter=1000)
            u = u.T
            y = y.astype(float)
            l1, l2 = multilabel_dist(y, u)
            print(f"Manhattan distance (L1): {l1:.4f}")
            print(f"Euclidean distance (L2): {l2:.4f}")
        else:
            y = y.argmax(axis=-1)
            num_clusters = len(np.unique(y))  # Number of unique classes
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            predicted_labels = kmeans.fit_predict(X)

            nmi_score = normalized_mutual_info_score(y, predicted_labels)
            print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
            ari_score = adjusted_rand_score(y, predicted_labels)
            print(f"Adjusted Rand Score (ARI): {ari_score:.4f}")

    for epoch in range(args.n_pretrain):
        if epoch:
            '''re-sampling'''
            eidict_ctr = get_contrastive_g(dataset, hlid, args.p_inf, args.dens_scale)
        loss = train(pret=True)
        print(epoch, loss)
    for epoch in range(args.n_epochs):
        loss = train(pret=False)
        print(epoch, loss)

    res = test()
    print('micro-f1/macro-f1/variance/variance_b:')
    print(res['micro-f1'])
    print(res['macro-f1'])
    print(res['var'])
    print(np.sqrt(res['var_b']))
    node_clus()
