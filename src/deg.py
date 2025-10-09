import os.path as osp
from typing import Dict, List, Union
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, HGTConv, Linear, GraphNorm
from tqdm import tqdm
import matplotlib.pyplot as plt

from new_geodata import IMDB, DBLP, ACM, Freebase


class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]], out_channels: int, type_t: str, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads, dropout=0.6, metadata=data.metadata())
        self.norm = GraphNorm(hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.type_t = type_t

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        out = out[self.type_t]
        out = self.norm(out)
        out = self.lin(out)
        return out


class HGT(nn.Module):
    def __init__(self, out_channels, metadata, device, hidden_channels=64, num_heads=2, num_layers=1):
        super().__init__()
        self.device = device
        self.lin_dict = nn.ModuleDict()
        node_types = metadata[0]
        self.type_t = node_types[0]
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        self.norm = GraphNorm(hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        # self.norm_dict = {node_type: GraphNorm(hidden_channels) for node_type in node_types}
        # self.lin_dict2 = {node_type: Linear(hidden_channels, out_channels) for node_type in node_types}

    def forward(self, x_dict, edge_index_dict):
        out = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}

        for conv in self.convs:
            out = conv(out, edge_index_dict)
        return self.lin(self.norm(out[self.type_t]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ACM")
    parser.add_argument("--model", type=str, default="HGT")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--n_buckets", type=int, default=7, help='number of buckets')
    parser.add_argument("--label_rate", type=float, default=.01)
    parser.add_argument("--rel_reg", type=float, default=500.0)
    parser.add_argument("--self_reg", type=float, default=3.0)
    args = parser.parse_args()
    return args


def draw(vals, metric: str):
    n = len(vals)
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 2))
    ax.bar(range(n), vals, color="gray", width=0.6)#orange, skyblue, gray
    ax.set_xticks(range(n), list(range(n)))
    # ax.set_xticks([])
    # ax.set_xlabel(metric)
    ax.set_ylabel("Accuracy")
    plt.savefig(osp.join("./saved", metric + "+" + args.dataset + "+" + args.model + "+" + str(args.label_rate).replace('.', '_')))
    rk = np.argsort(np.argsort(vals))
    rk -= np.array([i for i in range(n)])
    rk **= 2
    print(1 - 6 * (np.sum(rk)) / n / (n**2 - 1))


if __name__ == "__main__":
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
    n_types = dataset.n_classes
    data = HeteroData()
    for i, node_type in enumerate(dataset.node_types):
        data[node_type].x = dataset.features[i]
    for i, link_type in enumerate(dataset.link_types):
        data[link_type].edge_index = dataset.edge_index_dict[i]

    type_t = dataset.type_t
    if args.dataset == 'IMDB':
        data[type_t].y = torch.from_numpy(dataset.labels).float()  # multi-label target
    else:
        data[type_t].y = torch.from_numpy(dataset.labels).long()  # only masked positions have true label
    data_t = data[type_t]  # for training

    if args.model == 'HAN':
        model = HAN(in_channels=-1, out_channels=n_types, type_t=type_t)
    elif args.model == 'HGT':
        model = HGT(n_types, data.metadata(), device)
    data, model = data.to(device), model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    def train() -> float:
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)

        train_i = dataset.train_i
        loss = F.cross_entropy(out[train_i], data_t.y[train_i])
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test():
        model.eval()
        mask = dataset.test_mask
        pred = model(data.x_dict, data.edge_index_dict)[mask]
        '''average of 0,1=0.5'''
        pred = (pred > 0.5).int()
        return dataset.dl.evaluate(pred.cpu())

    for epoch in range(args.n_epochs):
        loss = train()
        print(loss)
        # train_acc, val_acc, test_acc = test()

    res = test()
    print('micro-f1: ', res['micro-f1'])
    print('macro-f1: ', res['macro-f1'])

    model.eval()
    with torch.no_grad():
        pred = model(data.x_dict, data.edge_index_dict)  # (n_nodes_t,)
        pred = (pred > 0.5).int().cpu()
        # for i in range(n_nodes_t):
        #     tot[deg[i]] += 1
        #     pos[deg[i]] += 1 if pred[i] == data_t.y[i] else 0

        indices_test = dataset.test_idx
        n_test_t = indices_test.shape[0]
        n_nodes_t = data_t.x.size(0)  # number of target type nodes

        '''Degree'''
        deg = np.zeros(n_nodes_t)
        for edge_type, edges in data.edge_index_dict.items():
            nd_type0, nd_type1 = edge_type[0], edge_type[-1]
            if nd_type0 == type_t:
                _, n_edges = edges.shape  # (2,n_edges)
                for j in range(n_edges):
                    '''bi-directional'''
                    deg[edges[0][j]] += 1
        deg = deg[indices_test]
        x_l, x_r = min(deg), max(deg)
        diff = x_r - x_l + 1
        n_buckets_deg = min(args.n_buckets, diff)
        interval_len = diff // n_buckets_deg
        rem = diff % n_buckets_deg
        bucket_deg = [0] * n_nodes_t
        id_deg = np.stack((deg, indices_test), axis=-1)
        id_deg = id_deg[np.lexsort(id_deg.T[::-1])]
        y_l, j = x_l, 0
        for i in range(n_buckets_deg):
            y_r = y_l + interval_len + (1 if i < rem else 0)
            while j < n_test_t and id_deg[j][0] < y_r:
                bucket_deg[round(id_deg[j][1])] = i
                j += 1
            y_l = y_r
        assert y_l == x_r + 1 and j == n_test_t

        hlid = dataset.get_hlid().squeeze()
        n_buckets_hlid = min(args.n_buckets, n_test_t)
        interval_len = n_test_t // n_buckets_hlid
        rem = n_test_t % n_buckets_hlid
        id_hlid = np.stack((hlid[indices_test], indices_test), axis=-1)  # convert indices into float here!
        id_hlid = id_hlid[np.lexsort(id_hlid.T[::-1])]
        bucket_hlid = [0] * n_nodes_t
        y_l = 0
        for i in range(n_buckets_hlid):
            y_r = y_l + interval_len + (1 if i < rem else 0)
            for j in range(y_l, y_r):
                bucket_hlid[round(id_hlid[j][1])] = i
            y_l = y_r
        assert y_l == n_test_t

        y_true = dataset.y_true
        tot_deg = np.zeros(n_buckets_deg)
        pos_deg = np.zeros(n_buckets_deg)
        tot_hlid = np.zeros(n_buckets_hlid)
        pos_hlid = np.zeros(n_buckets_hlid)
        for i, idx in enumerate(indices_test):
            # res = 1 if pred[idx].argmax() == y_true[i].argmax() else 0
            res = f1_score(pred[idx], y_true[i])
            tot_deg[bucket_deg[idx]] += 1
            pos_deg[bucket_deg[idx]] += res
            tot_hlid[bucket_hlid[idx]] += 1
            pos_hlid[bucket_hlid[idx]] += res
        tot_deg[tot_deg == 0] = 1

        plt.rcParams.update({'font.size': 18})
        draw(pos_deg / tot_deg, 'deg')
        draw(pos_hlid / tot_hlid, 'hlid')
