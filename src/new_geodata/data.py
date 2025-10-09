from .data_loader import load_data

import random
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, inv
from heapq import heappop, heappush
from tqdm import tqdm
import threading
import os
import os.path as osp


class GeoDataset:
    def __init__(self, root, name_, args, alpha=0.85) -> None:
        '''alpha: teleport probability presented in appnp'''
        self.name_ = name_
        self.features, self.adjM, self.labels, train_val_test_idx, dl = load_data(root, name_)
        # do not divide validate
        n_nodes_t = dl.nodes["count"][0]
        train_mask = dl.labels_train["mask"][:n_nodes_t]
        # label rate works here
        train_i = np.nonzero(train_mask)[0]
        n_train = train_i.shape[0]
        self.train_i = np.random.choice(train_i, int(args.label_rate * n_train), replace=False)
        self.test_mask = dl.labels_test["mask"][:n_nodes_t]
        self.test_idx = train_val_test_idx['test_idx']
        self.y_true = dl.labels_test["data"][dl.labels_test["mask"]]
        self.dl = dl
        self.alpha = alpha
        self.rel_reg = args.rel_reg
        self.self_reg = args.self_reg
        # self.node_types = None
        # self.link_types = None

    def _get_ed(self):
        node_types = self.node_types
        link_types = self.link_types
        edge_index_dict = self.dl.edge_index_dict
        nd_dict = {s: i for i, s in enumerate(node_types)}
        sft = self.dl.nodes['shift']
        for i, link_type in enumerate(link_types):
            nd0, nd1 = nd_dict[link_type[0]], nd_dict[link_type[-1]]
            edge_index_dict[i][0] -= sft[nd0]
            edge_index_dict[i][1] -= sft[nd1]

        return edge_index_dict

    def _get_rel(self):
        nt_dict = {nt: i for i, nt in enumerate(self.node_types)}
        n_nt = len(self.node_types)
        Rel = np.ones((n_nt, n_nt))
        for i, edge_index in self.edge_index_dict.items():
            src_type, dst_type = self.link_types[i][0], self.link_types[i][-1]
            src_type, dst_type = nt_dict[src_type], nt_dict[dst_type]
            Rel[src_type][dst_type] = 1 + self.rel_reg / edge_index.size(1)
        Rel[0][0] += self.self_reg
        return Rel

    def get_deg(self):
        nt_dict = {nt: i for i, nt in enumerate(self.node_types)}
        nd_cnt = self.dl.nodes['count']
        deg = dict()
        for i in range(len(self.node_types)):
            deg[i] = np.zeros(nd_cnt[i])
        for i, edge_index in self.edge_index_dict.items():
            src_type, dst_type = self.link_types[i][0], self.link_types[i][-1]
            src_type, dst_type = nt_dict[src_type], nt_dict[dst_type]
            for j in edge_index[0]:
                deg[src_type][j] += 1
        return deg

    def get_sp(self, inf_d):
        '''shortest path'''
        offsets = list(self.dl.nodes['shift'].values())
        adjM = self.adjM
        dl = self.dl
        n_tot = dl.nodes['total']
        assert n_tot == adjM.shape[0]
        tp = np.zeros(n_tot).astype(int)
        offsets.append(n_tot)
        for i in range(len(offsets) - 1):
            tp[offsets[i] : offsets[i + 1]] = i
        g = [[] for _ in range(n_tot)]
        i_indices, j_indices = adjM.nonzero()
        for i, j in zip(i_indices, j_indices):
            g[i].append(j)
            g[j].append(i)
        dmin = np.ones(n_tot) * inf_d

        def _dijkstra(i):
            '''Dijkstra'''
            vi = [False] * n_tot
            hq = []
            heappush(hq, [0, i])
            while hq:
                d, u = heappop(hq)
                if vi[u] or d == inf_d:
                    continue
                vi[u] = True
                dmin[u] = min(dmin[u], d)
                for v in g[u]:
                    if not vi[v]:
                        heappush(hq, [d + 1, v])

        srcs = self.train_i
        for i in tqdm(srcs):
            _dijkstra(i)
        return dmin

    def get_hlid(self, weigh=False):
        '''time consuming, inverse matrix calculation'''
        n_node_types = len(self.node_types)
        offsets = self.dl.nodes['shift']
        # calculate pagerank here
        adjM = self.adjM
        dl = self.dl
        N = dl.nodes['total']

        # node_shift=dl.nodes['shift'].values()
        # node_count=dl.nodes['count'].values()
        # mymat=adjM.toarray()
        # for il,ix in zip(node_shift,node_count):
        #     ir=il+ix
        #     for jl,jx in zip(node_shift,node_count):
        #         jr=jl+jx
        #         if (mymat[il:ir,jl:jr]!=0).any():
        #             print(il,jl)
        # adjmat=adjM.toarray()
        # for i in range(n_node_types):
        #     li,ri=offsets[i] , (offsets[i + 1] if i + 1 < n_node_types else N)
        #     for j in range(n_node_types):
        #         lj,rj=offsets[j] , (offsets[j + 1] if j + 1 < n_node_types else N)
        #         if (adjmat[li:ri,lj:rj]!=0).any():
        #             print(i,j)
        assert N == adjM.shape[0]
        file_path=...
        file_path = './midres/' + self.name_ + '+' + str(self.rel_reg).replace('.', '_') + '+' + str(self.self_reg).replace('.', '_') + '.npz'
        if os.path.exists(file_path):
            ppr = np.load(file_path)['ppr']
        else:
            out_links = np.array(adjM.sum(axis=1))  # (N,1)
            out_links[out_links == 0] = 1
            At = adjM.multiply(1 / out_links)  # norm
            '''obtain label matrix'''
            lm = np.zeros((N, n_node_types))
            for i in range(n_node_types):
                lm[offsets[i] : (offsets[i + 1] if i + 1 < n_node_types else N), i] = 1
            At = At.toarray()
            if weigh:
                Rel = self.Rel
                # Bt=At
                # At *= lm @ Rel @ lm.T
                # i_indices,j_indices=Bt.nonzero()
                # for i,j in zip(i_indices,j_indices):
                #     if At[i][j]==0:
                #         print(i,j)
            else:
                Rel = np.ones((n_node_types, n_node_types))

            # for i in range(n_node_types):
            #     li,ri=offsets[i] , (offsets[i + 1] if i + 1 < n_node_types else N)
            #     for j in range(n_node_types):
            #         lj,rj=offsets[j] , (offsets[j + 1] if j + 1 < n_node_types else N)
            #         if (At[li:ri,lj:rj]!=0).all():
            #             print(i,j)
            for i in range(n_node_types):
                li, ri = offsets[i], (offsets[i + 1] if i + 1 < n_node_types else N)
                for j in range(n_node_types):
                    lj, rj = offsets[j], (offsets[j + 1] if j + 1 < n_node_types else N)
                    At[li:ri, lj:rj] *= Rel[i][j]

            I = np.eye(N)
            print('Calculating inverse matrix...')
            ppr = np.linalg.inv(I - (1 - self.alpha) * At)
            np.savez_compressed(file_path, ppr=ppr)
        label_v = np.zeros(dl.nodes['total'])
        for i in self.train_i:
            label_v[i] = 1
        label_v = label_v.reshape(-1, 1)
        hlid = np.array(ppr @ label_v).squeeze()
        return hlid


class IMDB(GeoDataset):
    def __init__(
        self,
        root: str,
        args,
    ) -> None:
        self.node_types = ["movie", "director", "actor", "keyword"]
        self.link_types = [
            ('movie', 'director'),
            ('director', 'movie'),
            ('movie', 'actor'),
            ('actor', 'movie'),
            ('movie', 'keyword'),
            ('keyword', 'movie'),
        ]
        self.mps = [[0, 1], [2, 3]]  # MDM, MAM
        self.type_t = "movie"
        self.n_classes = 5
        super().__init__(root, 'IMDB', args)
        # modify edge index here, subtract by offset
        self.edge_index_dict = super()._get_ed()
        self.Rel = super()._get_rel()


class DBLP(GeoDataset):
    def __init__(
        self,
        root: str,
        args,
    ) -> None:
        self.node_types = ["author", "paper", "term", "venue"]
        self.link_types = [
            ("author", "paper"),
            ("paper", "term"),
            ("paper", "venue"),
            ("paper", "author"),
            ("term", "paper"),
            ("venue", "paper"),
        ]
        self.mps = [[0, 3], [0, 1, 4, 3], [0, 2, 5, 3]]  # APA, APTPA, APVPA
        self.type_t = "author"
        self.n_classes = 4
        super().__init__(root, 'DBLP', args)
        # modify edge index here, subtract by offset
        self.edge_index_dict = super()._get_ed()
        self.Rel = super()._get_rel()


class ACM(GeoDataset):
    def __init__(
        self,
        root: str,
        args,
    ) -> None:
        self.node_types = ["paper", "author", "subject", "term"]
        self.link_types = [
            ("paper", "cite", "paper"),
            ("paper", "ref", "paper"),
            ("paper", "author"),
            ("author", "paper"),
            ("paper", "subject"),
            ("subject", "paper"),
            ("paper", "term"),
            ("term", "paper"),
        ]
        self.mps = [[2, 3], [4, 5]]  # PAP, PSP
        self.type_t = "paper"
        self.n_classes = 3
        super().__init__(root, 'ACM', args)
        # modify edge index here, subtract by offset
        self.edge_index_dict = super()._get_ed()
        self.Rel = super()._get_rel()


class Freebase(GeoDataset):
    def __init__(
        self,
        root: str,
        args,
    ) -> None:
        self.node_types = ["BOOK", "FILM", "MUSIC", "SPORTS", "PEOPLE", "LOCATION", "ORGANIZATION", "BUSINESS"]
        self.link_types = [
            ('BOOK', 'and', 'BOOK'),
            ('BOOK', 'to', 'FILM'),
            ('BOOK', 'on', 'SPORTS'),
            ('BOOK', 'on', 'LOCATION'),
            ('BOOK', 'about', 'ORGANIZATION'),
            ('FILM', 'and', 'FILM'),
            ('MUSIC', 'in', 'BOOK'),
            ('MUSIC', 'in', 'FILM'),
            ('MUSIC', 'and', 'MUSIC'),
            ('MUSIC', 'for', 'SPORTS'),
            ('MUSIC', 'on', 'LOCATION'),
            ('SPORTS', 'in', 'FILM'),
            ('SPORTS', 'and', 'SPORTS'),
            ('SPORTS', 'on', 'LOCATION'),
            ('PEOPLE', 'to', 'BOOK'),
            ('PEOPLE', 'to', 'FILM'),
            ('PEOPLE', 'to', 'MUSIC'),
            ('PEOPLE', 'to', 'SPORTS'),
            ('PEOPLE', 'and', 'PEOPLE'),
            ('PEOPLE', 'on', 'LOCATION'),
            ('PEOPLE', 'in', 'ORGANIZATION'),
            ('PEOPLE', 'in', 'BUSINESS'),
            ('LOCATION', 'in', 'FILM'),
            ('LOCATION', 'and', 'LOCATION'),
            ('ORGANIZATION', 'in', 'FILM'),
            ('ORGANIZATION', 'to', 'MUSIC'),
            ('ORGANIZATION', 'to', 'SPORTS'),
            ('ORGANIZATION', 'on', 'LOCATION'),
            ('ORGANIZATION', 'and', 'ORGANIZATION'),
            ('ORGANIZATION', 'for', 'BUSINESS'),
            ('BUSINESS', 'about', 'BOOK'),
            ('BUSINESS', 'about', 'FILM'),
            ('BUSINESS', 'about', 'MUSIC'),
            ('BUSINESS', 'about', 'SPORTS'),
            ('BUSINESS', 'on', 'LOCATION'),
            ('BUSINESS', 'and', 'BUSINESS'),
        ]
        self.type_t = "BOOK"
        self.n_classes = 7
        super().__init__(root, 'Freebase', args)
        # modify edge index here, subtract by offset
        self.edge_index_dict = super()._get_ed()
        self.Rel = super()._get_rel()
