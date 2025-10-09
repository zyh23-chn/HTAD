from typing import Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, HGTConv, GraphNorm, Linear, GATConv, SAGEConv


def tup2str(o):
    return '+'.join(o)


class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]], out_channels: int, metadata, device, hidden_channels=128, heads=8):
        super().__init__()
        self.device = device
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads, dropout=0.6, metadata=metadata)
        node_types = metadata[0]
        self.norm_dict = nn.ModuleDict()
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.norm_dict[node_type] = GraphNorm(hidden_channels)
            self.lin_dict[node_type] = Linear(hidden_channels, out_channels)
        # self.norm_dict = {node_type: GraphNorm(hidden_channels) for node_type in node_types}
        # self.lin_dict = {node_type: Linear(hidden_channels, out_channels) for node_type in node_types}
        '''for HGCL'''
        self.fc = nn.ModuleDict()
        for node_type in node_types:
            n_channels = in_channels[node_type]
            self.fc[node_type] = nn.Linear(n_channels, n_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)
        for node_type in x_dict:
            if out[node_type] is not None:
                out[node_type] = self.norm_dict[node_type](out[node_type])
                out[node_type] = self.lin_dict[node_type](out[node_type])
        return out

    def get_aux_view(self, x_dict):
        '''for HGCL, self-gating'''
        x_dict_aux = {}
        for node_type, fea in x_dict.items():
            # x_dict_aux[node_type] = torch.multiply(fea, F.sigmoid(self.fc[node_type](fea)))
            if fea.is_sparse:
                fea = fea.to_dense()
            x_dict_aux[node_type] = fea * F.sigmoid(self.fc[node_type](fea))
        return x_dict_aux


class HGT(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, device, num_heads=2, num_layers=1):
        super().__init__()
        self.device = device
        self.lin_dict = nn.ModuleDict()
        node_types = metadata[0]
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        self.norm_dict = nn.ModuleDict()
        self.lin_dict2 = nn.ModuleDict()
        for node_type in node_types:
            self.norm_dict[node_type] = GraphNorm(hidden_channels)
            self.lin_dict2[node_type] = Linear(hidden_channels, out_channels)
        # self.norm_dict = {node_type: GraphNorm(hidden_channels) for node_type in node_types}
        # self.lin_dict2 = {node_type: Linear(hidden_channels, out_channels) for node_type in node_types}

    def forward(self, x_dict, edge_index_dict):
        out = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}

        for conv in self.convs:
            out = conv(out, edge_index_dict)
        for node_type in out:
            out[node_type] = self.norm_dict[node_type](out[node_type])
            out[node_type] = self.lin_dict2[node_type](out[node_type])
        return out


class GAT(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, device, num_heads=8):
        super().__init__()
        self.device = device
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        node_types, edge_types = metadata
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        self.gat_convs = nn.ModuleDict()
        for edge_type in edge_types:
            self.gat_convs[tup2str(edge_type)] = GATConv(-1, hidden_channels // num_heads, num_heads, dropout=0.6)

        self.norm_dict = nn.ModuleDict()
        self.lin_dict2 = nn.ModuleDict()
        for node_type in node_types:
            self.norm_dict[node_type] = GraphNorm(hidden_channels)
            self.lin_dict2[node_type] = Linear(hidden_channels, out_channels)
        # self.norm_dict = {node_type: GraphNorm(hidden_channels).to(device) for node_type in node_types}
        # self.lin_dict2 = {node_type: Linear(hidden_channels, out_channels).to(device) for node_type in node_types}

    def forward(self, x_dict, edge_index_dict):
        out_dict = {node_type: torch.zeros((x.size(0), self.hidden_channels)).to(self.device) for node_type, x in x_dict.items()}
        emb_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            gat_conv = self.gat_convs[tup2str(edge_type)]
            src_x = emb_dict[src_type]
            dst_x = emb_dict[dst_type]
            out = gat_conv((src_x, dst_x), edge_index)
            out_dict[dst_type] += out

        for node_type, out in out_dict.items():
            out_dict[node_type] = self.norm_dict[node_type](out_dict[node_type])
            out_dict[node_type] = self.lin_dict2[node_type](out_dict[node_type])
        return out_dict


class SAGE(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, device):
        super().__init__()
        self.device = device
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        node_types, edge_types = metadata
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        self.sage_convs = nn.ModuleDict()
        for edge_type in edge_types:
            self.sage_convs[tup2str(edge_type)] = SAGEConv(-1, hidden_channels)

        self.norm_dict = nn.ModuleDict()
        self.lin_dict2 = nn.ModuleDict()
        for node_type in node_types:
            self.norm_dict[node_type] = GraphNorm(hidden_channels)
            self.lin_dict2[node_type] = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out_dict = {node_type: torch.zeros((x.size(0), self.hidden_channels)).to(self.device) for node_type, x in x_dict.items()}
        emb_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            sage_conv = self.sage_convs[tup2str(edge_type)]
            src_x = emb_dict[src_type]
            dst_x = emb_dict[dst_type]
            out = sage_conv((src_x, dst_x), edge_index)
            out_dict[dst_type] += out

        for node_type, out in out_dict.items():
            out_dict[node_type] = self.norm_dict[node_type](out_dict[node_type])
            out_dict[node_type] = self.lin_dict2[node_type](out_dict[node_type])
        return out_dict
