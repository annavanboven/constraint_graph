import torch
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from GNNS.opt_functions import *
from GNNS.opt_functions import feas as feas_func
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, GraphConv, GraphNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import to_dense_batch

pooling_function = {'mean': global_mean_pool, 'add': global_add_pool, 'max': global_max_pool}
conv_layer = {'gat': GATv2Conv, 'gcn': GCNConv, 'gin': GINConv, 'graph': GraphConv}

class PDFeasNetworkV2(nn.Module):
    def __init__(self, constraint_dict, proj_dict, indices_dict, num_edges, num_feas_edges,
                 num_nodes, batch_size, optim_lst, feas_lst, convergence_tol, scale = 1, **kwargs):
        super(PDFeasNetworkV2, self).__init__()
        self.conv_tol = convergence_tol
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.feas_threshold = 0.75
        self.num_feas_edges = num_feas_edges
        self.x1_indices = [i for i in range(num_nodes*batch_size) if i % 4 == 0]
        self.x2_indices = [i for i in range(num_nodes*batch_size) if i % 4 == 1]
        self.l1_indices = [i for i in range(num_nodes*batch_size) if i % 4 == 2]
        self.l2_indices = [i for i in range(num_nodes*batch_size) if i % 4 == 3]
        self._build_network_(constraint_dict, proj_dict, indices_dict, optim_lst, feas_lst)

    def _build_network_(self, constraint_dict, proj_dict, indices_dict, optim_lst, feas_lst):
        self.optim_lst = nn.ModuleList()
        self.optim_norm = nn.ModuleList()
        self.feas_lst = nn.ModuleList()
        self.feas_norm = nn.ModuleList()
        for feas in feas_lst:
            feas_layer = PDConv(constraint_dict, proj_dict, indices_dict, 
                    num_edges = self.num_feas_edges, itrs = feas[0], alpha = feas[1], convergence_tol=self.conv_tol)
            self.feas_lst.append(feas_layer)
            self.feas_norm.append(nn.LayerNorm(self.num_feas_edges))
        for optim in optim_lst:
            optim_layer = PDConv(constraint_dict, proj_dict, indices_dict, 
                           num_edges = self.num_edges, itrs = optim[0], alpha = optim[1], convergence_tol=self.conv_tol)
            self.optim_lst.append(optim_layer)
            self.optim_norm.append(nn.LayerNorm(self.num_edges))

    def forward(self, x, edge_index, f_edge_index):
        for (layer, norm_layer) in zip(self.optim_lst, self.optim_norm):
            x = layer(x, edge_index)
            # layer.edge_weight = torch.sigmoid(layer.edge_weight)
            if feas_func(x, self.x1_indices, self.x2_indices, self.l1_indices, self.l2_indices) < self.feas_threshold:
                for (feas_layer, feas_norm) in zip(self.feas_lst, self.feas_norm):
                    x = feas_layer(x, f_edge_index)
                    # feas_layer.edge_weight = torch.sigmoid(feas_layer.edge_weight)
                    if feas_func(x, self.x1_indices, self.x2_indices, self.l1_indices, self.l2_indices) > self.feas_threshold:
                        break
        return x



class PDFeasNetwork(nn.Module):
    def __init__(self, constraint_dict, proj_dict, indices_dict, num_edges, num_feas_edges,
                 num_nodes, batch_size,
                 optim_lst, feas_lst, convergence_tol, scale = 1, **kwargs):
        super(PDFeasNetwork, self).__init__()
        self.conv_tol = convergence_tol
        self.num_edges = num_edges
        self.num_feas_edges = num_feas_edges
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.x1_indices = [i for i in range(num_nodes*batch_size) if i % 4 == 0]
        self.x2_indices = [i for i in range(num_nodes*batch_size) if i % 4 == 1]
        self.l1_indices = [i for i in range(num_nodes*batch_size) if i % 4 == 2]
        self.l2_indices = [i for i in range(num_nodes*batch_size) if i % 4 == 3]
        self._build_network_(constraint_dict, proj_dict, indices_dict, optim_lst, feas_lst)

    def _build_network_(self, constraint_dict, proj_dict, indices_dict, optim_lst, feas_lst):
        self.optim_lst = nn.ModuleList()
        self.feas_lst = nn.ModuleList()
        for (feas, optim) in zip(feas_lst, optim_lst):
            optim_layer = PDConv(constraint_dict, proj_dict, indices_dict, 
                           num_edges = self.num_edges, itrs = optim[0], alpha = optim[1], convergence_tol=self.conv_tol)
            feas_layer = PDConv(constraint_dict, proj_dict, indices_dict, 
                             num_edges = self.num_feas_edges, itrs = feas[0], alpha = feas[1], convergence_tol=self.conv_tol)
            self.optim_lst.append(optim_layer)
            self.feas_lst.append(feas_layer)

    def forward(self, x, edge_index, f_edge_index):
        for (optim_layer, feas_layer) in zip(self.optim_lst, self.feas_lst):
            infeas = 1 - feas_func(x, self.x1_indices, self.x2_indices, self.l1_indices, self.l2_indices)
            # infeas = torch.sigmoid(amt_feas(x, self.x1_indices, self.x2_indices, self.l1_indices, self.l2_indices))
            optim = optim_layer(x, edge_index)
            feas = feas_layer(x, f_edge_index)
            x= (1 - infeas) * optim + infeas * feas
        return x


class PDNetwork(nn.Module):
    def __init__(self, constraint_dict, proj_dict, indices_dict, num_edges, 
                 itr_lst, alpha_lst, convergence_tol, scale = 1, **kwargs):
        super(PDNetwork, self).__init__()
        self.conv_tol = convergence_tol
        self.num_edges = num_edges
        self._build_network_(constraint_dict, proj_dict, indices_dict, itr_lst, alpha_lst)

    def _build_network_(self, constraint_dict, proj_dict, indices_dict, itr_lst, alpha_lst):
        self.pd_lst = nn.ModuleList()
        for (itr, alpha) in zip(itr_lst, alpha_lst):
            layer = PDConv(constraint_dict, proj_dict, indices_dict, 
                           num_edges = self.num_edges, itrs = itr, alpha = alpha, convergence_tol=self.conv_tol)
            self.pd_lst.append(layer)

    def forward(self, x, edge_index):
        for layer in self.pd_lst:
            x = layer(x, edge_index)

        return x


 
class PDConv(MessagePassing):
    def __init__(self, constraint_dict, proj_dict, indices_dict, num_edges, itrs, 
                 alpha, convergence_tol, scale = 1, **kwargs,):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.constr_dict = constraint_dict
        self.proj_dict = proj_dict
        self.num_itrs = itrs
        self.conv_tol = convergence_tol
        self.prev_vals = []
        self.edge_dict = indices_dict
        self.edge_scale = np.ones(num_edges)
        self.scale = scale
        self.alpha = alpha
        self.edge_weight = nn.Parameter(torch.ones(num_edges) * alpha, requires_grad=True)

    def __build_network__(self):
        x = 10

    def forward(self, x, edge_index):
        self.prev_vals = []
        self.edge_scale = np.ones(len(edge_index[0]))
        self.prev_vals.append(x[:, 1].tolist() + x[[2,3],[4]].tolist())
        feas_point = None
        conv = True
        for _ in range(self.num_itrs):
            norm_weight = torch.tanh(self.edge_weight)
            x = self.propagate(edge_index, x=x, edge_weight = norm_weight, edge_scale = self.edge_scale)
            self.prev_vals.append(x[:, 1].detach().cpu().numpy().tolist() + x[[2,3],[4]].detach().cpu().numpy().tolist())
        # return feas_point if feas_point is not None else x
        return x

    def message(self, x_i, x_j, edge_weight, edge_scale):
        msg = torch.zeros_like(edge_weight)
        for (ind,(i, j, e, s)) in enumerate(zip(x_i, x_j, edge_weight, edge_scale)):
            msg[ind] = s * e * self.constr_dict[f'{int(i[0].round().item())}-{int(j[0].round().item())}'](i, j)
        return msg.view(-1, 1) 
        

    def update(self, aggr_out, x):
        new_nodes = []
        for msg, node in zip(aggr_out, x):
            node = node.clone()          
            node[1] = node[1] + msg[0]  
            node[1] = self._projection_(node)
            node[4] = msg[0]/self.alpha # store the previous iteration's constraint value
            new_nodes.append(node)
        return torch.stack(new_nodes, dim=0)


    def _projection_(self, node):
        return max([self.proj_dict[int(node[0])][0], min([node[1], self.proj_dict[int(node[0])][1]])])

