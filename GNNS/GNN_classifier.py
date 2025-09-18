import torch
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from GNNS.opt_functions import *
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GINConv, GraphConv, GraphNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import to_dense_batch

pooling_function = {'mean': global_mean_pool, 'add': global_add_pool, 'max': global_max_pool}
conv_layer = {'gat': GATv2Conv, 'gcn': GCNConv, 'gin': GINConv, 'graph': GraphConv}
 
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
        self.edge_weight = nn.Parameter(torch.ones(num_edges) * alpha, requires_grad=True)

    def __build_network__(self):
        x = 10

    def forward(self, x, edge_index):
        self.prev_vals = []
        self.edge_scale = np.ones(len(edge_index[0]))
        self.prev_vals.append(x[:, 1].tolist())
        feas_point = None
        conv = True
        for _ in range(self.num_itrs):
            x = self.propagate(edge_index, x=x, edge_weight = self.edge_weight, edge_scale = self.edge_scale)
            # # store feasible point
            # if feas(x):
            #     feas_point = x.clone()
            #     conv = True
            # else:
            #     conv = False
            #     # go back to feas 
            #     x = feas_point.clone() if feas_point is not None else x
            #     # take smaller steps in forward direction
            #     self.edge_scale[self.edge_dict[0]] = self.edge_scale[self.edge_dict[0]] * 1/self.scale
            #     self.edge_scale[self.edge_dict[1]] = self.edge_scale[self.edge_dict[1]] * 1/self.scale
            self.prev_vals.append(x[:, 1].detach().cpu().numpy())
            # convergence criteria
            # if (abs(self.prev_vals[-1][0] - self.prev_vals[-2][0]) + 
            #     abs(self.prev_vals[-1][1] - self.prev_vals[-2][1])) < self.conv_tol:
            #     if conv:
            #         break
        # return feas_point if feas_point is not None else x
        return x

    def message(self, x_i, x_j, edge_weight, edge_scale):
        msg = torch.zeros_like(edge_weight)
        for (ind,(i, j, e, s)) in enumerate(zip(x_i, x_j, edge_weight, edge_scale)):
            msg[ind] = s * e * self.constr_dict[f'{int(i[0])}-{int(j[0])}'](i, j)
        return msg.view(-1, 1) 
        

    def update(self, aggr_out, x):
        new_nodes = []
        for msg, node in zip(aggr_out, x):
            node = node.clone()          # break the view link
            node[1] = node[1] + msg[0]   # safe update
            node[1] = self._projection_(node)
            new_nodes.append(node)
        return torch.stack(new_nodes, dim=0)
    

    # def update(self, node_state, message): 
    #     update_node = node_state
    #     update_node[1] = node_state[1] + message
    #     return self._projection_(update_node)


    def _projection_(self, node):
        return max([self.proj_dict[int(node[0])][0], min([node[1], self.proj_dict[int(node[0])][1]])])

