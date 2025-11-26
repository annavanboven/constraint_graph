import sys, os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GNNS.GNN_classifier import PDConv, PDNetwork, PDFeasNetwork, PDFeasNetworkV2
from GNNS.loss_functions import TestLoss
from GNNS.opt_functions import *
import torch
import json
from math import sin, cos
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
import numpy as np
import wandb
from torch_geometric.nn.conv import MessagePassing
import os
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
debug_mode = False

model_dict = {
    'feas': PDFeasNetwork,
    'feasv2': PDFeasNetworkV2
}
class TestConfig():
    def __init__(self):
        self.lr = 0.05
        self.num_epochs = 5
        self.conv_tol = 0.001
        self.batch_size = 50
        self.optim_lst = [[4, 0.1], [6, 0.05], [8, 0.025]]
        self.feas_lst = [[3, 0.1], [5, 0.05], [7, 0.025]]
        self.constr_scale = 3
        self.model_name = 'feas'

def setup_wandb():
    wandb.login()
    sweep_config = {
        'method': 'grid'
        }
    metric = {
        'name': 'mse',
        'goal': 'minimize'   
        }
    sweep_config['metric'] = metric
    json_path = os.path.join(os.path.dirname(__file__), '..', 'parameters', 'gnn_params.JSON')
    with open(json_path) as file:
        params = json.load(file)

    sweep_config['parameters'] = params
    return sweep_config


def create_data(row):
    l1, l2 = _determine_init_violation_(row)
    # data point has 1) label, 2) value, 3) slope, 4) intercept (for this problem), 5) constraint value
    point = torch.tensor([[1, 5, 0, 0, 0],
            [2, 5, 0, 0, 0],
            [3, l1, row['m1'], row['b1'], l1],
            [4, l2, row['m2'], row['b2'], l2]], dtype = torch.float)
    label = torch.tensor([[row['xopt']], [row['yopt']]], dtype=torch.float)
    data = Data(x=point, edge_index=edge_index, f_edge_index = edge_infex_feas, y = label)
    return data

def _determine_init_violation_(row): 
    # how much is the first constraint violated?
    l1 = max(sin(5) + row['m1']*5 + row['b1'] - 5, 0)
    # how much is the second constraint violated?
    l2 = max(-cos(5) - row['m2']*5 + row['b2'] + 5, 0)
    return l1, l2


def train(train_loader, model, loss, opt):
    model.train()
    total_loss = 0
    n_dps = 0
    for (i, data) in enumerate(train_loader):
        print(f'training batch {i}')
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.f_edge_index)
        l = loss(out)
        l.backward()
        opt.step()
        total_loss += l.item()
        n_dps += 1
    return total_loss/n_dps

def evaluate(model, loader, conv_tol):
    model.eval()
    predictions = []
    labels = []
    num_optimal = 0
    num_feas = 0
    with torch.no_grad():
        for j,data in enumerate(loader):
            print(f'evaluating batch {j}')
            out = model(data.x, data.edge_index, data.f_edge_index)
            lab = data.y.numpy()
            predictions.append(list(zip(out[[i for i in range(len(out)) if i % 4 == 0], 1], 
                                out[[i for i in range(len(out)) if i % 4 == 1], 1])))
            labels.append(lab)
            i, k = 0, 0
            while i + 4 < len(out):
                batch = out[i:i+4, :]
                batch_lab = lab[k:k+2, :]
                # check for optimality
                if abs(batch[0][1] - batch_lab[0]) + abs(batch[1][1] - batch_lab[1]) < conv_tol:
                    num_optimal += 1
                # check for feasibility
                if feas(batch, [0], [1], [2], [3]):
                    num_feas += 1
                i += 4
                k += 2
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return mse, r2, predictions, labels, num_optimal, num_feas

def main(data_list, constr_dict, proj_dict, indices_dict, n_edges, nf_edges, config = None):
    if not debug_mode:
        wandb.init(config=config)
        run_name = wandb.run.name
        config = wandb.config
    model_name = config.model_name
    # create data
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle = True)
    
    model = model_dict[config.model_name]
    gnn = model(constr_dict, proj_dict, indices_dict, 
                          num_edges = n_edges*config.batch_size, num_feas_edges = nf_edges*config.batch_size,
                          num_nodes = 4, batch_size=config.batch_size,
                        optim_lst = config.optim_lst, feas_lst = config.feas_lst, convergence_tol=config.conv_tol)

    # create loss function
    obj_loss = TestLoss(obj_scale = 1, constr_scale = config.constr_scale)

    # create optimizer 
    optim = torch.optim.Adam(gnn.parameters(), lr = config.lr)

    # train network
    for epoch in range(config.num_epochs):
        print(f"epoch {epoch}")
        loss = train(train_loader, gnn, obj_loss, optim)
        mse, r2, predictions, labels, num_optimal, num_feas = evaluate(gnn, test_loader, config.conv_tol)
        if debug_mode:
            print(f"loss: {loss} \t num optimal: {num_optimal} \t num feas: {num_feas} \t mse: {mse} \n ")
        else:
            wandb.log({'epoch': epoch, 'loss': loss, 'num_optimal': num_optimal, 'num_feas': num_feas, 'mse': mse})
    
    # final evaluation
    mse, r2, predictions, labels, num_optimal, num_feas = evaluate(gnn, test_loader, config.conv_tol)
    success_criteria = num_optimal >= 0.95 * len(test_loader.dataset)
    run_str = round(num_optimal/len(test_loader.dataset), 3)
    if success_criteria:
        # save model
        pth = os.path.join(os.path.dirname(__file__), '..', 'models')
        torch.save(gnn.state_dict(), os.path.join(pth, f'{model_name}_{run_str}_{run_name}_statedict.pt'))
        json_pth = os.path.join(pth, 'stored_models.JSON')
        with open(json_pth, 'r') as file:
            runs = json.load(file)
        runs[run_name] =  {'optim_lst': config.optim_lst, 
                            'feas_lst': config.feas_lst,
                            'conv_tol': config.conv_tol,
                            'constr_scale': config.constr_scale}
        with open(json_pth, 'w') as f:
            json.dump(runs, f, indent=4)

#  setup
constr_dict = {
    '1-3': c1x1,
    '1-4': c2x1,
    '2-3': c1x2,
    '2-4': c2x2,
    '3-1': x1l1,
    '3-2': x2l1,
    '4-1': x1l2,
    '4-2': x2l2,
    '1-1': x1x1,
    '2-2': x2x2,
    '3-3': l1l1,
    '4-4': l2l2
    }

proj_dict = {
    1: [0, 10],
    2: [0,10],
    3: [0, 100],
    4: [0,100]
}
# create data point 
edge_index = EdgeIndex(
    [[0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3],
     [2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 2, 3]]
)
edge_infex_feas = EdgeIndex(
    [[0, 0, 1, 1, 2, 2, 3, 3, 2, 3],
     [2, 3, 2, 3, 0, 1, 0, 1, 2, 3]]
)
indices_dict = {
    0: [4, 6, 8],
    1: [5, 7, 9],
    2: [0, 2, 10],
    3: [1, 3, 11]
}   

if __name__ == '__main__':
    # pull in data
    data_df = pd.read_excel(os.path.join(os.path.dirname(__file__), "..", 'datasets', 'base_file.xlsx'))
    data_list = [create_data(row) for _, row in data_df.iterrows()]

    # run sweep
    if debug_mode:
        wb_config = TestConfig()
        main(data_list, constr_dict, proj_dict, indices_dict, 12, 10, config = wb_config)
    else:
        sweep_config = setup_wandb()
        sweep_id = wandb.sweep(sweep_config, project = f'constraint_graph_test')
        wandb.agent(sweep_id, function = lambda: main(data_list, constr_dict, proj_dict, indices_dict, 12, 10), count = 1000)
