import sys, os
import argparse
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GNNS.GNN_classifier import  PDFeasNetworkV2, PDFeasNetworkV3
from GNNS.loss_functions import OPFLoss
from GNNS.pdgnn import PDGnn
from GNNS import opf_boundary_functions
from GNNS import opf_linelimit_functions
from GNNS import opf_powerbalance_functions
from GNNS import opt_functions
import pickle
import numpy as np
import json
import torch
import wandb
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

debug_mode = True

model_dict = {
    'feasv3': PDFeasNetworkV3,
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
    json_path = os.path.join(os.path.dirname(__file__), '..', 'parameters', 'pdgnn_params.JSON')
    with open(json_path) as file:
        params = json.load(file)

    sweep_config['parameters'] = params
    return sweep_config

def train(train_loader, pdgnn, model, loss, opt):
    model.train()
    total_loss = 0
    n_dps = 0
    for (i, data) in enumerate(train_loader):
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.f_edge_index)
        obj_val, constr_val = pdgnn.get_objective(out), pdgnn.get_violation(out)
        l = loss(obj_val, constr_val)
        l.backward()
        opt.step()
        total_loss += l.item()
        n_dps += 1
    return total_loss/n_dps


def evaluate(pdgnn, model, loader):
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
            pred = pdgnn.get_predictions(out)
            predictions.append(pred)
            labels.append(lab)
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return mse, r2, predictions, labels, num_optimal, num_feas

def main(case_name, config = None): 
    model_name = 'TEST'
    if not debug_mode:
        wandb.init(config=config)
        run_name = wandb.run.name
        config = wandb.config
        model_name = config.model_name
        pdgnn = load_model(case_name, config)

    # create data
    train_data, test_data = train_test_split(pdgnn.dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle = True)
    # create model 
    model = model_dict[config.model_name]
    gnn = pdgnn.create_model(model, batch_size = config.batch_size, 
                             optim_lst = config.optim_lst, feas_lst = config.feas_lst, 
                             conv_tol = config.conv_tol)
    # create loss function
    obj_loss = OPFLoss(obj_scale = 1, constr_scale = config.constr_scale)

    # create optimizer 
    optim = torch.optim.Adam(gnn.parameters(), lr = config.lr)

    # train model 
    for epoch in range(config.num_epochs):
        loss = train(train_loader, gnn, obj_loss, optim)
        mse, r2, predictions, labels, num_optimal, num_feas = evaluate(pdgnn, gnn, test_loader)
        if debug_mode:
            print(f"loss: {loss} \t mse: {mse} \n ")
        else:
            wandb.log({'epoch': epoch, 'loss': loss, 'mse': mse})

    # final evaluation
    mse, r2, predictions, labels, num_optimal, num_feas = evaluate(pdgnn, gnn, test_loader)
    success_criteria = num_optimal >= 0.6 * len(test_loader.dataset)
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

def load_model(case_name, config): 
    # unpickle object
    with open(os.path.join(os.path.dirname(__file__), f"../models/pdgnn_parents/{case_name}_pdgnn.pkl"), "rb") as f:
        pdgnn = pickle.load(f)
    # store function pointers 
    pdgnn.translate_constr_dict(pickle = False)
    # create a model in the parent function
    model = pdgnn.create_model('feasv3',  optim_lst = config.optim_lst,
                            feas_lst = config.feas_lst)
    return model

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Test Constraint Graph')
    parser.add_argument(
        '--case_name', 
        type=str, 
        default=os.path.join(os.path.dirname(__file__), ".."), 
        help="case name"
    )
    parser.add_argument(
        '--sweep_id', 
        type=str, 
        default="", 
        help="id for wandb sweep"
    )
    args = parser.parse_args()
    wandb.agent(args.sweep_id, project="case6ww_acopf", entity="annavb", function = lambda: main(args.case_name), count = 1000)



