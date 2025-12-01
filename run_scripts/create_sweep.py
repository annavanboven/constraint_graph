import wandb 
from practice_gnn_network import setup_wandb

sweep_config = setup_wandb()
sweep_id = wandb.sweep(sweep_config, project="constraint_graph_test")
print(sweep_id)