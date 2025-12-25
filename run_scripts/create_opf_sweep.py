import wandb 
from practice_gnn_network import setup_wandb


sweep_config = setup_wandb()
sweep_id = wandb.sweep(sweep_config, project="case6ww_acopf")

with open("sweep_id.txt", "w") as f:
    f.write(sweep_id)
