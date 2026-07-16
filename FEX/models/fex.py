from ..training.train_controller import train_network_controller, train_controller
from ..training.train_fex import train_network_fex, train_fex 
from ..training.train_configs import FEXConfig, ControllerConfig
from FEX.utils.tree_configs import *


import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class CoupledFEX():
    def __init__(self, self_fex_struct, inter_fex_struct, target_dim, *, 
            controller_lr=0.005, controller_epochs=200, cands_per_cycle=10, controller_threshold=0.4, poolsize=10, epsilon_greedy=0.2,
            num_fex_epochs=80, bfgs_epochs=20, self_lr=0.2, inter_lr=0.2, bfgs_lr=0.5, 
            finetune_epochs=2000, finetune_lr=2e-3, expression_threshold=0.001,
            device='cpu'
        ):
        self.self_fex_struct = get_tree_config(self_fex_struct)
        self.inter_fex_struct = get_tree_config(inter_fex_struct)
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        
        self.fex_config = FEXConfig(
            target_dim=target_dim,
            lr=self_lr,
            inter_lr=inter_lr,
            num_epochs=num_fex_epochs,
            bfgs_epochs=bfgs_epochs,
            bfgs_lr=bfgs_lr,
            expression_threshold=expression_threshold,

        )
        self.controller_config = ControllerConfig(
            lr=controller_lr,
            num_epochs=controller_epochs,
            num_cands_per_epoch=cands_per_cycle,
            percentile_threshold=controller_threshold,
            poolsize=poolsize,
            epsilon_greedy=epsilon_greedy,
        )

        self.best_model = None
        self.device=device

    def to(self, device):
        self.device = device
        if self.best_model is not None:
            self.best_model.forcing_tree.to(device)
            self.best_model.inter_fex.to(device)
        return self



    def fit(self, data, target, adjacency, batch_size=64, num_workers=2, finetune_bs=512):
        dataloader = DataLoader(
            TensorDataset(data, target),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device == "cuda",
        )
        best_candidates = train_network_controller(self.self_fex_struct, self.inter_fex_struct, dataloader, adjacency, self.controller_config, self.fex_config, num_workers=num_workers)
        self.fex_config.num_epochs = self.finetune_epochs
        self.fex_config.lr = self.finetune_lr
        self.fex_config.inter_lr = self.finetune_lr
        self.fex_config.bfgs_epochs = 0
        if finetune_bs:
            dataloader= DataLoader(
                TensorDataset(data, target),
                batch_size=finetune_bs,
                shuffle=True,
                pin_memory=self.device == "cuda",
            )
        for candidate in best_candidates:
            loss = train_network_fex(candidate.forcing_tree, candidate.inter_tree, dataloader, adjacency, self.fex_config, verbose=True, log_every=100)
            candidate.reward = 1 / (1 + loss)

        self.best_model = max(best_candidates, key=lambda c: c.reward)

    
    def predict(self, x, adjacency):
        self_fex = self.best_model.forcing_tree.to(self.device)
        inter_fex = self.best_model.inter_tree.to(self.device)

        self_fex.eval()
        inter_fex.eval()

        x = x.to(self.device)
        adjacency = adjacency.to(self.device)

        with torch.no_grad():
            predictions = self_fex(x).squeeze(-1)

            self_nodes, neighbor_nodes = adjacency.nonzero(as_tuple=True)

            inter_input = torch.cat((x[self_nodes], x[neighbor_nodes]), dim=1)
            inter_predictions = inter_fex(inter_input).squeeze(-1)
            predictions.index_add_(0, self_nodes, inter_predictions)

        return predictions
    
    
    def __str__(self):
        return f"FEX(self_fex_struct={self.best_model.forcing_tree} \ninter_fex_struct={self.best_model.inter_fex})"
    
class SingleFEX():
    def __init__(self, self_fex_struct: str, target_dim, *,
            controller_lr=0.01, controller_epochs=200, cands_per_cycle=10, controller_threshold=0.4, poolsize=5, epsilon_greedy=0.2,
            num_fex_epochs=60, self_lr=0.2, bfgs_epochs=20, bfgs_lr=0.4, 
            num_finetune_epochs=2000, finetune_lr=0.02, expression_threshold=0.01,
            device='cpu',
        ):
        self.self_fex_struct = get_tree_config(self_fex_struct)
        self.fex_config = FEXConfig(
            num_epochs=num_fex_epochs,
            lr=self_lr,
            bfgs_epochs=bfgs_epochs,
            bfgs_lr=bfgs_lr,
            target_dim=target_dim,
            expression_threshold=expression_threshold,
        )
        self.controller_config = ControllerConfig(
            lr=controller_lr,
            num_epochs=controller_epochs,
            num_cands_per_epoch=cands_per_cycle,
            percentile_threshold=controller_threshold,
            poolsize=poolsize,
            epsilon_greedy=epsilon_greedy,
        )
        self.best_model = None
        self.device=device
        self.num_finetune_epochs = num_finetune_epochs
        self.finetune_lr = finetune_lr

    def to(self, device):
        self.device = device
        if self.best_model is not None:
            self.best_model.forcing_tree.to(device)
        return self

    def fit(self, data, target, batch_size=64, num_workers=None):
        dataloader = DataLoader(
            TensorDataset(data, target),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device == "cuda",
        )
        best_candidates = train_controller(self.self_fex_struct, dataloader, self.controller_config, self.fex_config, checkpoint_dir=None, num_workers=num_workers)

        self.fex_config.num_epochs = self.num_finetune_epochs
        self.fex_config.bfgs_epochs = 0
        self.fex_config.lr = self.finetune_lr
        self.fex_config.inter_lr = self.finetune_lr
        for candidate in best_candidates:
            self_fex = candidate.tree
            loss = train_fex(self_fex, dataloader, self.fex_config, self.device, verbose=True)
            updated_reward = 1/(1 + loss)
            candidate.reward = updated_reward
        self.best_model = max(best_candidates, key=lambda c: c.reward)

    def predict(self, x):
        x = x.to(self.device)
        fex = self.best_model.tree.to(self.device)
        fex.eval()
        with torch.no_grad():
            pred = fex(x)
        return pred
    
    def __str__(self):
        return f"FEX = {self.best_model.tree}"