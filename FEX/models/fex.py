from .learnable_tree import FEX
from .controllers import Controller
from ..training.train_controller import train_network_controller, train_controller
from ..training.train_fex import train_network_fex, train_fex 
from ..training.train_configs import FEXConfig, ControllerConfig


import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class CoupledFEX():
    def __init__(self, self_fex_struct, inter_fex_struct, target_dim, *, 
            controller_lr=0.003, controller_epochs=100, cands_per_cycle=10, controller_threshold=0.4, poolsize=10, epsilon_greedy=0.2,
            num_fex_epochs=60, bfgs_epochs=20, self_lr=0.2, inter_lr=0.2, bfgs_lr=0.8, finetune_epochs=1000,
            device='cpu'
        ):
        self.self_fex_struct = self_fex_struct
        self.inter_fex_struct = inter_fex_struct
        self.target_dim = target_dim
        self.finetune_epochs = finetune_epochs
        
        self.fex_config = FEXConfig(
            lr=self_lr,
            inter_lr=inter_lr,
            num_epochs=num_fex_epochs,
            bfgs_epochs=bfgs_epochs,
            bfgs_lr=bfgs_lr,
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



    def fit(self, data, target, adjacency, batch_size=64, num_workers=2):
        dataloader = DataLoader(
            TensorDataset(data, target),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.device == "cuda",
        )
        best_candidates = train_network_controller(self.self_fex_struct, self.inter_fex_struct, dataloader, adjacency, self.fex_config, self.controller_config, num_workers=num_workers)

        for candidate in best_candidates:
            self_fex = candidate.forcing_tree
            inter_fex = candidate.inter_fex
            self.fex_config["num_epochs"] = self.finetune_epochs
            loss = train_network_fex(self_fex, inter_fex, dataloader, adjacency, self.fex_config, num_workers=num_workers)
            candidate.reward = 1 / (1 + loss)
        self.best_model = max(best_candidates, key=lambda c: c.reward)

    
    def predict(self, data, adjacency, batch_size=64):
        self_fex = self.best_model.forcing_tree
        inter_fex = self.best_model.inter_fex
        self_fex.eval()
        inter_fex.eval()
        dataloader = DataLoader(
            TensorDataset(data),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device == "cuda",
        )
        predictions = []
        for batch in dataloader:
            x = batch[0].to(self.device)
            with torch.no_grad():
                #TODO: fix inter_fex inputs
                raise NotImplementedError()
                pred = self_fex(x) + inter_fex(x, adjacency.to(self.device))
            predictions.append(pred)
        return torch.cat(predictions, dim=0)
    
    def evaluate(self, data, target, adjacency, batch_size=64):
        predictions = self.predict(data, adjacency, batch_size=batch_size)
        loss = F.mse_loss(predictions, target.to(self.device))
        return loss.item()
    
    def __str__(self):
        return f"FEX(self_fex_struct={self.best_model.forcing_tree} \ninter_fex_struct={self.best_model.inter_fex})"
    
from FEX.utils.tree_configs import *
class SingleFEX():
    def __init__(self, self_fex_struct: str, target_dim, *,
            controller_lr=0.03, controller_epochs=100, cands_per_cycle=10, controller_threshold=0.4, poolsize=10, epsilon_greedy=0.2,
            num_fex_epochs=60, self_lr=0.2, bfgs_epochs=10, bfgs_lr=0.4, 
            num_finetune_epochs=1000, finetune_lr=0.02,
            device='cpu',
        ):
        self.self_fex_struct = get_tree_config(self_fex_struct)
        self.target_dim = target_dim
        self.fex_config = FEXConfig(
            num_epochs=num_fex_epochs,
            lr=self_lr,
            bfgs_epochs=bfgs_epochs,
            bfgs_lr=bfgs_lr,
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
        self.fex_config.lr = self.finetune_lr
        self.fex_config.inter_lr = self.finetune_lr
        for candidate in best_candidates:
            self_fex = candidate.tree
            loss = train_fex(self_fex, dataloader, self.fex_config, self.device)
            updated_reward = 1/(1 + loss)
            candidate.reward = updated_reward
        self.best_model = max(best_candidates, key=lambda c: c.reward)

    def predict(self, data, batch_size=64):
        dataloader = DataLoader(
            TensorDataset(data),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device == "cuda",
        )
        predictions = []
        for batch in dataloader:
            x = batch[0].to(self.device)
            with torch.no_grad():
                pred = self.best_model.forcing_tree(x)
            predictions.append(pred)
        return torch.cat(predictions, dim=0)
    
    def __str__(self):
        return f"FEX = {self.best_model.tree}"