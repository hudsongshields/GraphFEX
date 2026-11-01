import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, ops_per_node: list[int], input_size: int, hidden_size: int):
        super().__init__()
        
        self.ops_per_node = ops_per_node


        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, sum(self.ops_per_node))
        )
    
    def forward(self, x: torch.Tensor=torch.zeros(1)):
        logits = self.net(x) # shape: (batch_size, total_num_ops)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        pmf_blocks = torch.split(logits, self.ops_per_node, dim=-1)
        pmfs = [torch.softmax(block, dim=-1) for block in pmf_blocks]

        return pmfs
