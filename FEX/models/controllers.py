import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, ops_per_node: list[int], num_trees: int, input_size: int, hidden_size: int):
        super().__init__()
        
        self.ops_per_node = ops_per_node
        self.num_trees = num_trees

        self.ops_per_node = ops_per_node * num_trees


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


if __name__ == "__main__":
    from ..utils.sampler import epsilon_greedy_sample
    from ..utils.operations import UNARY_OPS, BINARY_OPS
    NUM_NODES = 5
    """
    Example tree structure

    Node 0: Binary (Node 1, Node 2)
    Node 1: Unary (Node 3)
    Node 2: Unary (Node 4)
    Node 3: Leaf
    Node 4: Leaf
    """
    ops_per_node = [BINARY_OPS, UNARY_OPS, UNARY_OPS]
    CONTROLLER_INPUT = torch.zeros(10)
    controller = Controller(input_size=len(CONTROLLER_INPUT), ops_per_node=ops_per_node, hidden_size=20)
    
    pdf_matrix = controller(CONTROLLER_INPUT)
    print(pdf_matrix)
    op_indices = epsilon_greedy_sample(pdf_matrix, 0.1)
    print(op_indices)
