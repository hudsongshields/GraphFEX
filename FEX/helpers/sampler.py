from torch.distributions import Categorical
import torch

def epsilon_greedy_sample(pmfs: list[torch.Tensor], epsilon: float) -> torch.Tensor:
    batch_size = pmfs[0].shape[0]
    device = pmfs[0].device

    if torch.rand(1, device=device).item() > epsilon:
        samples = []
        for node_pmf in pmfs:
            dist = Categorical(probs=node_pmf)
            node_sample = dist.sample()     # (batch_size,)
            samples.append(node_sample)
        return torch.stack(samples, dim=1)  # (batch_size, num_nodes)

    else:
        samples = []
        for node_pmf in pmfs:
            num_ops = node_pmf.shape[1]
            node_sample = torch.randint(
                low=0,
                high=num_ops,
                size=(batch_size,),
                device=device
            )
            samples.append(node_sample)
        return torch.stack(samples, dim=1) # (batch_size, num_nodes)