"""Graph maximin groups.

Adapted from the 1-maximin loop in GP-inverse
(src/utils/sparse_cholesky/kolesky/ordering.py). That routine picks the point
farthest from the points already chosen. Here the distance is hop distance on
the interaction graph, and each node is then assigned to the nearest chosen
center with group sizes matched to the random partition in group_loss.
"""

from __future__ import annotations

from collections import deque

import torch


def _balanced_sizes(num_nodes: int, num_groups: int) -> list[int]:
    """Group sizes produced by ``i * num_groups // num_nodes``."""
    cuts = [((k * num_nodes + num_groups - 1) // num_groups) for k in range(num_groups + 1)]
    return [cuts[k + 1] - cuts[k] for k in range(num_groups)]


def _undirected_neighbors(num_nodes: int, left: torch.Tensor, right: torch.Tensor) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    seen = [set() for _ in range(num_nodes)]
    for a, b in zip(left.detach().cpu().tolist(), right.detach().cpu().tolist()):
        a = int(a)
        b = int(b)
        if a == b or a < 0 or b < 0 or a >= num_nodes or b >= num_nodes:
            continue
        if b not in seen[a]:
            seen[a].add(b)
            neighbors[a].append(b)
        if a not in seen[b]:
            seen[b].add(a)
            neighbors[b].append(a)
    return neighbors


def _hop_distance(center: int, neighbors: list[list[int]], num_nodes: int) -> list[int]:
    unreachable = num_nodes
    distance = [unreachable] * num_nodes
    distance[center] = 0
    queue: deque[int] = deque([center])
    while queue:
        node = queue.popleft()
        next_distance = distance[node] + 1
        for other in neighbors[node]:
            if next_distance < distance[other]:
                distance[other] = next_distance
                queue.append(other)
    return distance


def _maximin_centers(neighbors: list[list[int]], num_nodes: int, num_groups: int) -> list[int]:
    """Greedy farthest-point centers. The first center is node 0, matching the
    all-infinite start of the Euclidean maximin heap."""
    unreachable = num_nodes
    nearest = [unreachable] * num_nodes
    centers: list[int] = []
    for _ in range(num_groups):
        if not centers:
            center = 0
        else:
            center = max(range(num_nodes), key=lambda node: (nearest[node], -node))
        centers.append(center)
        nearest[center] = 0
        queue: deque[int] = deque([center])
        while queue:
            node = queue.popleft()
            next_distance = nearest[node] + 1
            if next_distance >= unreachable:
                continue
            for other in neighbors[node]:
                if next_distance < nearest[other]:
                    nearest[other] = next_distance
                    queue.append(other)
    return centers


def maximin_group_ids(
    num_nodes: int,
    edge_left: torch.Tensor,
    edge_right: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Assign every node to a maximin center, with the same sizes as a random partition.

    ``edge_left`` and ``edge_right`` are the two endpoints of each interaction.
    Distance treats those edges as undirected. Groups are filled closest-first
    until each group reaches the random-partition size, so a later comparison
    uses the same number of nodes per group.
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_groups > num_nodes:
        raise ValueError("num_groups cannot exceed num_nodes")

    neighbors = _undirected_neighbors(num_nodes, edge_left, edge_right)
    centers = _maximin_centers(neighbors, num_nodes, num_groups)
    distances = [_hop_distance(center, neighbors, num_nodes) for center in centers]
    capacity = _balanced_sizes(num_nodes, num_groups)

    group_id = [-1] * num_nodes
    for group, center in enumerate(centers):
        group_id[center] = group
        capacity[group] -= 1

    candidates: list[tuple[int, int, int]] = []
    for node in range(num_nodes):
        if group_id[node] != -1:
            continue
        for group in range(num_groups):
            candidates.append((distances[group][node], group, node))
    candidates.sort()

    for _, group, node in candidates:
        if group_id[node] != -1 or capacity[group] <= 0:
            continue
        group_id[node] = group
        capacity[group] -= 1

    if any(group < 0 for group in group_id):
        raise RuntimeError("maximin assignment left a node without a group")
    return torch.tensor(group_id, dtype=torch.long)
