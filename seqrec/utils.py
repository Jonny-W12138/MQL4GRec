import math
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hr_at_k(ranks, k: int = 10):
    """
    ranks: list of ranks (1-based) of the true item in the predicted list
    """
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks) if ranks else 0.0


def ndcg_at_k(ranks, k: int = 10):
    """
    NDCG@k for single positive: DCG = 1 / log2(rank+1) if rank<=k else 0
    """
    def dcg(rank):
        if rank is None or rank > k:
            return 0.0
        return 1.0 / math.log2(rank + 1)
    scores = [dcg(r) for r in ranks]
    # IDCG for single positive is 1/log2(2)=1
    return sum(scores) / len(scores) if scores else 0.0


def compute_rank(logits: torch.Tensor, target: torch.Tensor):
    """
    logits: [B, N], target: [B]
    Return list of ranks (1-based)
    """
    # argsort descending
    top = torch.argsort(logits, dim=1, descending=True)
    ranks = []
    for i in range(logits.size(0)):
        order = top[i]
        # position of target in order
        pos = (order == target[i]).nonzero(as_tuple=False)
        if pos.numel() == 0:
            ranks.append(None)
        else:
            ranks.append(int(pos[0, 0].item()) + 1)
    return ranks