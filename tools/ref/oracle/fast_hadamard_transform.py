"""Stand-in for Dao-AILab fast_hadamard_transform (used by DeepSeek's rotate_activation): Sylvester
(natural-order) Walsh-Hadamard over the last dim, computed in fp32, returned in the input dtype."""
import functools

import torch


@functools.lru_cache(maxsize=8)
def _hadamard(n: int, device: str) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.size(0) < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.size(0) == n, "power-of-two sizes only"
    return h.to(device)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    H = _hadamard(x.size(-1), str(x.device))
    return ((x.float() @ H) * scale).to(x.dtype)
