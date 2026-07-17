import numpy as np
from typing import List, Tuple, Dict


def create_module_rngs(root_seed: int, module_names: List[str]) -> Tuple[Dict[str, np.random.Generator], Dict[str, int]]:
    root = np.random.default_rng(int(root_seed))
    seeds: Dict[str, int] = {}
    rngs: Dict[str, np.random.Generator] = {}

    # Use a stable maximum for integer generation (63 bits to be safe)
    max_int = 2**63 - 1

    for name in module_names:
        s = int(root.integers(0, max_int, endpoint=False))
        seeds[name] = s
        rngs[name] = np.random.default_rng(s)

    return rngs, seeds