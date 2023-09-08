import hashlib
import json
import random
import warnings

import numpy as np
import torch



def hash_dict(d: dict):
    sorted_dict_str = json.dumps(d, sort_keys=True)
    m = hashlib.sha256()
    m.update(sorted_dict_str.encode('utf-8'))
    return m.hexdigest()



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.cuda.manual_seed_all(seed)
    except AttributeError:
        warnings.info("CUDA not available; failed to seed")

