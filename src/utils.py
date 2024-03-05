import numpy as np
import random
import torch


def set_seed(seed=1234):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
