import numpy as np
import random
import torch


def set_seed(seed=1234):
    """
    set random seed 

    Parameters
    ----------
    seed: int
        Specify the seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    

