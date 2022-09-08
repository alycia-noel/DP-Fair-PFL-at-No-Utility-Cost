import os
import random
import numpy as np
import torch
import logging

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def metrics(tp, fp, tn, fn):

    acc = (tp + tn) / (tp + tn + fp + fn)

    return acc
