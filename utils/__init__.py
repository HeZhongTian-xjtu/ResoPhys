import argparse
import random
import numpy as np
import torch

from .logger import setup_logger
from .dist import *


# Wrap tqdm.write so it's transparent to users; users can still use print()
# Reference: https://zhuanlan.zhihu.com/p/450780357
# Example implementation (commented out):
# import contextlib
# class DummyFile:
#     def __init__(self, file):
#         if file is None:
#             file = sys.stderr
#         self.file = file
#
#     def write(self, x):
#         if len(x.rstrip()) > 0:
#             tqdm.write(x, file=self.file)
#
# @contextlib.contextmanager
# def redirect_stdout(file=None):
#     if file is None:
#         file = sys.stderr
#     old_stdout = file
#     sys.stdout = DummyFile(file)
#     yield
#     sys.stdout = old_stdout

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.enabled = False
    # print("only init seed, not deterministic")
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# rank = torch.distributed.get_rank()
# Problem solved!
# init_seeds(1 + rank)

def uprint(*params, **kwargs):
    if get_rank() == 0:
        # tqdm.write(*params, **kwargs)
        print(*params, **kwargs)
# def uprint(out_str, color='green'):
#     if get_rank() == 0:
#         print(out_str)


import secrets
import string

# wandb
def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))

def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)