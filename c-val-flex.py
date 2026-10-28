import os
from tqdm.auto import trange

import torch

import mytorch as my
import myutils as utils


# Initializer of distributed valiadation process
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument(
        '--optimizer', type=str, default="flex-adamw", nargs="?",
        help='Optimzier name')
    parser.add_argument(
        '--freq', type=str, default="c", nargs="?", help='Skew frequency')
    parser.add_argument(
        '--port', type=int, default=10098, nargs="?", help='Port')
    parser.add_argument(
        '--replicas', type=int, default=0, nargs="?", help='Replicas num')
    args = parser.parse_args()

    if args.replicas <= 0:
        args.replicas = torch.cuda.device_count()
    working_dir = os.getenv('SCRATCHDIR_USER')
    if working_dir is None:
        working_dir = os.getenv("HOME")
    os.environ['TORCH_HOME'] = working_dir
    os.environ['HF_HOME'] = working_dir

    torch.multiprocessing.spawn(
        utils.train.flex_val_process,
        args=(args.replicas, args.port, working_dir, args.optimizer, args.freq),
        nprocs=args.replicas)
