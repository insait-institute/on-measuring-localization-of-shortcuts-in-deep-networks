import os
from tqdm.auto import trange

import torch

import mytorch as my
import myutils as utils


# Initializer of distributed training process
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Distributed training job')
    parser.add_argument('samples', type=str, help='Samples name')
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument(
        '--optimizer', type=str, default="adamw", nargs="?",
        help='Optimzier name')
    parser.add_argument(
        '--freq', type=str, default="c", nargs="?", help='Skew frequency')
    parser.add_argument(
        '--epochs', type=int, default=0, nargs="?", help='Epochs')
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

    samples_params =utils.config.get_samples(
        args.samples, f"{working_dir}/datasets")
    samples = samples_params["samples_fn"](**samples_params["samples_params"])

    torch.multiprocessing.spawn(
        utils.train.freeze_train_process,
        args=(
            args.replicas, args.port, working_dir,
            args.samples, args.model, args.optimizer, args.freq, args.epochs,
            samples),
        nprocs=args.replicas)
