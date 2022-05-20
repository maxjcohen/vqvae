import argparse

parser = argparse.ArgumentParser(description="vqvae helper script.")
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Number of epochs for training. Default is 100.",
)
parser.add_argument(
    "--num-workers",
    default=1,
    type=int,
    help="Number of workers for dataloaders. Default is 1.",
)
parser.add_argument(
    "--batch-size",
    default=16,
    type=int,
    help="Batch size for dataloaders. Default is 16.",
)
parser.add_argument("--gpus", default='0', type=str, help="Lightning gpus.")
parser.add_argument("--lr", default=None, type=float, help="Learning rate.")
