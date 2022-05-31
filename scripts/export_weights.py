import argparse
from pathlib import Path

import torch

from src.trainer.cifar import LitCifarTrainer


parser = argparse.ArgumentParser(
    description="Export weights from Lightning checkpoint."
)
parser.add_argument("checkpoint", type=Path, help="Pytorch Lightning checkpoint path.")
parser.add_argument("save_path", type=Path, help="Path to save the model's state dict.")
args = parser.parse_args()

litmodule = LitCifarTrainer.load_from_checkpoint(args.checkpoint)
torch.save(litmodule.model.state_dict(), args.save_path)
