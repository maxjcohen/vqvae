import torch
import pytorch_lightning as pl

from ..dataset.oze import Experiment, parser


class LitExport(pl.LightningModule):
    def __init__(self, model, export_path):
        super().__init__()
        self.model = model
        self.export_path = export_path

    def validation_step(self, batch, batch_idx):
        u, y = batch
        encoding = self.model.encode(y)
        distances = self.model.codebook.compute_distances(encoding)
        indices = distances.argmin(-1)
        return indices

    def validation_epoch_end(self, outputs):
        outputs = torch.cat(outputs, dim=1)
        print(f"Exporting tensor with shape {outputs.shape}.")
        torch.save(outputs, self.export_path)


if __name__ == "__main__":
    parser.add_argument("exportpath", type=str, help="Export path.")
    args = parser.parse_args()

    Experiment.exp_name = "oze-export-indices"
    exp = Experiment(args)

    litmodule = LitExport.load_from_checkpoint(
        "checkpoints/oze-vqvae/current.ckpt",
        model=exp.litmodule.model,
        export_path=args.exportpath,
    )

    # Load dataloader
    exp.datamodule.setup()
    dataloader = exp.datamodule.export_dataloader()

    # Export
    exp.trainer.validate(litmodule, dataloader)
