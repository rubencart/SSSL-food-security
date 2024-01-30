import copy
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
from sssl.config import Config
from sssl.data.landsat8 import Batch
from sssl.model.backbone_model import CNNModel, CNNOutput, LossOutput, PretrainLoss
from torch import optim
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR

logger = logging.getLogger("pytorch_lightning")


class BackboneModule(pl.LightningModule):
    """
    https://pytorch-lightning.readthedocs.io/en/latest/advanced/transfer_learning.html
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg.to_dict(), ignore=[])

        self.predictions = {}
        self.statistics = {}
        self.save_predictions_to_file = False
        self.save_predictions_location = os.path.join(
            self.cfg.run_output_dir, "val_preds.pkl"
        )

        logger.info("Initializing model and criterion")
        self.model = CNNModel(cfg)
        self.train_criterion = PretrainLoss(cfg)
        self.eval_criterion = self.train_criterion

    def configure_optimizers(self):

        optimizer_class = (
            optim.Adam if self.cfg.pretrain.optimizer == "adam" else optim.SGD
        )
        optimizer = optimizer_class(
            **{
                "params": self.parameters(),
                "lr": self.cfg.pretrain.lr,
                **(
                    {
                        "betas": (
                            self.cfg.pretrain.adam_beta_1,
                            self.cfg.pretrain.adam_beta_2,
                        ),
                        "eps": self.cfg.pretrain.adam_eps,
                        "weight_decay": self.cfg.pretrain.weight_decay,
                    }
                    if self.cfg.pretrain.optimizer == "adam"
                    else {
                        "weight_decay": 0.0,
                        "momentum": 0.9,
                        "nesterov": True,
                    }
                ),
            }
        )
        logger.info(optimizer)

        schedulers = []
        if self.cfg.pretrain.lr_schedule is not None:
            logger.info("Using LR scheduler: %s" % self.cfg.pretrain.lr_schedule)
            if self.cfg.pretrain.lr_schedule == "linear_with_warmup":
                scheduler1 = LinearLR(
                    optimizer,
                    start_factor=1 / self.cfg.pretrain.lr_schedule_warmup_epochs,
                    end_factor=1.0,
                    total_iters=self.cfg.pretrain.lr_schedule_warmup_epochs - 1,
                )
                scheduler2 = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=1e-8,
                    total_iters=self.cfg.train.max_epochs
                    - self.cfg.pretrain.lr_schedule_warmup_epochs,
                )
                scheduler = SequentialLR(
                    optimizer,
                    [scheduler1, scheduler2],
                    milestones=[self.cfg.pretrain.lr_schedule_warmup_epochs - 1],
                )
                schedulers.append(scheduler)
            elif self.cfg.pretrain.lr_schedule == "reduce_on_plateau":
                # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
                val_every_n_steps = self.cfg.pretrain.val_every_n_steps
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(
                            optimizer,
                            mode=self.cfg.pretrain.lr_schedule_mode,
                            factor=self.cfg.pretrain.lr_schedule_factor,
                            patience=self.cfg.pretrain.lr_schedule_patience,
                            verbose=True,
                        ),
                        "monitor": self.cfg.pretrain.lr_schedule_monitor,
                        "interval": "step" if val_every_n_steps > 1 else "epoch",
                        "frequency": val_every_n_steps if val_every_n_steps > 1 else 1,
                    },
                }

        return [optimizer], schedulers

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, Batch):
            # move all tensors in your custom data structure to the device
            batch.tiles = batch.tiles.to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def on_train_epoch_start(self) -> None:
        if self.cfg.debug and self.cfg.pretrain.lr_schedule not in (
            None,
            "reduce_on_plateau",
        ):
            logger.info(f"Current LR: {self.lr_schedulers().get_last_lr()}")

    def forward(self, batch: Batch) -> CNNOutput:
        output = self.model(batch.tiles)
        return CNNOutput(output)

    def training_step(self, batch: Batch, batch_idx: int):
        ps = torch.cat([p.view(-1) for p in self.parameters()])  # .clone().detach()

        output = CNNOutput(self.model(batch.tiles))
        loss_output = self.train_criterion(output)

        log_dict = loss_output.to_dict()
        # exclude aux loss logs
        excludes = ["scores"]
        self.log_dict(
            {
                k: v
                for k, v in log_dict.items()
                if not any([k.startswith(excl) for excl in excludes])
            },
            # on_step=True, on_epoch=True, logger=True, prog_bar=False,
            batch_size=batch.bs,
        )
        self.log("param_norm", ps.norm().clone().detach().item(), batch_size=batch.bs)

        return log_dict

    def on_after_backward(self):
        if self.trainer.global_step % 1 == 0:
            grad = torch.cat([p.grad.view(-1) for p in self.parameters()])
            grad_norm = grad.norm()
            self.log(
                "grad_norm",
                grad_norm.clone().detach(),
                on_step=True,
                on_epoch=False,
                batch_size=1,
            )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch: Batch, batch_idx: int):
        output: CNNOutput = CNNOutput(self.model(batch.tiles))
        loss_output: LossOutput = self.train_criterion(output)

        log_dict = {
            **loss_output.to_dict(),
            "space_gap": np.mean(copy.deepcopy(batch.sgs)),
            "time_gap": np.mean(copy.deepcopy(batch.tgs)),
        }
        excludes = ["scores"]
        self.log_dict(
            {
                f"val_{k}": v
                for k, v in log_dict.items()
                if not any([k.startswith(excl) for excl in excludes])
            },
            # on_step=True, on_epoch=True, logger=True, prog_bar=False,
            batch_size=batch.bs,
        )
