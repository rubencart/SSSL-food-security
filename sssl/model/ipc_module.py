import logging
import os
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from sssl import utils
from sssl.config import Config
from sssl.data.ipc import IPCBatch, IPCBatchForEval
from sssl.model.backbone_model import CNNOutput
from sssl.model.backbone_module import BackboneModule
from sssl.model.ipc_model import IPCClassifier, IPCLoss, IPCOutput, LossOutput
from sssl.utils import Constants
from torch import Tensor, optim
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau, SequentialLR
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)

logger = logging.getLogger("pytorch_lightning")


class IPCModule(pl.LightningModule):
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
            self.cfg.run_output_dir, "preds.pkl"
        )

        logger.info("Initializing model and criterion")
        if cfg.finetune.pretrained_on == "own":
            logger.info(
                "Initializing SSSL/tile2vec pretrained backbone from %s"
                % cfg.finetune.pretrained_ckpt_path
            )
            self.backbone = BackboneModule.load_from_checkpoint(
                cfg.finetune.pretrained_ckpt_path, cfg=cfg
            )
        else:
            assert cfg.finetune.pretrained_on in ("ImageNet", "random", "")
            logger.info("Initializing other backbone: %s" % cfg.finetune.pretrained_on)
            self.backbone = BackboneModule(cfg)

        if cfg.finetune.freeze_backbone:
            self.backbone.freeze()

        self.classifier = IPCClassifier.build_classifier(cfg)

        self.train_criterion = IPCLoss(cfg)
        self.eval_criterion = self.train_criterion

    def configure_optimizers(self):
        def is_backbone(n):
            return "backbone" in n

        params_and_lrs = [
            (
                [p for n, p in self.named_parameters() if not is_backbone(n)],
                self.cfg.finetune.lr,
            )
        ]
        if not self.cfg.finetune.freeze_backbone:
            params_and_lrs += [
                (
                    [p for n, p in self.named_parameters() if is_backbone(n)],
                    self.cfg.finetune.backbone_lr,
                )
            ]

        optimizer_class = (
            optim.Adam if self.cfg.finetune.optimizer == "adam" else optim.SGD
        )
        optimizer = optimizer_class(
            [
                {
                    "params": params,
                    "lr": lr,
                    **(
                        {
                            "betas": (
                                self.cfg.finetune.adam_beta_1,
                                self.cfg.finetune.adam_beta_2,
                            ),
                            "eps": self.cfg.finetune.adam_eps,
                            "weight_decay": self.cfg.finetune.weight_decay,
                        }
                        if self.cfg.finetune.optimizer == "adamw"
                        else {
                            "weight_decay": 0.0,
                            "momentum": 0.9,
                            "nesterov": True,
                        }
                    ),
                }
                for (params, lr) in params_and_lrs
            ],
        )
        logger.info(optimizer)

        schedulers = []
        if self.cfg.finetune.lr_schedule is not None:
            logger.info("Using LR scheduler: %s" % self.cfg.finetune.lr_schedule)
            if self.cfg.finetune.lr_schedule == "linear_with_warmup":
                scheduler1 = LinearLR(
                    optimizer,
                    start_factor=1 / self.cfg.finetune.lr_schedule_warmup_epochs,
                    end_factor=1.0,
                    total_iters=self.cfg.finetune.lr_schedule_warmup_epochs - 1,
                )
                scheduler2 = LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=1e-8,
                    total_iters=self.cfg.train.max_epochs
                    - self.cfg.finetune.lr_schedule_warmup_epochs,
                )
                scheduler = SequentialLR(
                    optimizer,
                    [scheduler1, scheduler2],
                    milestones=[self.cfg.finetune.lr_schedule_warmup_epochs - 1],
                )
                schedulers.append(scheduler)
            elif self.cfg.finetune.lr_schedule == "reduce_on_plateau":
                # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
                val_every_n_steps = self.cfg.finetune.val_every_n_steps
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": ReduceLROnPlateau(
                            optimizer,
                            mode=self.cfg.finetune.lr_schedule_mode,
                            factor=self.cfg.finetune.lr_schedule_factor,
                            patience=self.cfg.finetune.lr_schedule_patience,
                            verbose=True,
                        ),
                        "monitor": self.cfg.finetune.lr_schedule_monitor,
                        "interval": "step" if val_every_n_steps > 1 else "epoch",
                        "frequency": val_every_n_steps if val_every_n_steps > 1 else 1,
                    },
                }

        return [optimizer], schedulers

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, IPCBatch):
            # move all tensors in your custom data structure to the device
            batch.tiles = batch.tiles.to(device)
            batch.ipcs = batch.ipcs.to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def on_train_epoch_start(self) -> None:
        if self.cfg.debug and self.cfg.pretrain.lr_schedule not in (
            None,
            "reduce_on_plateau",
        ):
            logger.info(f"Current LR: {self.lr_schedulers().get_last_lr()}")

    def forward(self, batch: IPCBatch):
        cnn_out: CNNOutput = self.backbone(batch)
        ipc_out: IPCOutput = self.classifier(cnn_out)
        return {
            "cnn_out": cnn_out,
            "ipc_out": ipc_out,
            "batch": batch,
        }

    def training_step(self, batch: IPCBatch, batch_idx: int):
        cnn_out: CNNOutput = self.backbone(batch)
        ipc_out: IPCOutput = self.classifier(cnn_out)
        loss_output: LossOutput = self.train_criterion(ipc_out, batch)
        log_dict = loss_output.to_dict()

        # exclude aux loss logs
        excludes = ["logits"]
        self.log_dict(
            {
                k: v
                for k, v in log_dict.items()
                if not any([k.startswith(excl) for excl in excludes])
            },
            # on_step=True, on_epoch=True, logger=True, prog_bar=False,
            batch_size=batch.bs,
        )
        return log_dict

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.valtest_step(batch, "test")

    def validation_step(
        self, batch: IPCBatch, batch_idx: int
    ) -> Dict[str, Union[IPCBatch, IPCOutput, LossOutput]]:
        return self.valtest_step(batch, "val")

    def valtest_step(
        self, batch: IPCBatch, stage: str
    ) -> Dict[str, Union[IPCBatch, IPCOutput, LossOutput]]:
        cnn_out: CNNOutput = self.backbone(batch)
        ipc_out: IPCOutput = self.classifier(cnn_out)
        loss_output: LossOutput = self.eval_criterion(ipc_out, batch)

        excludes = ["logits"]
        self.log_dict(
            {
                f"{stage}_{k}": v
                for k, v in loss_output.to_dict().items()
                if not any([k.startswith(excl) for excl in excludes])
            },
            batch_size=batch.bs,
        )
        return {
            "ipc_out": ipc_out.detach(),
            "loss_out": loss_output.detach(),
            "batch": IPCBatchForEval(batch),
        }

    def test_epoch_end(self, outputs):
        # test, ood, val = outputs
        all_results = {}
        dsets = (
            ("id", "ood", "val")
            if not self.cfg.save_train_predictions
            else ("id", "ood", "val", "train")
        )
        for i, prefix in enumerate(dsets):
            rdict = self.compute_metrics(outputs[i])
            results = {f"test_{prefix}_{k}": v for k, v in rdict["result_dict"].items()}
            self.statistics.update(results)
            self.log_dict(results)
            all_results[prefix] = rdict

        self.predictions = all_results
        if self.save_predictions_to_file:
            logger.info("Saving test preds to %s" % self.save_predictions_location)
            with open(self.save_predictions_location, "wb") as f:
                torch.save(all_results, f)

        return super().test_epoch_end(outputs)

    def validation_epoch_end(
        self,
        list_of_step_outputs: List[Dict[str, Union[IPCBatch, IPCOutput, LossOutput]]],
    ):

        ps = [p.view(-1) for p in self.parameters()]
        self.log("param_norm", torch.cat(ps).detach().norm())

        rdict = self.compute_metrics(list_of_step_outputs)
        self.log_dict({f"val_{k}": v for k, v in rdict["result_dict"].items()})

        return super().validation_epoch_end(list_of_step_outputs)

    def compute_metrics(
        self, output_list: List[Dict[str, Union[IPCBatch, IPCOutput, LossOutput]]]
    ) -> Dict[str, Tensor]:
        zone_date_2_ipc = {}
        zone_date_2_logits = {}
        tile_ipcs, tile_logits, tile_zones, tile_dates = [], [], [], []
        for d in output_list:
            batch, l_output = d["batch"], d["loss_out"]
            tile_ipcs.append(batch.ipcs.cpu())
            tile_zones.append(batch.zones.cpu())
            tile_dates.append(batch.dates.cpu())
            tile_logits.append(l_output.logits.cpu())
            for (zi, di, ipc, logits) in zip(
                batch.zones, batch.dates, batch.ipcs, l_output.logits
            ):
                zone_date_2_ipc[(zi.item(), di.item())] = ipc.item()
                zone_date_2_logits.setdefault((zi.item(), di.item()), []).append(
                    logits.cpu()
                )

        zone_dates = list(zone_date_2_ipc.keys())  # order fixed
        zone_date_2_maj_vote = {
            # max selects greatest logit per tile prediction, mode selects most frequently occurring prediction
            #  across tiles
            zd: torch.stack(logits).argmax(dim=-1).mode().values.item()
            for (zd, logits) in zone_date_2_logits.items()
        }
        zone_date_2_max_vote = {
            zd: torch.stack(logits).argmax(dim=-1).max().item()
            for (zd, logits) in zone_date_2_logits.items()
        }
        maj_votes = torch.tensor([zone_date_2_maj_vote[zd] for zd in zone_dates])
        max_votes = torch.tensor([zone_date_2_max_vote[zd] for zd in zone_dates])
        zd_ipcs = torch.tensor([zone_date_2_ipc[zd] for zd in zone_dates]).to(maj_votes)
        tile_ipcs = torch.cat(tile_ipcs).cpu()
        tile_zones = torch.cat(tile_zones).cpu()
        tile_dates = torch.cat(tile_dates).cpu()
        tile_logits = torch.cat(tile_logits).cpu()
        tile_preds = tile_logits.argmax(dim=-1)
        ipc_bins = (
            torch.bincount(tile_preds, minlength=utils.Constants.NB_IPC_SCORES).cpu()
            / tile_preds.shape[0]
        )

        result_dict = {"pretrained_epoch": self.cfg.finetune.pretrained_epoch}
        for i, b in enumerate(ipc_bins):
            result_dict[f"percent_score_{i}"] = b

        for avg in ("weighted", "micro", "macro"):
            for (name, preds, tgt) in (
                ("maj_vote", maj_votes, zd_ipcs),
                ("max_vote", max_votes, zd_ipcs),
                ("tile", tile_preds, tile_ipcs),
            ):
                # https://openknowledge.worldbank.org/handle/10986/34510 page 4
                maj_class, bin_maj_class = utils.get_maj_class(tgt, return_binary=True)
                maj_class, bin_maj_class = torch.full_like(
                    tgt, maj_class
                ), torch.full_like(tgt, bin_maj_class)
                num_classes = (
                    Constants.NB_IPC_SCORES if not self.cfg.finetune.binarize_ipc else 2
                )

                # remap in case some labels don't occur in preds/tgt
                pred_bins = torch.bincount(preds, minlength=4).cpu() / preds.shape[0]
                gt_bins = torch.bincount(tgt, minlength=4).cpu() / tgt.shape[0]
                cum_tags = (gt_bins + pred_bins).gt(0).long().cumsum(-1) - 1
                mapped_nc = int(cum_tags.max()) + 1
                mapped_preds = torch.tensor([cum_tags[v] for v in preds])
                mapped_tgt = torch.tensor([cum_tags[v] for v in tgt])

                result_dict.update(
                    {
                        f"{name}_f1_{avg}": multiclass_f1_score(
                            preds, tgt, num_classes=num_classes, average=avg
                        ).cpu(),
                        f"{name}_acc_{avg}": multiclass_accuracy(
                            preds, tgt, num_classes=num_classes, average=avg
                        ).cpu(),
                        f"{name}_f1_{avg}_mapped": multiclass_f1_score(
                            mapped_preds,
                            mapped_tgt,
                            num_classes=mapped_nc,
                            average=avg,
                        ).cpu()
                        if mapped_nc > 1
                        else 1.0,
                        f"{name}_acc_{avg}_mapped": multiclass_accuracy(
                            mapped_preds,
                            mapped_tgt,
                            num_classes=mapped_nc,
                            average=avg,
                        ).cpu()
                        if mapped_nc > 1
                        else 1.0,
                        f"maj_baseline_{name}_f1_{avg}": multiclass_f1_score(
                            maj_class, tgt, num_classes=num_classes, average=avg
                        ).cpu(),
                        f"maj_baseline_{name}_acc_{avg}": multiclass_accuracy(
                            maj_class, tgt, num_classes=num_classes, average=avg
                        ).cpu(),
                    }
                )
        preds = {
            "zone_date_2_ipc": zone_date_2_ipc,
            "zone_date_2_logits": zone_date_2_logits,
            "tile_ipcs": tile_ipcs,
            "tile_zones": tile_zones,
            "tile_dates": tile_dates,
            "tile_logits": tile_logits,
            "zone_dates": zone_dates,
            "zone_date_2_maj_vote": zone_date_2_maj_vote,
            "zone_date_2_max_vote": zone_date_2_max_vote,
            "result_dict": result_dict,
        }
        return preds
