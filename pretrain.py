import json
import logging
import os
import pprint
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler

from sssl import utils
from sssl.config import Config, ConfigNs
from sssl.data.landsat8 import build_dataloaders
from sssl.model.backbone_module import BackboneModule
from sssl.utils import generate_output_dir_name, initialize_logging, mkdir_p

os.environ["TORCH_HOME"] = "../.torch"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
logger = logging.getLogger("pytorch_lightning")


def main(cfg: Config, train_ns_cfg: ConfigNs) -> Tuple[str, str]:
    # Make sure to set the random seed before the instantiation of a Trainer() so
    # that each model initializes with the same weights.
    # https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html#distributed-data-parallel
    seed_everything(cfg.seed)
    if not cfg.do_pretrain:
        raise ValueError()

    run_output_dir = generate_output_dir_name(cfg)
    mkdir_p(run_output_dir)

    initialize_logging(run_output_dir, to_file=True)
    cfg.run_output_dir = run_output_dir
    logger.info("#############################################################")
    logger.info("Running pretraining with output dir %s" % run_output_dir)
    logger.info("#############################################################")

    logger.info(pprint.pformat(cfg.to_dict(), indent=2))
    # cfg.save(os.path.join(cfg.run_output_dir, 'cfg.json'), skip_unpicklable=True)
    with open(os.path.join(cfg.run_output_dir, "cfg.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    dataloader_dict = build_dataloaders(cfg)
    model = make_model(cfg)

    trainer_kwargs, model_checkpoint = build_trainer_kwargs(cfg, run_output_dir)
    trainer = Trainer.from_argparse_args(
        train_ns_cfg,
        **trainer_kwargs,
    )
    ckpt_path = None
    if cfg.continue_training_from_checkpoint:  #  or cfg.load_weights_from_checkpoint:
        logger.info("Continue training from checkpoint %s" % cfg.checkpoint)
        ckpt_path = cfg.checkpoint

    if cfg.do_test:
        model.save_predictions_to_file = True
        trainer.test(model, dataloaders=dataloader_dict["val"], ckpt_path=ckpt_path)
    else:
        assert cfg.do_train
        trainer.fit(
            model,
            train_dataloaders=dataloader_dict["train"],
            val_dataloaders=dataloader_dict["val"],
            ckpt_path=ckpt_path,
        )

        if cfg.do_validate_during_training:
            logger.info(
                "Best model: %s, %s"
                % (model_checkpoint.best_model_score, model_checkpoint.best_model_path)
            )

    return run_output_dir, model_checkpoint.best_model_path


def build_trainer_kwargs(
    cfg: Config, run_output_dir: str
) -> Tuple[Dict, ModelCheckpoint]:
    tags = [
        cfg.cnn_type,
        cfg.pretrain.loss_type,
        cfg.pretrain.space_limit_type,
        str(cfg.seed),
        cfg.landsat8_bands,
    ]
    if cfg.pretrain.loss_type == "sssl":
        tags.append(cfg.pretrain.augmentations)

    wandb_logger = WandbLogger(
        name=os.path.split(run_output_dir)[1],
        project=cfg.pretrain.wandb_project_name,
        entity=cfg.wandb_entity,
        save_dir=run_output_dir,
        offline=cfg.wandb_offline,
        save_code=True,
        tags=tags,
        config=cfg.to_dict(),
    )
    for metric, summary in (("val_loss", "min"),):
        logger.info("defining metric as %s: %s" % (summary, metric))
        wandb_logger.experiment.define_metric(metric, summary=summary)
        wandb_logger.experiment.define_metric(metric, summary="last")

    trainer_kwargs = {
        "logger": wandb_logger,
        "deterministic": cfg.deterministic,
    }
    if cfg.profiler is not None:
        if cfg.profiler == "advanced":
            profiler = AdvancedProfiler(
                output_filename=os.path.join(run_output_dir, "profile.txt")
            )
        else:
            assert cfg.profiler == "pytorch"
            profiler = PyTorchProfiler(
                output_filename=os.path.join(run_output_dir, "profile.txt")
            )
        trainer_kwargs["profiler"] = profiler

    callbacks = []
    model_checkpoint = None
    if cfg.do_validate_during_training:
        model_checkpoint = ModelCheckpoint(
            monitor=cfg.pretrain.model_checkpoint_monitor,
            dirpath=os.path.join(run_output_dir, "checkpoints"),
            filename="{epoch}-{step}",
            save_top_k=-1,  # save all checkpoints
            mode=cfg.pretrain.model_checkpoint_monitor_min_or_max,
            save_last=True,
        )
        callbacks.append(model_checkpoint)
    if cfg.pretrain.early_stop not in [None, ""]:
        logger.info("Using early stopping on %s" % cfg.pretrain.early_stop)
        early_stopping = EarlyStopping(
            cfg.pretrain.early_stop,
            patience=cfg.pretrain.early_stop_patience,
            mode=cfg.pretrain.early_stop_min_or_max,
            min_delta=0.0001,
            strict=False,  # so monitoring only when epochs > E todo does this work?
            verbose=True,
        )
        callbacks.append(early_stopping)
    callbacks.append(ModelSummary(max_depth=10))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    trainer_kwargs.update(
        {
            "callbacks": callbacks,
            "check_val_every_n_epoch": 1,
            "val_check_interval": cfg.pretrain.val_every_n_steps,
            "sync_batchnorm": True,
            "num_sanity_val_steps": cfg.num_sanity_val_steps,
        }
    )
    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)
        trainer_kwargs.update(
            {
                "num_sanity_val_steps": 2,
                "min_epochs": 1,
                "max_epochs": cfg.debug_max_epochs,
                "check_val_every_n_epoch": 1,
                "val_check_interval": 1.0,
                "log_every_n_steps": 10,
                "detect_anomaly": True,
            }
        )
        if cfg.train.overfit_batches > 0:
            trainer_kwargs.update(
                {
                    "min_epochs": 2000,
                    "max_epochs": 2000,
                }
            )
        else:
            trainer_kwargs.update(
                {
                    "limit_train_batches": 20,
                    "limit_val_batches": 5,
                }
            )
    if not cfg.do_validate_during_training:
        trainer_kwargs.update({"limit_val_batches": 0.0})

    return trainer_kwargs, model_checkpoint


def make_model(cfg: Config) -> pl.LightningModule:
    model = BackboneModule(cfg)
    return model


if __name__ == "__main__":
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (32000, rlimit[1]))

    cfg, train_ns_cfg = utils.build_configs()
    main(cfg, train_ns_cfg)
