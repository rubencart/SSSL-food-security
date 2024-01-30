import json
import logging
import os
import pprint
import re
import time
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import wandb
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
from sssl.data.ipc import build_dataloaders
from sssl.model.ipc_module import IPCModule
from sssl.utils import generate_output_dir_name, initialize_logging, mkdir_p

os.environ["TORCH_HOME"] = "../.torch"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
logger = logging.getLogger("pytorch_lightning")


def main(cfg: Config, train_ns_cfg: ConfigNs):
    # Make sure to set the random seed before the instantiation of a Trainer() so
    # that each model initializes with the same weights.
    # https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html#distributed-data-parallel
    seed_everything(cfg.seed)
    if not cfg.do_downstream:
        raise ValueError()

    run_output_dir = generate_output_dir_name(cfg)
    mkdir_p(run_output_dir)

    initialize_logging(run_output_dir, to_file=True)
    cfg.run_output_dir = run_output_dir

    logger.info(pprint.pformat(cfg.to_dict(), indent=2))
    with open(os.path.join(run_output_dir, "cfg.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    best_ckpt = (
        cfg.finetune.pretrained_ckpt_path
        if cfg.finetune.pretrained_on == "own"
        else cfg.finetune.pretrained_on
    )
    if cfg.finetune.all_backbone_ckpts_in_dir:
        ckpts = get_ckpt_filenames(cfg.finetune.all_backbone_ckpts_in_dir)

        best_score = (
            0
            if cfg.finetune.pt_checkpoint_monitor_min_or_max == "max"
            else float("inf")
        )
        best_path = ""
        # >= to make sure path gets overwritten even when all runs score 0
        is_better = (
            lambda s, b: (s >= b)
            if cfg.finetune.pt_checkpoint_monitor_min_or_max == "max"
            else (s <= b)
        )

        for ckpt in ckpts:
            cfg = configure_for_ckpt(cfg, ckpt, run_output_dir)
            score, path = run_trainer(cfg, train_ns_cfg)
            if is_better(score, best_score):
                logger.info(
                    "Improved previous best pretraining ckpt %s (score %s), new: %s (score %s)"
                    % (best_ckpt, best_score, ckpt, score)
                )
                best_score, best_path, best_ckpt = (
                    score,
                    path,
                    cfg.finetune.pretrained_ckpt_path,
                )
            else:
                logger.info(
                    "Did NOT improve previous best pretraining ckpt %s (score %s), new: %s (score %s)"
                    % (best_ckpt, best_score, ckpt, score)
                )

    else:
        log_run_begin(cfg)
        best_score, best_path = run_trainer(cfg, train_ns_cfg)

    return best_score, best_path, best_ckpt


def log_run_begin(cfg):
    if wandb.run is not None:
        wandb.finish()
        time.sleep(10)
    logger.info("################################")
    logger.info(
        "Running IPC %s on %s%% of training data with output_dir %s, mode %s, path %s"
        % (
            "training" if cfg.do_train else "eval",
            cfg.finetune.percentage_of_training_data,
            cfg.run_output_dir,
            cfg.finetune.pretrained_on if cfg.finetune.pretrained else "random",
            cfg.finetune.pretrained_ckpt_path
            if cfg.finetune.pretrained_on == "own"
            else "/",
        )
    )
    logger.info("################################")


def configure_for_ckpt(cfg, ckpt, run_output_dir):
    cfg.finetune.pretrained_on = "own"
    m = re.search(r"epoch=([0-9]+)-", ckpt)
    cfg.finetune.pretrained_epoch = int(m.group(1) if m else -1)
    cfg.finetune.pretrained_ckpt_path = os.path.join(
        cfg.finetune.all_backbone_ckpts_in_dir, ckpt
    )
    cfg.run_output_dir = os.path.join(run_output_dir, ckpt.replace(".ckpt", ""))
    mkdir_p(cfg.run_output_dir)
    log_run_begin(cfg)
    return cfg


def run_trainer(cfg, train_ns_cfg):
    dataloader_dict = build_dataloaders(cfg)
    model = make_model(cfg)

    trainer_kwargs, model_checkpoint = build_trainer_kwargs(cfg, cfg.run_output_dir)
    trainer = Trainer.from_argparse_args(
        train_ns_cfg,
        **trainer_kwargs,
    )
    ckpt_path = None
    if cfg.continue_training_from_checkpoint:  # or cfg.load_weights_from_checkpoint:
        ckpt_path = cfg.checkpoint

    if cfg.do_test:
        model.save_predictions_to_file = True
        model.save_predictions_location = os.path.join(
            cfg.run_output_dir, "test_preds.pkl"
        )
        trainer.test(
            model,
            dataloaders=[
                dataloader_dict["test"],
                dataloader_dict["ood"],
                dataloader_dict["val"],
            ],
            ckpt_path=ckpt_path,
        )
        return model.statistics[cfg.finetune.pt_checkpoint_monitor], None
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

        model.save_predictions_to_file = True
        model.save_predictions_location = os.path.join(
            cfg.run_output_dir, "test_preds.pkl"
        )
        trainer.test(
            dataloaders=[
                dataloader_dict["test"],
                dataloader_dict["ood"],
                dataloader_dict["val"],
            ],
            ckpt_path="best",
        )

        return (
            model.statistics[cfg.finetune.pt_checkpoint_monitor],
            model_checkpoint.best_model_path,
        )


def get_ckpt_filenames(dir_name):
    ckpts = []
    _, _, files = next(os.walk(dir_name))
    for file in files:
        if file != "last.ckpt" and file.endswith(".ckpt"):
            ckpts.append(file)
    return ckpts


def run_name(path: str, cfg: Config) -> str:
    path, out_dir = os.path.split(path)
    if cfg.finetune.pretrained_on == "own" and cfg.finetune.all_backbone_ckpts_in_dir:
        _, run_out_dir = os.path.split(path)
        return f"{run_out_dir}_{out_dir}"
    else:
        # pre_path = .../output/ , out_dir = 2022_12_08_...
        return out_dir


def build_trainer_kwargs(
    cfg: Config, run_output_dir: str
) -> Tuple[Dict, ModelCheckpoint]:
    tags = [
        str(cfg.seed),
        cfg.finetune.pretrained_on,
        "bin" if cfg.finetune.binarize_ipc else "mult",
        "%%%s" % cfg.finetune.percentage_of_training_data,
        cfg.landsat8_bands,
    ]

    kwargs = {
        "name": run_name(run_output_dir, cfg),
        "project": cfg.finetune.wandb_project_name,
        "entity": cfg.wandb_entity,
        "save_dir": run_output_dir,
        "offline": cfg.wandb_offline,
        "save_code": True,
        "tags": tags,
        "config": cfg.to_dict(),
    }
    wandb_logger = WandbLogger(**kwargs)

    for metric, summary in (
        ("val_loss", "min"),
        ("val_maj_vote_f1_weighted", "max"),
        ("val_maj_vote_f1_macro", "max"),
        ("val_maj_vote_acc_micro", "max"),
        ("val_maj_vote_acc_macro", "max"),
    ):
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
            monitor=cfg.finetune.model_checkpoint_monitor,
            # monitor='mean_training_loss',
            dirpath=os.path.join(run_output_dir, "checkpoints"),
            filename="{epoch}-{step}",
            save_top_k=1,
            mode=cfg.finetune.model_checkpoint_monitor_min_or_max,
            save_last=True,
        )
        callbacks.append(model_checkpoint)
    if cfg.finetune.early_stop not in [None, ""]:  # todo handle properly
        logger.info("Using early stopping on %s" % cfg.finetune.early_stop)
        early_stopping = EarlyStopping(
            cfg.finetune.early_stop,
            patience=cfg.finetune.early_stop_patience,
            mode=cfg.finetune.early_stop_min_or_max,
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
            "val_check_interval": cfg.finetune.val_every_n_steps,
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
            }
        )
        lr_monitor = LearningRateMonitor(logging_interval=None)
        callbacks.append(lr_monitor)
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
                    "limit_test_batches": 5,
                }
            )
    if not cfg.do_validate_during_training:
        trainer_kwargs.update({"limit_val_batches": 0.0})

    return trainer_kwargs, model_checkpoint


def make_model(cfg: Config) -> pl.LightningModule:
    model = IPCModule(cfg)
    return model


if __name__ == "__main__":
    _cfg, train_ns_cfg = utils.build_configs()
    main(_cfg, train_ns_cfg)
