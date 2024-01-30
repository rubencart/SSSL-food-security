import argparse
import logging
import os
from typing import Any, Dict

import yaml

import finetune
import pretrain
from sssl import utils

logger = logging.getLogger("__main__")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def run(cfg: Dict):
    if cfg.get("pretrained_on", "own") == "own":
        pt_cfg, pt_train_ns = utils.path_to_config(
            path=cfg["pretrain_cfg"], seed=cfg.get("seed", -1)
        )
        if cfg.get("do_pretrain", True):
            logger.info("Running pretraining with config file %s" % cfg["pretrain_cfg"])
            pt_out_dir, best_pt_model_path = pretrain.main(pt_cfg, pt_train_ns)
        else:
            pt_out_dir = cfg.get("pretrained_dir", None)
            best_pt_model_path = cfg.get("best_model_path", None)

    # best gets overwritten if you use multiple configs here
    assert len(cfg.get("downstream_cfg", [])) <= 1 or not cfg.get("best_cfg", None)
    best_pt_ckpt = None
    for cfg_path in cfg["downstream_cfg"]:
        logger.info("Running finetuning with config file %s" % cfg_path)

        downstr_cfg, downstr_train_ns = utils.path_to_config(
            cfg_path, seed=cfg.get("seed", -1)
        )
        if cfg.get("pretrained_on", "own") == "own":
            downstr_cfg.finetune.pretrained_on = "own"
            downstr_cfg, pt_cfg = copy_pt_settings(downstr_cfg, pt_cfg)

            if cfg["checkpoints"] == "best":
                if best_pt_model_path is None:
                    raise ValueError
                downstr_cfg.finetune.pretrained_ckpt_path = best_pt_model_path
                downstr_cfg.finetune.all_backbone_ckpts_in_dir = ""
            if cfg["checkpoints"] == "last":
                downstr_cfg.finetune.pretrained_ckpt_path = os.path.join(
                    pt_out_dir, "checkpoints", "last.ckpt"
                )
                downstr_cfg.finetune.all_backbone_ckpts_in_dir = ""
            if cfg["checkpoints"] == "all":
                downstr_cfg.finetune.pretrained_ckpt_path = ""
                if "checkpoints" not in pt_out_dir:
                    pt_out_dir = os.path.join(pt_out_dir, "checkpoints")
                downstr_cfg.finetune.all_backbone_ckpts_in_dir = pt_out_dir

            downstr_cfg.cfg_name = pt_cfg.cfg_name + "_#_" + downstr_cfg.cfg_name

        best_score, _, best_pt_ckpt = finetune.main(downstr_cfg, downstr_train_ns)
        logger.info(
            "Best pretraining ckpt: %s, with score: %s" % (best_pt_ckpt, best_score)
        )

    if cfg.get("best_cfg", None) and cfg.get("pretrained_on", "own") == "own":
        for cfg_path in cfg["best_cfg"]:
            logger.info(
                "Running finetuning on BEST CHECKPOINT %s with config file %s"
                % (best_pt_ckpt, cfg_path)
            )

            best_cfg, best_train_ns = utils.path_to_config(
                cfg_path, seed=cfg.get("seed", -1)
            )
            best_cfg, pt_cfg = copy_pt_settings(best_cfg, pt_cfg)

            if best_pt_ckpt is None:
                raise ValueError
            best_cfg.finetune.pretrained_ckpt_path = best_pt_ckpt
            best_cfg.finetune.all_backbone_ckpts_in_dir = ""

            best_cfg.cfg_name = pt_cfg.cfg_name + "_#_best_" + best_cfg.cfg_name

            score, finetuned_ckpt, pt_ckpt = finetune.main(best_cfg, best_train_ns)


def copy_pt_settings(downstr_cfg, pt_cfg):
    downstr_cfg.cnn_type = pt_cfg.cnn_type
    downstr_cfg.landsat8_bands = pt_cfg.landsat8_bands
    downstr_cfg.pretrain.loss_type = pt_cfg.pretrain.loss_type
    downstr_cfg.pretrain.augmentations = pt_cfg.pretrain.augmentations
    downstr_cfg.pretrain.time_pair_limit = pt_cfg.pretrain.time_pair_limit
    downstr_cfg.pretrain.space_pair_limit = pt_cfg.pretrain.space_pair_limit
    downstr_cfg.pretrain.space_limit_type = pt_cfg.pretrain.space_limit_type
    return downstr_cfg, pt_cfg


def build_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--cfg", type=str, default="config/pretrain_ipc/sssl_resnet18_t4_s015_ALL.yaml"
    )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--vsc", action="store_true")

    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    if args.seed > -1:
        dict_cfg["seed"] = args.seed

    return dict_cfg


if __name__ == "__main__":

    cfg = build_config()

    utils.initialize_logging(output_dir=None, to_file=False, logger_name="__main__")
    logger.info("Running pretraining and finetuning with settings...")
    logger.info(cfg)

    run(cfg)
