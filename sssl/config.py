import logging
from types import MethodType
from typing import Any, List, Literal

from tap import Tap

logger = logging.getLogger("pytorch_lightning")


class FinetuneConfig(Tap):
    wandb_project_name: str = "SSSL-IPC"

    pretrained_on: Literal["ImageNet", "own", "random"] = "random"
    pretrained_ckpt_path: str = ""
    pretrained_epoch: int = -1

    freeze_backbone: bool = False
    all_backbone_ckpts_in_dir: str = ""

    batch_size: int = 50
    val_batch_size: int = 50
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    optimizer: str = "adam"
    lr_schedule: Literal["linear_with_warmup", "reduce_on_plateau"] = "reduce_on_plateau"  # or None
    lr_schedule_warmup_epochs: int = 0
    lr_schedule_monitor: str = "val_maj_vote_f1_macro"
    lr_schedule_mode: str = "max"
    lr_schedule_factor: float = 0.1
    lr_schedule_patience: int = 2
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_eps: float = 1e-6
    weight_decay: float = 0.01
    loss: Literal["xent"] = "xent"
    weight_classes: bool = True
    clf_head: Literal["mlp", "linear"] = "mlp"

    early_stop: str = "val_maj_vote_f1_macro"
    early_stop_min_or_max: str = "max"
    early_stop_patience: int = 12
    model_checkpoint_monitor_min_or_max: str = "max"
    # metric to select which finetuning checkpoint is best
    model_checkpoint_monitor: str = "val_maj_vote_f1_macro"
    pt_checkpoint_monitor_min_or_max: str = "max"
    # metric to select which pretrained checkpoint to use
    pt_checkpoint_monitor: str = "test_val_maj_vote_f1_macro"
    val_every_n_steps: int = 0.25  # set to 1.0 for every epoch
    percentage_of_training_data: Literal[100, 70, 50, 20, 5, 1] = 100
    binarize_ipc: bool = False
    n_steps_in_future: int = 0
    temporally_separated: bool = False


class PretrainConfig(Tap):
    wandb_project_name: str = "SSSL"

    lr: float = 1e-4
    optimizer: str = "adam"
    lr_schedule: Literal["linear_with_warmup", "reduce_on_plateau"] = (
        None  # 'reduce_on_plateau'  # or None
    )
    lr_schedule_warmup_epochs: int = 0
    lr_schedule_monitor: str = "val_loss"
    lr_schedule_mode: str = "min"
    lr_schedule_factor: float = 0.1
    lr_schedule_patience: int = 5
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    adam_eps: float = 1e-6
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    focal_loss: bool = True

    early_stop: str = None  # 'val_loss'
    early_stop_min_or_max: str = "min"
    early_stop_patience: int = 10
    model_checkpoint_monitor_min_or_max: str = "min"
    model_checkpoint_monitor: str = "val_loss"

    loss_type: Literal["tile2vec", "sssl"] = "sssl"
    time_pair_limit: int = 12  # in months: 4, 12, 36, 84 (= all)
    space_pair_limit: float = 0.15  # in lat/lon degrees: 0.15, 0.4
    space_limit_type: Literal["degrees", "admin"] = "degrees"
    spatial_pos_shape: Literal["square", "circle"] = "square"
    augmentations: Literal["rel_reasoning", "sssl"] = "sssl"
    K: int = 8
    include_same_loc: bool = True
    batch_size: int = 50
    val_batch_size: int = 50
    val_every_n_steps: int = 1.0  # set to 1.0 for every epoch

    merge_val_test: bool = False
    # so tiles in val set have same neighbors across epochs
    pseudo_random_sampler: bool = True
    # skip tiles that do not have enough neighbors
    skip_valtest_wo_neighbors: bool = True


class Tile2VecConfig(Tap):
    margin: float = 1.0  # 50.0  # see paper
    l2_weight_decay: float = 1e-4


class TrainConfig(Tap):
    accelerator: str = "gpu"
    max_epochs: int = 300
    min_epochs: int = 0
    max_steps: int = -1
    precision: int = 32
    gradient_clip_val: float = 1.0
    gpus: Any = None
    overfit_batches: int = 0


class Config(Tap):
    cfg_name: str = "test"
    wandb_entity: str = ""  # todo

    tiles_dir: str = "/path/to/landsat8/somalia/tiles/"  # todo
    tiles_h5: str = "/path/to/landsat8/h5/tiles_v2.h5"  # todo
    use_h5: bool = True
    use_h5_swmr: bool = False

    output_dir: str = "output/"
    run_output_dir: str = ""

    indices_dir: str = "data/indices/"
    path2zb: str = "tilepath_2_adminzone_box.json"
    zone2box2p: str = "adminzone_2_box_2_tilepath.json"
    box2pz: str = "box_2_tilepath_adminzone.json"
    oor_tiles: str = "out_of_region_tiles.json"
    dicts_path: str = "dicts.json"
    maps_path: str = "maps.json"
    ood_splits_path: str = "ood_splits.json"
    val_splits_path: str = "val_splits.json"
    test_splits_path: str = "test_splits.json"
    train_splits_path: str = "train_splits.json"
    downstr_splits_path: str = "downstr_splits_incl_small.json"
    valtest_wo_neighbors: str = "to_exclude.json"
    fixed_random_order_path: str = "fixed_random_order.json"
    ipc_scores_csv_path: str = "data/predicting_food_crises_data_somalia_from2013-05-01.csv"
    future_ipc_shp: List[str] = [
        "data/SO_202006/SO_202006_CS.shp",
        "data/SO_202010/SO_202010_CS.shp",
        "data/SO_202102/SO_202102_CS.shp",  # don't use, all test data has same ipc score
    ]
    path_to_h5_idx: str = "path_to_h5_idx.json"
    path_to_h5_virtual_idx: str = "path_to_virtual_h5_idx.json"

    landsat8_bands: Literal["RGB", "ALL"] = "ALL"
    landsat8_replace_nan: bool = True
    landsat8_normalize: bool = True

    cnn_type: Literal["conv4", "resnet18", "resnet34"] = "resnet18"
    conv4_feature_size: int = 64
    feature_size: int = 1000

    do_train: bool = True
    do_test: bool = False
    do_val: bool = False
    do_validate_during_training: bool = True
    save_train_predictions: bool = False

    do_pretrain: bool = True
    do_downstream: bool = False
    downstream_task: Literal["IPC"] = "IPC"

    cuda: bool = True
    pin_memory: bool = True
    num_workers: int = 10
    debug_num_workers: int = 0
    debug_max_epochs: int = 1
    num_sanity_val_steps: int = 2
    debug: bool = False
    wandb_offline: bool = False
    deterministic: bool = False
    profiler: str = None
    persistent_workers: bool = False

    continue_training_from_checkpoint: bool = False
    checkpoint: str = ""

    seed: int = 42

    def __init__(self):
        super().__init__()

    def configure(self) -> None:
        self.add_argument("--train", type=lambda x: TrainConfig, required=False)
        self.add_argument("--pretrain", type=lambda x: PretrainConfig, required=False)
        self.add_argument("--finetune", type=lambda x: FinetuneConfig, required=False)
        self.add_argument("--tile2vec", type=lambda x: Tile2VecConfig, required=False)

    def process_args(self, process=True) -> None:
        train_cfg = TrainConfig()
        train_cfg.from_dict(self.train if hasattr(self, "train") else {})
        self.train: TrainConfig = train_cfg

        pretrain_cfg = PretrainConfig()
        pretrain_cfg.from_dict(self.pretrain if hasattr(self, "pretrain") else {})
        self.pretrain: PretrainConfig = pretrain_cfg

        finetune_cfg = FinetuneConfig()
        finetune_cfg.from_dict(self.finetune if hasattr(self, "finetune") else {})
        self.finetune: FinetuneConfig = finetune_cfg

        tile2vec_cfg = Tile2VecConfig()
        tile2vec_cfg.from_dict(self.tile2vec if hasattr(self, "tile2vec") else {})
        self.tile2vec: Tile2VecConfig = tile2vec_cfg

        if process:
            self.process()

    def process(self):
        self.feature_size = 1000 if "resnet" in self.cnn_type else self.conv4_feature_size

        if self.pretrain.loss_type == "sssl" and self.pretrain.augmentations == "rel_reasoning":
            logger.info("Disabling pin_memory for rel_reasoning augmentations...")
            self.pin_memory = False

    def __getstate__(self):
        return self.to_dict()

    def to_dict(self):
        dct = self.as_dict()
        dct.update({"train": self.train.as_dict()})
        dct.update({"finetune": self.finetune.as_dict()})
        dct.update({"tile2vec": self.tile2vec.as_dict()})
        dct.update({"pretrain": self.pretrain.as_dict()})
        return {
            k: v if not isinstance(v, List) else str(v)
            for (k, v) in dct.items()
            if not isinstance(v, MethodType)
        }

    def __str__(self):
        return str(self.to_dict())


class ConfigNs(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
