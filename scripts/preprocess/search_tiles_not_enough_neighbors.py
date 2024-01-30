import json
import logging

import yaml
from sssl import utils
from sssl.config import Config
from sssl.data.landsat8 import PretrainLandsat8Files, build_dataloaders
from sssl.utils import generate_output_dir_name, initialize_logging, mkdir_p
from tqdm import tqdm

logger = logging.getLogger("pytorch_lightning")


if __name__ == "__main__":
    """
    Use this script to check which val/test tiles don't have enough neighbors for the pretrain loss to be
        correctly computed. Run with `cfg.pretrain.skip_valtest_wo_neighbors = False`.

    Save the tiles to be excluded in to_exclude.json. Run again with `cfg.pretrain.skip_valtest_wo_neighbors = True`
        to verify that when the marked tiles are exluded, all the remaining tiles in the val/test set have
        enough neighbors.
    """

    with open("config/pretrain/debug.yaml", "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Config()
    cfg.from_dict(dict_cfg)
    cfg.process_args()

    run_output_dir = generate_output_dir_name(cfg)
    mkdir_p(run_output_dir)
    initialize_logging(run_output_dir, to_file=True)
    cfg.run_output_dir = run_output_dir

    cfg.debug = False
    cfg.num_workers = 10
    cfg.pretrain.space_pair_limit = 0.15
    cfg.pretrain.space_limit_type = "degrees"
    cfg.pretrain.loss_type = "sssl"
    cfg.pretrain.include_same_loc = False

    cfg.landsat8_bands = "ALL"
    cfg.landsat8_replace_nan = False
    cfg.landsat8_normalize = False
    cfg.deterministic = True
    cfg.pretrain.pseudo_random_sampler = False
    cfg.pretrain.merge_val_test = False
    cfg.use_h5 = False

    result = {}
    dt = 1
    K = 8  # set to number of positives
    skip = False

    cfg.pretrain.time_pair_limit = dt
    cfg.pretrain.K = K
    cfg.pretrain.skip_valtest_wo_neighbors = skip

    # We assume there is no pre-training test set
    dl = build_dataloaders(cfg)["val"]

    no_t_pos, no_s_pos = [], []
    for batch in tqdm(iter(dl)):
        for (anchor, tgap, sgap) in zip(batch.anchors, batch.tgs, batch.sgs):
            if tgap > 0:
                no_t_pos.append(anchor.path)
            if sgap > 0:
                no_s_pos.append(anchor.path)

    no_t_pos, no_s_pos = list(set(no_t_pos)), list(set(no_s_pos))
    print(
        f"K={cfg.pretrain.K} - Dg={cfg.pretrain.space_pair_limit} - Dt={cfg.pretrain.time_pair_limit} "
        f"- no time pos={len(no_t_pos)} - no space pos={len(no_s_pos)}"
    )

    dataset = dl.dataset
    files = PretrainLandsat8Files(cfg, "val", load_box2pz=True)

    cfg_name = "K-%s_s-%s_t-%s_skip-%s" % (
        K,
        str(cfg.pretrain.space_pair_limit).replace(".", ""),
        dt,
        skip,
    )
    result[cfg_name] = {
        "config": {
            "K": cfg.pretrain.K,
            "space_pair_limit": cfg.pretrain.space_pair_limit,
            "time_pair_limit": cfg.pretrain.time_pair_limit,
        },
    }
    for name, no_pos in (
        ("time", no_t_pos),
        ("space", no_s_pos),
        ("all", list(set(no_t_pos + no_s_pos))),
    ):
        boxes = list(set([utils.box_to_str(files.path2zb[p][1]) for p in no_pos]))
        is_val_box = [b in files.boxes for b in boxes]
        val_boxes = [b for (b, isvb) in zip(boxes, is_val_box) if isvb]
        test_boxes = [b for (b, isvb) in zip(boxes, is_val_box) if not isvb]
        all_val_paths_box = list(
            set([pz[0] for b in val_boxes for pz in files.box2pz[b]])
        )
        all_test_paths_box = list(
            set([pz[0] for b in test_boxes for pz in files.box2pz[b]])
        )
        is_val_path = [
            utils.box_to_str(files.path2zb[p][1]) in val_boxes for p in no_pos
        ]
        val_paths = [p for (p, isvp) in zip(no_pos, is_val_path) if isvp]
        test_paths = [p for (p, isvp) in zip(no_pos, is_val_path) if not isvp]
        result[cfg_name].update(
            {
                name: {
                    "all_paths": no_pos,
                    "val_paths": val_paths,
                    "test_paths": test_paths,
                    "val_boxes": val_boxes,
                    "test_boxes": test_boxes,
                    "all_val_paths_box": all_val_paths_box,
                    "all_test_paths_box": all_test_paths_box,
                },
            }
        )

    with open("data/indices/val_test_without_neighbors.json", "w") as f:
        json.dump(result, f)

    boxes_to_excl = (
        result["K-8_s-015_t-1_skip-False"]["all"]["val_boxes"]
        + result["K-8_s-015_t-1_skip-False"]["all"]["test_boxes"]
    )
    paths_to_excl = list(set([pz[0] for b in boxes_to_excl for pz in files.box2pz[b]]))
    with open("data/indices/to_exclude.json", "w") as f:
        json.dump(
            {
                "boxes": boxes_to_excl,
                "paths": paths_to_excl,
                "based_on": "K-8_s-015_t-1_skip-False",
            },
            f,
        )
