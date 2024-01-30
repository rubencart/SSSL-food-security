import json
import logging

import h5py
import numpy as np
import torch
import yaml
from sssl.config import Config
from sssl.data.landsat8 import build_dataloaders
from sssl.utils import generate_output_dir_name, initialize_logging, mkdir_p
from tqdm import tqdm

logger = logging.getLogger("pytorch_lightning")


if __name__ == "__main__":

    with open("config/pretrain/debug.yaml", "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Config()
    cfg.from_dict(dict_cfg)
    cfg.process_args()

    run_output_dir = generate_output_dir_name(cfg)
    mkdir_p(run_output_dir)
    initialize_logging(run_output_dir, to_file=True)
    cfg.run_output_dir = run_output_dir

    cfg.landsat8_bands = "ALL"
    cfg.landsat8_replace_nan = False
    cfg.landsat8_normalize = False
    cfg.deterministic = True
    cfg.pretrain.pseudo_random_sampler = False
    cfg.num_workers = 16
    cfg.debug = False
    cfg.pretrain.K = 1
    cfg.pretrain.loss_type = "sssl"
    cfg.pretrain.skip_valtest_wo_neighbors = False
    cfg.pretrain.merge_val_test = False
    cfg.use_h5 = False

    out_file = "/path/to/landsat8/h5/tiles.h5"  # todo
    path_to_h5_idx = {}

    DEBUG = False
    TILE_SIZE = 145  # set to tile size

    dls = build_dataloaders(cfg)
    with h5py.File(out_file, "w", libver="latest") as h5file:
        h5file.swmr_mode = True

        # skip 'test' if there are no pretraining test paths
        #  (ie if --test_fraction in preprocess.py was 0.0)
        for split in ("ood", "val", "train"):
            loader = dls[split]
            logger.info("Running for split %s" % split)
            ds = loader.dataset
            size = len(ds) if not DEBUG else 10 * cfg.pretrain.batch_size
            ds = h5file.create_dataset(
                split,
                (size, 7, TILE_SIZE, TILE_SIZE),
                chunks=(1, 7, TILE_SIZE, TILE_SIZE),
                dtype=np.float32,
                compression="gzip",
            )
            path_to_h5_idx[split] = {}

            last = 0
            for batch_idx, batch in enumerate(tqdm(iter(loader))):
                assert isinstance(batch.tiles, torch.Tensor)
                if DEBUG and batch_idx >= 10:
                    break

                ds[last : last + batch.bs] = batch.tiles.squeeze(1).numpy()

                for i, tile in enumerate(batch.anchors):
                    key = tile.path
                    path_to_h5_idx[split][key] = last + i

                last += batch.bs

        # https://docs.h5py.org/en/stable/vds.html
        # https://github.com/h5py/h5py/blob/master/examples/dataset_concatenation.py
        # create virtual dataset for downstream task, where train/val/test split is different (based on regions
        # instead of on coordinates), so same downstream dataset (e.g. val) may need to access tiles from any
        # pretraining split
        nval = h5file["val"].shape[0]
        ntrain = h5file["train"].shape[0]
        nood = h5file["ood"].shape[0]
        layout = h5py.VirtualLayout(
            shape=(nval + ntrain + nood, 7, 145, 145), dtype=h5file["val"].dtype
        )

        valsrc = h5py.VirtualSource(h5file["val"])
        trainsrc = h5py.VirtualSource(h5file["train"])
        oodsrc = h5py.VirtualSource(h5file["ood"])

        layout[:nval, :, :, :] = valsrc
        layout[nval : nval + ntrain, :, :, :] = trainsrc
        layout[nval + ntrain : nval + ntrain + nood, :, :, :] = oodsrc

        h5file.create_virtual_dataset("all", layout)

    path_to_virt_h5_idx = {
        p: idx + (nval if s == "train" else (nval + ntrain if s == "ood" else 0))
        for s in path_to_h5_idx
        for (p, idx) in path_to_h5_idx[s].items()
    }
    with open("data/indices/path_to_h5_idx.json", "w") as f:
        json.dump(path_to_h5_idx, f)
    with open("data/indices/path_to_virtual_h5_idx.json", "w") as f:
        json.dump(path_to_virt_h5_idx, f)
