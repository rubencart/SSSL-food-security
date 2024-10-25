import copy
import datetime
import json
import logging
import os
import pprint
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from sssl import utils
from sssl.config import Config
from sssl.data.landsat8 import Landsat8Dataset, Landsat8Files, Tile
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("pytorch_lightning")


class Sample:
    tile: Tile
    ipc: int

    def __init__(self, tile: Tile, ipc: int):
        self.tile = copy.deepcopy(tile)
        self.ipc = copy.deepcopy(ipc)

    def to_dict(self) -> Dict[str, Union[int, Tile]]:
        return {"tile": self.tile, "ipc": self.ipc}


class IPCBatch:
    bs: int
    tiles: Union[torch.FloatTensor, np.ndarray]
    ipcs: Union[torch.LongTensor, Tuple]
    zones: Union[torch.LongTensor, List]
    dates: Union[torch.LongTensor, List]

    def __init__(self, samples: List[Sample], transform=None, to_pt=True):
        tiles, self.ipcs = zip(*((s.tile, s.ipc) for s in samples))
        self.bs = len(self.ipcs)
        self.zones, self.dates = zip(*((t.admin_zone, t.end_date) for t in tiles))
        shapes = np.array([a.array.shape for a in tiles])
        _, h, w = shapes.max(axis=0)

        # bs x ch x h x w
        self.tiles = np.stack([
            np.pad(
                a.array,
                ((0, 0), (0, h - a.array.shape[1]), (0, w - a.array.shape[2])),
                mode="constant",
                constant_values=0.0,
            )
            for a in tiles
        ])
        self.tiles = np.expand_dims(self.tiles, axis=1)
        if to_pt:
            self.tiles = torch.from_numpy(self.tiles)
            if transform is not None:
                self.tiles = transform(self.tiles)
            self.ipcs = torch.tensor(self.ipcs, dtype=torch.long)
            self.zones = torch.tensor(self.zones, dtype=torch.long)
            self.dates = torch.tensor(self.dates, dtype=torch.long)

    def __len__(self):
        return self.bs


class IPCBatchForEval:
    def __init__(self, batch: IPCBatch):
        self.bs = batch.bs
        self.ipcs = batch.ipcs.clone()
        self.zones = batch.zones.clone()
        self.dates = batch.dates.clone()


class IPCLandsat8Files(Landsat8Files):
    def __init__(
        self,
        cfg: Config,
        split: str,
        temporally_separated: bool = False,
        load_zone2box2p: bool = False,
        load_box2pz: bool = False,
        load_oor: bool = False,
    ):
        super().__init__(
            cfg,
            "downstream",
            load_zone2box2p=True,
            load_box2pz=load_box2pz,
            load_oor=load_oor,
        )
        self.split = split

        with open(os.path.join(cfg.indices_dir, self.split_fn), "r") as f:
            self.splits = json.load(f)
            self.regions = self.zones_4_split(split, self.splits, temporally_separated)
            self.paths = self.splits["paths"]

        if cfg.use_h5:
            with open(os.path.join(cfg.indices_dir, cfg.path_to_h5_virtual_idx)) as f:
                self.h5_idx = json.load(f)

    def zones_4_split(self, split: str, split_dict: Dict, temporally_separated: bool) -> List[str]:
        if temporally_separated:
            return (
                split_dict["ood_regions"]
                if split == "ood"
                else split_dict["val_regions"]
                + split_dict["test_regions"]
                + split_dict["train_regions"]
            )
        else:
            return split_dict[
                {
                    "val": "val_regions",
                    "test": "test_regions",
                    "train": f"train_regions_{self.cfg.finetune.percentage_of_training_data}",
                    "ood": "ood_regions",
                }[split]
            ]


class IPCScoreDataset(Landsat8Dataset):
    def __init__(self, cfg: Config, split: str = "train"):
        super().__init__(cfg, split=split)
        self.f: IPCLandsat8Files = IPCLandsat8Files(cfg, split, cfg.finetune.temporally_separated)
        # so workers initialize correct h5 dataset
        self.h5_split_name = "downstream"

        # this csv is from https://microdata.worldbank.org/index.php/catalog/3811 , it uses some different admin zone
        # names than the shapefile we used to link tiles to admin zones (https://fews.net/fews-data/334)
        self.df = pd.read_csv(cfg.ipc_scores_csv_path)
        self.df.admin_name = self.df.admin_name.apply(
            lambda x: utils.Constants.ADMIN_NAME_MAP.get(x, x)
        )

        if self.cfg.finetune.binarize_ipc:
            logger.info("Making IPC scores binary")
            self.df.fews_ipc = utils.binarize_ipcs(self.df.fews_ipc)
        all_ipcs = self.df.fews_ipc.unique()
        all_ipcs.sort()
        self.f.all_ipcs = all_ipcs.astype(int)
        self.f.ipc_dict = {ipc: i for (i, ipc) in enumerate(self.f.all_ipcs)}

        self.regions = self.f.regions
        self.zone_ids = [self.f.admin_dict[z] for z in self.regions]

        logger.info("Filtering downstream boxes for %s" % split)
        self.boxes = [b for z in tqdm(self.regions) for b in self.f.zone2box2p[z].keys()]
        fut = cfg.finetune.n_steps_in_future
        self.temp_sep = cfg.finetune.temporally_separated
        if self.temp_sep:
            date_idcs_dict = {
                "train": list(range(min(3 - fut, 2), len(self.f.all_end_dates) - max(fut + 1, 2))),
                "val": [len(self.f.all_end_dates) - max(fut + 1, 2)],
                "test": [len(self.f.all_end_dates) - max(fut, 1)],
                "ood": [len(self.f.all_end_dates) - max(fut, 1)],
            }
            date_dict = {
                sp: [self.f.all_end_dates[di] for di in idcs] for sp, idcs in date_idcs_dict.items()
            }

        logger.info("Filtering downstream paths for %s" % split)
        self.paths = [
            p
            for z in tqdm(self.regions)
            for plist in self.f.zone2box2p[z].values()
            for p in (
                utils.filter_paths_by_date(plist, date_dict[split]) if self.temp_sep else plist
            )
        ]
        logger.info(
            "Using %s boxes, %s paths, %s regions for %s"
            % (len(self.boxes), len(self.paths), len(self.regions), split)
        )

        if self.temp_sep and fut > 0:
            # append 1 step in future shp to df with ipc annotations
            shp = gpd.read_file(cfg.future_ipc_shp[0])
            fut_df = pd.DataFrame(
                data={
                    "admin_name": shp.ADMIN2,
                    "fews_ipc": shp.CS,
                }
            )
            col = shp.report_mon.apply(
                lambda x: (
                    datetime.datetime.strptime(x, "%m-%Y") + relativedelta(months=1)
                ).strftime("%Y-%m-%d")
            )
            fut_df = fut_df.assign(ymd=col.values)
            if self.cfg.finetune.binarize_ipc:
                fut_df.fews_ipc = utils.binarize_ipcs(fut_df.fews_ipc)
            self.df = pd.concat([self.df, fut_df], ignore_index=True)

            # add the new date to the list of possible dates
            self.f.all_end_dates += [col.iloc[0]]

        self.zone_time_combos = [
            (z, d, self.get_ipc_for_zone_date(z, d))
            for z in self.zone_ids
            for d in (
                range(len(self.f.all_end_dates) - cfg.finetune.n_steps_in_future)
                if not self.temp_sep
                else date_idcs_dict[split]
            )
        ]
        self.ipc_2_zd = defaultdict(list)
        for z, d, ipc in self.zone_time_combos:
            self.ipc_2_zd[ipc].append((z, d))

        self.log_ipc_distributions()
        print("")

    def log_ipc_distributions(self):
        per_date = {}
        per_zone = {}
        for z, d, ipc in self.zone_time_combos:
            per_date.setdefault(d, []).append(ipc)
            per_zone.setdefault(z, []).append(ipc)
        per_date = {
            self.f.all_end_dates[d]: np.bincount(ipcs, minlength=4) / len(ipcs)
            for d, ipcs in per_date.items()
        }
        per_zone = {
            self.f.all_admins[z]: np.bincount(ipcs, minlength=4) / len(ipcs)
            for z, ipcs in per_zone.items()
        }
        overall = np.bincount([ipc for (z, d, ipc) in self.zone_time_combos], minlength=4) / len(
            self.zone_time_combos
        )
        logger.info("IPC distribution overall: %s" % pprint.pformat(overall))
        logger.info("IPC distribution per date: \n%s" % pprint.pformat(per_date))
        logger.info("IPC distribution per zone: \n%s" % pprint.pformat(per_zone))

    def __len__(self) -> int:
        return len(self.paths)

    def get_ipc_for_zone_date(self, zonei: int, datei: int) -> int:
        datei2 = datei + self.cfg.finetune.n_steps_in_future
        try:
            rows = self.df[
                (self.df.admin_name == self.f.all_admins[zonei])
                & (self.df.ymd == self.f.all_end_dates[datei2])
            ]
            return self.f.ipc_dict[int(rows.iloc[0].fews_ipc)]
        except IndexError as e:
            logger.error(
                "Did not find ipc rows for zone %s and date %s"
                % (
                    (zonei, self.f.all_admins[zonei]),
                    (datei2, self.f.all_end_dates[datei2]),
                )
            )
            raise e

    def __getitem__(self, item: int) -> Sample:
        path = self.paths[item]
        # boxes stored in [left, bottom, right, top] order
        tile = self.tile_from_path(path)
        ipc = self.get_ipc_for_zone_date(tile.admin_zone, tile.end_date)
        return Sample(tile, ipc)

    def sample_from_ipc_class(self, ipc: int, shuffle=False) -> Sample:
        paths = []
        for z, d in (
            random.sample(self.ipc_2_zd[ipc], k=len(self.ipc_2_zd[ipc]))
            if shuffle
            else self.ipc_2_zd[ipc]
        ):
            paths += [
                pathlist[d]
                for pathlist in self.f.zone2box2p[self.f.all_admins[z]].values()
                if len(pathlist) > d
            ]
        for path in random.sample(paths, k=len(paths)) if shuffle else paths:
            # boxes stored in [left, bottom, right, top] order
            tile = self.tile_from_path(path)
            ipc = self.get_ipc_for_zone_date(tile.admin_zone, tile.end_date)
            yield Sample(tile, ipc)

    def collate(self, list_of_samples: List[Sample]) -> IPCBatch:
        return IPCBatch(list_of_samples, transform=self.norm_transform)


def train_dataloader(ds: IPCScoreDataset, cfg: Config) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.pretrain.batch_size,
        pin_memory=cfg.cuda and cfg.pin_memory,
        drop_last=cfg.do_train and not cfg.do_test,
        num_workers=cfg.num_workers if not cfg.debug else cfg.debug_num_workers,
        collate_fn=ds.collate,
        shuffle=not cfg.deterministic,
        persistent_workers=cfg.persistent_workers,
    )


def inference_dataloader(ds: IPCScoreDataset, cfg: Config) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.pretrain.val_batch_size,
        pin_memory=cfg.cuda and cfg.pin_memory,
        num_workers=cfg.num_workers if not cfg.debug else cfg.debug_num_workers,
        drop_last=False,
        collate_fn=ds.collate,
        shuffle=False,
        persistent_workers=cfg.persistent_workers,
    )


def build_dataloaders(cfg: Config) -> Dict[str, DataLoader]:
    ds_dict = {}
    splits = ("test", "val", "ood")
    for split in splits:
        ds = IPCScoreDataset(cfg, split)
        ds_dict[split] = inference_dataloader(ds, cfg)
    if cfg.do_train or cfg.save_train_predictions:
        ds_dict["train"] = train_dataloader(IPCScoreDataset(cfg, "train"), cfg)
    return ds_dict
