import json
import logging
import math
import os
import random
from abc import ABC
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np
import rasterio as rio
import torch
import torchvision.transforms
from dateutil.relativedelta import relativedelta
from sssl import utils
from sssl.config import Config
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

logger = logging.getLogger("pytorch_lightning")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class Tile:
    def __init__(
        self,
        path: str,
        dirpath: str = None,
        lat: int = None,
        lon: int = None,
        end_date: int = None,
        zone: int = None,
        array: Union[np.ndarray, Tensor] = None,
        rgb=False,
        replace_nan=True,
        to_pt=False,
    ):
        self.path = path
        self.admin_zone = zone
        # with stmt closes ds, because not pickleable
        if array is None:
            with rio.open(Path(dirpath) / path) as ds:
                self.array = np.nan_to_num(ds.read()) if replace_nan else ds.read()
                self.bounds = ds.bounds
        else:
            self.array = array
            if replace_nan:
                self.array = (
                    np.nan_to_num(array)
                    if isinstance(array, np.ndarray)
                    else torch.nan_to_num(array)
                )
        if to_pt and not isinstance(self.array, Tensor):
            self.array = utils.tensor(self.array)

        if rgb and array.shape[0] != 3:
            # https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2#bands
            self.array = self.array[utils.Constants.RGB_BANDS]
        if lat is None or lon is None or end_date is None:
            raise NotImplementedError
        else:
            self.lat = lat
            self.lon = lon
            self.end_date = end_date


class Sample:
    anchor: Tile
    positives: List[Tile]

    def __init__(
        self,
        anchor: Tile,
        positives: List[Tile] = None,
        negatives: List[Tile] = None,
        space_gap: int = 0,
        time_gap: int = 0,
    ):
        self.anchor = anchor
        self.space_gap = space_gap
        self.time_gap = time_gap
        self.negatives = negatives if negatives else []
        self.n_neg = len(negatives) if negatives else 0
        self.positives = positives if positives else []
        self.n_pos = len(positives) if positives else 0

    def to_dict(self):
        return {
            "anchor": self.anchor,
            "negatives": self.negatives,
            "positives": self.positives,
            "space_gap": self.space_gap,
            "time_gap": self.time_gap,
        }


class Batch:
    bs: int
    tiles: Union[torch.FloatTensor, np.ndarray]
    anchors: List[Tile]
    positives: List[List[Tile]]
    sgs: Tuple[int]
    tgs: Tuple[int]

    def __init__(self, samples: List[Sample], transform=None):
        # list_of_dicts = [s.to_dict() for s in samples]
        anchors, positives = zip(
            *(
                (
                    s.anchor,
                    s.positives,
                )
                for s in samples
            )
        )
        self.sgs, self.tgs = zip(
            *(
                (
                    s.space_gap,
                    s.time_gap,
                )
                for s in samples
            )
        )
        self.bs = len(anchors)
        self.anchors = anchors
        self.positives = positives

        shapes = np.array([a.array.shape for a in anchors])
        _, h, w = shapes.max(axis=0)

        # bs x ch x h x w
        # we always pad right/bottom but for tiles coming from west/north border of somalia it
        #   might make more sense to pad left/top
        ta = torch.stack(
            [
                #                             L, R, T, B
                F.pad(
                    utils.tensor(a.array),
                    (0, w - a.array.shape[2], 0, h - a.array.shape[1]),
                    mode="constant",
                    value=0.0,
                )
                for a in anchors
            ]
        )

        # bs x (K-1) x ch x h x w
        if len(positives[0]) > 0:
            tp = torch.stack(
                [
                    torch.stack(
                        [
                            F.pad(
                                utils.tensor(p.array),
                                (0, w - p.array.shape[2], 0, h - p.array.shape[1]),
                                mode="constant",
                                value=0.0,
                            )
                            for p in plist
                        ]
                    )
                    for plist in positives
                ],
            )
        else:
            tp = torch.empty(size=(len(positives), 0))
        # bs x K x ch x h x w
        self.tiles = (
            torch.cat((ta.unsqueeze(1), tp), dim=1) if tp.dim() > 2 else ta.unsqueeze(1)
        )
        if transform is not None:
            self.tiles = transform(self.tiles)

    def __len__(self):
        return self.bs


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Landsat8Files:
    paths: List[str]
    regions: List[str]
    h5_ds: Optional[h5py.Dataset] = None
    h5_file: Optional[h5py.File] = None
    h5_idx: Optional[Dict[str, int]] = None

    def __init__(
        self,
        cfg: Config,
        split: str,
        load_zone2box2p: bool = False,
        load_box2pz: bool = False,
        load_oor: bool = False,
        load_h5: bool = False,
    ):
        self.cfg = cfg

        logger.info("Reading files...")
        with open(os.path.join(cfg.indices_dir, cfg.path2zb), "r") as f:
            self.path2zb = json.load(f)

        if load_zone2box2p:
            with open(os.path.join(cfg.indices_dir, cfg.zone2box2p), "r") as f:
                self.zone2box2p = json.load(f)

                # counts = {z: sum([len(bpd[b]) for b in bpd]) for z, bpd in self.zone2box2p.items()}
                # logger.info('Admin zone distribution:')
                # logger.info(pprint.pformat(counts, indent=2))

        if load_box2pz:
            with open(os.path.join(cfg.indices_dir, cfg.box2pz), "r") as f:
                self.box2pz = json.load(f)

                logger.info("Paths per box:")
                c = Counter([len(bl) for bl in self.box2pz.values()])
                logger.info(c)

        if load_oor:
            with open(os.path.join(cfg.indices_dir, cfg.oor_tiles), "r") as f:
                self.oor_tiles = json.load(f)

        with open(os.path.join(cfg.indices_dir, cfg.dicts_path), "r") as f:
            dicts = json.load(f)
            self.all_lats = dicts["all_lats"]
            self.all_lons = dicts["all_lons"]
            self.all_end_dates = dicts["all_end_dates"]
            self.all_admins = dicts["all_admins"]
            self.lat_dict = dicts["lat_dict"]
            self.lon_dict = dicts["lon_dict"]
            self.admin_dict = dicts["admin_dict"]
            self.date_dict = dicts["date_dict"]
            self.all_end_dates = dicts["all_end_dates"]
        self.split_fn = {
            "ood": cfg.ood_splits_path,
            "val": cfg.val_splits_path,
            "test": cfg.test_splits_path,
            "train": cfg.train_splits_path,
            "downstream": cfg.downstr_splits_path,
        }[split]

        if False and load_h5:
            self.init_h5(split)

        logger.info("Done reading files")

    def init_h5(self, split: str):
        logger.debug("Initializing h5 from: %s" % self.cfg.tiles_h5)
        self.h5_file = h5py.File(
            self.cfg.tiles_h5, "r", swmr=self.cfg.use_h5_swmr, libver="latest"
        )
        ds_name = {
            "ood": "ood",
            "val": "val",
            "test": "test",
            "train": "train",
            "downstream": "all",
        }[split]
        self.h5_ds = self.h5_file[ds_name]

    @staticmethod
    def build(cfg: Config, split: str) -> "Landsat8Files":
        if cfg.do_pretrain:
            return PretrainLandsat8Files(cfg, split)
        else:
            assert cfg.do_downstream
            from data.ipc import IPCLandsat8Files

            return IPCLandsat8Files(cfg, split)

    def __del__(self):
        logger.info("Closing h5 from Landsat8Files")
        if self.h5_file is not None:
            logger.info("Closing h5")
            self.h5_file.close()


class PretrainLandsat8Files(Landsat8Files):
    def __init__(
        self,
        cfg: Config,
        split: str,
        load_zone2box2p: bool = False,
        load_box2pz: bool = False,
        load_oor: bool = False,
    ):
        super().__init__(cfg, split, load_zone2box2p, load_box2pz, load_oor)

        with open(os.path.join(cfg.indices_dir, self.split_fn), "r") as f:
            self.splits = json.load(f)
            self.paths = self.splits["paths"]
            self.regions = self.splits["regions"]
            self.boxes = self.splits["boxes"]
            self.lat_lon_date_path = self.splits["map"]
            # self.admin_date_paths = self.splits['admin_map']

        if cfg.pretrain.merge_val_test and split in ("val", "test"):
            self.splits["valtest_paths"] = (
                self.splits["val_paths"] + self.splits["test_paths"]
            )
            self.splits["valtest_boxes"] = (
                self.splits["val_boxes"] + self.splits["test_boxes"]
            )

        if cfg.pretrain.skip_valtest_wo_neighbors:
            with open(
                os.path.join(cfg.indices_dir, cfg.valtest_wo_neighbors), "r"
            ) as f:
                self.valtest_wo_neighbors = json.load(f)
                self.skip_paths = self.valtest_wo_neighbors["paths"]
                self.skip_boxes = self.valtest_wo_neighbors["boxes"]

        self.pseudo_random_boxes = None
        self.pseudo_random_dates = None
        if cfg.do_pretrain and cfg.pretrain.pseudo_random_sampler:
            with open(os.path.join(cfg.indices_dir, cfg.fixed_random_order_path)) as f:
                pseudo_dct = json.load(f)
                self.pseudo_random_boxes = pseudo_dct["boxes"]
                self.pseudo_random_dates = pseudo_dct["dates"]

        if cfg.use_h5:
            with open(os.path.join(cfg.indices_dir, cfg.path_to_h5_idx)) as f:
                self.h5_idx = json.load(f)[split]


class Landsat8Dataset(Dataset, ABC):
    f: Landsat8Files

    def __init__(self, cfg: Config, split: str = "train"):
        super().__init__()
        self.split = split
        self.h5_split_name = split
        self.cfg = cfg
        self.deterministic = (split != "train") or cfg.deterministic
        self.rgb = cfg.landsat8_bands == "RGB"
        self.replace_nan = cfg.landsat8_replace_nan
        if cfg.landsat8_normalize:
            bslice = range(7) if not self.rgb else utils.Constants.RGB_BANDS
            self.norm_transform = torchvision.transforms.Normalize(
                mean=utils.Constants.CHANNEL_MEANS[bslice],
                std=utils.Constants.CHANNEL_STDS[bslice],
            )
        else:
            self.norm_transform = None

    def tile_from_path(self, path: str, lldz=None) -> Tile:
        if lldz is None:
            lat, lon, end_date, zone = self.lat_lon_date_zone_from_path(path)
        else:
            lat, lon, end_date, zone = lldz
        if self.cfg.use_h5:
            if self.f.h5_ds is None:
                logger.debug("Initializing h5")
                self.f.init_h5(self.h5_split_name)
            h5_i = self.f.h5_idx[path.replace(self.cfg.tiles_dir, "")]
            arr = self.f.h5_ds[h5_i]
        else:
            arr = None
        return Tile(
            path,
            self.cfg.tiles_dir,
            lat,
            lon,
            end_date,
            zone,
            array=arr,
            rgb=self.rgb,
            replace_nan=self.replace_nan,
        )

    def lat_lon_date_zone_from_path(self, path: str) -> Tuple[int, int, int, int]:
        zone, (lon, _, _, lat) = self.f.path2zb[path]
        end_date = utils.path_to_end_date(path, ftstr=True)
        return (
            self.f.lat_dict[str(lat)],
            self.f.lon_dict[str(lon)],
            self.f.date_dict[end_date],
            self.f.admin_dict[zone],
        )

    def zone_from_path(self, path):
        zone = self.f.path2zb[path][0]
        return self.f.admin_dict[zone]

    def __del__(self):
        logger.info("Closing h5 from Landsat8Dataset")
        if self.f.h5_file is not None:
            logger.info("Closing h5")
            self.f.h5_file.close()


class ContrastiveDataset(Landsat8Dataset, ABC):
    def __init__(
        self, cfg: Config, split: str = "train", load_zone2box2p: bool = False
    ):
        super().__init__(cfg, split)
        self.f: PretrainLandsat8Files = PretrainLandsat8Files(
            cfg, split, load_zone2box2p
        )

        self.space_limit = cfg.pretrain.space_pair_limit
        self.time_limit = cfg.pretrain.time_pair_limit

        self.lat_lon_date_path = self.f.lat_lon_date_path
        self.boxes = self.f.boxes

        if (
            split in ("val", "test", "valtest")
            and cfg.pretrain.skip_valtest_wo_neighbors
        ):
            logger.info("Filtering tiles with too few neighbors out of test/val set...")
            logger.info(
                "BEFORE: Using %s boxes, %s paths, %s regions for %s"
                % (len(self.boxes), len(self.f.paths), len(self.f.regions), split)
            )
            self.f.boxes = [p for p in tqdm(self.boxes) if p not in self.f.skip_boxes]
            self.f.paths = [
                p
                for p in tqdm(self.f.paths)
                if utils.box_to_str(self.f.path2zb[p][1]) not in self.f.skip_boxes
            ]
        logger.info(
            "Using %s boxes, %s paths, %s regions for %s"
            % (len(self.boxes), len(self.f.paths), len(self.f.regions), split)
        )
        logger.info("Constructed %s dataset" % split)

        # confirm shaped like somalia
        # if self.cfg.debug:
        #     lat_lon_date_lens = np.array([
        #         [
        #             len([p for p in self.lat_lon_date_path[i][j] if p is not None])
        #             for j in range(len(self.all_lons))
        #         ] for i in range(len(self.all_lats))
        #     ])
        #     plt.close()
        #     plt.imshow(lat_lon_date_lens, cmap='hot', interpolation='nearest')
        #     plt.show()

    def sample_anchor(self, item: int) -> Tile:
        path = self.f.paths[item]
        # boxes stored in [left, bottom, right, top] order
        return self.tile_from_path(path)

    def sample_positives(
        self, anchor: Tile, anchor_idx: int, n: int
    ) -> Tuple[List[Tile], int, int]:

        space_pos_coords: List[Tuple[int, int]] = self.get_space_positives(anchor)
        shuffled_pos = self.pseudo_shuffle(
            space_pos_coords, anchor_idx, self.f.pseudo_random_boxes
        )
        positives = self.get_time_positives(anchor, anchor_idx, shuffled_pos, n)

        space_gap, time_gap = self.validate_nb_positives(
            space_pos_coords, positives, anchor, n
        )

        positives = positives[:n]
        return positives, space_gap, time_gap

    def validate_nb_positives(self, space_pos, positives, anchor, n):
        space_gap, time_gap = 0, 0
        if len(space_pos) * len(self.f.all_end_dates) < n:
            msg = (
                "Less than %s SPACE-positives were found for anchor %s with space limit %s and time limit %s"
                % (n, anchor.path, self.space_limit, self.time_limit)
            )
            logger.error(msg)
            space_gap = n - len(space_pos)
        elif len(positives) < n:
            msg = (
                "Less than %s TIME-positives were found for anchor %s with space limit %s and time limit %s"
                % (n, anchor.path, self.space_limit, self.time_limit)
            )
            logger.error(msg)
            time_gap = n - len(positives)
        return space_gap, time_gap

    def pseudo_shuffle(
        self, to_shuffle: List[Any], anchor_idx: int, all_idcs: List[int]
    ) -> List[Any]:
        if not self.deterministic:
            shuffled = random.sample(to_shuffle, k=len(to_shuffle))
        elif not (self.cfg.do_pretrain and self.cfg.pretrain.pseudo_random_sampler):
            shuffled = to_shuffle
        else:
            # fake a random shuffle, otherwise tiles from same location always have same positive
            shift = all_idcs[anchor_idx % max(1, len(all_idcs))]
            all_idcs = [i for i in all_idcs if i < len(to_shuffle)]
            all_idcs = all_idcs[shift:] + all_idcs[:shift]
            if len(all_idcs) < len(to_shuffle):
                logger.error("Too few pseudo random indices!")
                all_idcs = math.ceil(len(to_shuffle) / len(all_idcs)) * all_idcs
            idcs = all_idcs[: len(to_shuffle)]
            shuffled = [to_shuffle[i] for i in idcs]
        return shuffled

    def get_space_positives(self, anchor: Tile) -> List[Tuple[int, int]]:
        # These are indices!
        anchor_lati, anchor_loni = anchor.lat, anchor.lon
        anchor_lat, anchor_lon = float(self.f.all_lats[anchor_lati]), float(
            self.f.all_lons[anchor_loni]
        )
        space_positive = []
        offset = -1 if self.cfg.pretrain.include_same_loc else 0
        incr_offset = True
        while incr_offset:
            offset += 1
            indices = self.get_offset_indices(anchor_lati, anchor_loni, offset)
            indices = self.filter_out_of_bounds(indices, anchor_lat, anchor_lon)

            if len(indices) < 1 and offset > 1:
                incr_offset = False

            for (lat, lon) in indices:

                for (date, p) in enumerate(self.lat_lon_date_path[lat][lon]):
                    if p is not None and (lat, lon) not in space_positive:
                        space_positive.append((lat, lon))
                        break

        return space_positive

    def get_time_positives(
        self, anchor: Tile, anchor_i: int, space_pos: List[Tuple[int, int]], n: int
    ) -> List[Tile]:
        """
        space_pos: shuffled already
        """
        positives = []
        dati = anchor.end_date
        anchor_date = self.f.all_end_dates[dati]
        parsed_anchor_date = datetime.strptime(anchor_date, "%Y-%m-%d")
        lower_date = parsed_anchor_date - relativedelta(months=self.time_limit)
        upper_date = parsed_anchor_date + relativedelta(months=self.time_limit)

        # build list of time positive candidates that's either all the timestamps from the space positive boxes or
        #  a sample of them, so we don't have to load all paths in memory
        #  + round-robin now, so more random:
        time_pos_cands = []
        cand_idx = 0
        cand_idcs = {}
        while cand_idx < len(self.f.all_end_dates) and len(time_pos_cands) < n:
            for sp_i, (lat, lon) in enumerate(space_pos):
                if sp_i not in cand_idcs:
                    cand_idcs[sp_i] = self.pseudo_shuffle(
                        list(range(len(self.f.all_end_dates))),
                        anchor_i + sp_i,
                        self.f.pseudo_random_dates,
                    )
                date = cand_idcs[sp_i][cand_idx]
                # for p in ...:  don't use nested loop or break won't work -> very slow
                if (
                    len(self.lat_lon_date_path[lat][lon]) > date
                    and self.lat_lon_date_path[lat][lon][date] is not None
                ):
                    p = self.lat_lon_date_path[lat][lon][date]
                    cand_date = datetime.strptime(
                        self.f.all_end_dates[date], "%Y-%m-%d"
                    )
                    if lower_date <= cand_date <= upper_date:

                        time_pos_cands.append(
                            self.tile_from_path(
                                p, lldz=(lat, lon, date, self.zone_from_path(p))
                            )
                        )
                        if len(time_pos_cands) >= n:

                            break
            cand_idx += 1

        positives = time_pos_cands
        return positives

    def get_offset_indices(self, lati, loni, offset):
        indices = []
        for offset_sign in [1, -1]:
            # lat constant, corners inclusive
            lat = lati + offset_sign * offset
            indices += [(lat, lon) for lon in range(loni - offset, loni + offset + 1)]
            # lon constant, corners exclusive
            lon = loni + offset_sign * offset
            indices += [(lat, lon) for lat in range(lati - offset + 1, lati + offset)]
            indices = [
                (lat, lon)
                for (lat, lon) in indices
                if 0 <= lat < len(self.f.all_lats) and 0 <= lon < len(self.f.all_lons)
            ]
        return list(set(indices))

    def filter_out_of_bounds(
        self, indices: List[Tuple[int, int]], lat: float, lon: float
    ) -> List[Tuple[int, int]]:
        result = []
        dist_fn = (
            lambda *x: self.sq_out_of_bounds(*x)
            if self.cfg.pretrain.spatial_pos_shape == "square"
            else self.circ_out_of_bounds(*x)
        )
        for (lati, loni) in indices:
            if not dist_fn(lat, lon, lati, loni):
                result.append((lati, loni))
        return result

    def sq_out_of_bounds(
        self, anchor_lat: float, anchor_lon: float, lati: int, loni: int
    ) -> bool:
        lat = float(self.f.all_lats[lati])
        lon = float(self.f.all_lons[loni])
        return (
            math.fabs(lat - anchor_lat) > self.space_limit
            or math.fabs(lon - anchor_lon) > self.space_limit
        )

    def circ_out_of_bounds(self, anchor_lat, anchor_lon, lati, loni):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.f.paths)

    def collate(self, list_of_samples: List[Sample]) -> Batch:
        return Batch(list_of_samples, transform=self.norm_transform)


class RelReasoningDataset(ContrastiveDataset):
    def __init__(self, cfg: Config, split="train"):
        super().__init__(cfg, split)

        color_jitter = torchvision.transforms.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )
        rnd_color_jitter = torchvision.transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = torchvision.transforms.RandomGrayscale(p=0.2)
        rnd_resizedcrop = torchvision.transforms.RandomResizedCrop(
            size=utils.Constants.TILE_SIZE,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=InterpolationMode.BILINEAR,
        )
        rnd_hflip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        self.train_transform_no_color = torchvision.transforms.Compose(
            [
                # ToTensor expects (H, W, C)
                rnd_resizedcrop,
                rnd_hflip,
            ]
        )
        self.train_transform_color = torchvision.transforms.Compose(
            [
                rnd_color_jitter,
                rnd_gray,
            ]
        )

    def __getitem__(self, item: int) -> Sample:
        anchor: Tile = self.sample_anchor(item)

        n = self.cfg.pretrain.K - 1
        positives = []
        for _ in range(n):
            if self.rgb and anchor.array.shape[0] == 3:
                t_array = self.train_transform_color(
                    self.train_transform_no_color(utils.tensor(anchor.array))
                )
            else:
                t_array = self.train_transform_no_color(utils.tensor(anchor.array))
                rgb_t_array = t_array[utils.Constants.RGB_BANDS]
                rgb_t_array = self.train_transform_color(rgb_t_array)
                t_array[utils.Constants.RGB_BANDS] = rgb_t_array

            pos = Tile(
                anchor.path,
                self.cfg.tiles_dir,
                anchor.lat,
                anchor.lon,
                anchor.end_date,
                anchor.admin_zone,
                array=t_array,
                rgb=self.rgb,
                replace_nan=self.replace_nan,
            )
            positives.append(pos)

        return Sample(anchor, positives)


class SSSLDataset(ContrastiveDataset):
    def __init__(self, cfg: Config, split="train"):
        super().__init__(cfg, split)

    def __getitem__(self, item: int) -> Sample:
        anchor = self.sample_anchor(item)
        n = self.cfg.pretrain.K - 1
        sg, tg = 0, 0
        if n > 0:
            positives, sg, tg = self.sample_positives(anchor, item, n=n)
            # positives = self.sample_positives(anchor, n=n)
            if len(positives) < n:
                positives += [anchor for _ in range(n - len(positives))]
        else:
            positives = []

        return Sample(anchor, positives, space_gap=sg, time_gap=tg)


class AdminZoneMixin(ContrastiveDataset):
    # todo since admin zones contain many tiles, the chance that another sample in the batch
    #   is from the same zone is high, we should consider only using negatives from different zones
    def __init__(self, cfg, split):
        super().__init__(cfg, split, load_zone2box2p=True)

    def get_space_positives(self, anchor: Tile) -> List[Tuple[int, int]]:
        # These are indices!
        zonei = anchor.admin_zone
        zone = self.f.all_admins[zonei]

        def to_coord_ids(box_str: str) -> Tuple[int, int]:
            lats, lons = utils.boxstr_to_tl_lat_lon(box_str)
            return self.f.lat_dict[lats], self.f.lon_dict[lons]

        space_pos_coords: List[Tuple[int, int]] = [
            to_coord_ids(b)
            for b in self.f.zone2box2p[zone]
            if self.cfg.pretrain.include_same_loc
            or (to_coord_ids(b) != (anchor.lat, anchor.lon))
        ]

        return space_pos_coords


class AdminZoneSSSLDataset(SSSLDataset, AdminZoneMixin):
    pass


class Tile2VecDataset(ContrastiveDataset):
    def __init__(self, cfg: Config, split="train"):
        super().__init__(cfg, split)
        # self.time_limit = utils.Constants.MAX_TIME_DIFFERENCE

    def __getitem__(self, item: int) -> Sample:
        anchor = self.sample_anchor(item)
        # only 1 positive!
        positives, sg, tg = self.sample_positives(anchor, item, n=1)
        return Sample(anchor, positives=positives, space_gap=sg, time_gap=tg)


class AdminZoneTile2VecDataset(Tile2VecDataset, AdminZoneMixin):
    pass


class FixedOrderSampler(Sampler[int]):
    """
    Use this sampler for deterministic sample order in val/test sets or for debugging.
    Loading samples in 0,1,2,3,... order is not a good idea because one location's timestamps have adjacent
    dataset indices, so a batch will consist of multiple tiles of the same location, so negatives will
    be close to each other!
    """

    def __init__(self, cfg: Config, split: str):
        with open(os.path.join(cfg.indices_dir, cfg.fixed_random_order_path)) as f:
            self.indices = json.load(f)[split]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def train_dataloader(ds: ContrastiveDataset, cfg: Config) -> DataLoader:
    if cfg.deterministic and cfg.pretrain.pseudo_random_sampler:
        kwargs = {
            "sampler": FixedOrderSampler(cfg, "train"),
        }
    else:
        kwargs = {"shuffle": not cfg.deterministic}
    return DataLoader(
        ds,
        batch_size=cfg.pretrain.batch_size,
        pin_memory=cfg.cuda and cfg.pin_memory,
        drop_last=False,
        num_workers=cfg.num_workers if not cfg.debug else cfg.debug_num_workers,
        collate_fn=ds.collate,
        **kwargs,
    )


def inference_dataloader(ds: ContrastiveDataset, cfg: Config, split: str) -> DataLoader:
    if cfg.pretrain.pseudo_random_sampler:
        split = (
            split.replace("val", "val_excl")
            if cfg.pretrain.skip_valtest_wo_neighbors
            else split
        )
        kwargs = {
            "sampler": FixedOrderSampler(cfg, split),
        }
    else:
        kwargs = {"shuffle": False}
    return DataLoader(
        ds,
        batch_size=cfg.pretrain.val_batch_size,
        pin_memory=cfg.cuda and cfg.pin_memory,
        num_workers=cfg.num_workers if not cfg.debug else cfg.debug_num_workers,
        drop_last=False,
        collate_fn=ds.collate,
        **kwargs,
    )


def build_dataloaders(cfg: Config) -> Dict[str, DataLoader]:
    if cfg.pretrain.loss_type == "tile2vec":
        if cfg.pretrain.space_limit_type == "admin":
            DsType = AdminZoneTile2VecDataset
        else:
            assert cfg.pretrain.space_limit_type == "degrees"
            DsType = Tile2VecDataset
    else:
        assert cfg.pretrain.loss_type == "sssl"
        if cfg.pretrain.augmentations == "rel_reasoning":
            DsType = RelReasoningDataset
        else:
            assert cfg.pretrain.augmentations == "sssl"
            if cfg.pretrain.space_limit_type == "admin":
                DsType = AdminZoneSSSLDataset
            else:
                assert cfg.pretrain.space_limit_type == "degrees"
                DsType = SSSLDataset
    logger.info("Using dataset type: %s" % DsType)

    ds_dict = {}
    splits = ("valtest", "ood") if cfg.pretrain.merge_val_test else ("val", "ood")
    if cfg.do_train:
        ds_dict["train"] = train_dataloader(DsType(cfg, "train"), cfg)
    for split in splits:
        ds = DsType(cfg, split)
        ds_dict[split.replace("valtest", "val")] = inference_dataloader(ds, cfg, split)
    return ds_dict
