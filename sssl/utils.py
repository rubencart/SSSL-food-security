import copy
import errno
import json
import logging
import os
import re
import socket
import sys
from argparse import ArgumentParser
from datetime import datetime
from shutil import rmtree
from typing import Any, Collection, Dict, List, Optional, Tuple, Union

import dateutil.tz
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
import yaml
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from rasterio.coords import BoundingBox
from torch import Tensor

from sssl.config import Config, ConfigNs

logger = logging.getLogger("pytorch_lightning")


def generate_output_dir_name(cfg: Config) -> str:
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    host = socket.gethostname()

    output_dir = cfg.output_dir

    run_output_dir = os.path.join(
        output_dir,
        "%s%s_s%s_%s"
        % (
            timestamp,
            "_debug" if cfg.debug else "",
            cfg.seed,
            cfg.cfg_name,
        ),
    )
    return run_output_dir


def initialize_logging(output_dir, to_file=True, logger_name="pytorch_lightning"):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
    if len(logger.handlers) > 0:
        logger.handlers[0].setFormatter(formatter)
    else:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.setLevel(logging.INFO)

    if to_file:
        path = os.path.join(output_dir, "console-output.log")
        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        logger.info("Initialized logging to %s" % path)
        logger.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    return logger


def build_configs() -> Tuple[Config, ConfigNs]:

    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--vsc", action="store_true")

    args = parser.parse_args()

    cfg, train_ns_cfg = path_to_config(args.cfg)

    if args.seed > 0:
        cfg.seed = args.seed
    return cfg, train_ns_cfg


def path_to_config(path: str, seed=-1) -> Tuple[Config, ConfigNs]:
    dict_cfg = (
        path_to_dict_config(path)
        if path.endswith("yaml")
        else json_path_to_dict_config(path)
    )
    if seed > -1:
        dict_cfg["seed"] = seed
    cfg = Config()
    cfg.from_dict(dict_cfg)
    cfg.process_args()
    print(cfg.train.as_dict())
    train_ns_cfg = ConfigNs(cfg.train.as_dict())
    return cfg, train_ns_cfg


def path_to_dict_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    return dict_cfg


def json_path_to_dict_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        dict_cfg = json.load(f)
    return dict_cfg


def ckpt_dir_to_name(ckpt_dir: str) -> str:
    return os.path.split(
        ckpt_dir.rstrip("/").replace("/checkpoints2", "").replace("/checkpoints", "")
    )[-1]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rmdir_p(path):
    if os.path.isdir(path):
        rmtree(path)
    elif not os.path.exists(path):
        pass
    elif os.path.isfile(path):
        raise FileExistsError("Path is file: %s" % path)
    else:
        raise Exception


def box_to_str(b: Union[BoundingBox, List]) -> str:
    if not isinstance(b, BoundingBox):
        b = BoundingBox(*b)
    return f"TL(lon:{b.left}-lat:{b.top})-BR(lon:{b.right}-lat:{b.bottom})"


TL_LAT_REGEX = r"TL\(lon:[-0-9.]+-lat:([-0-9.]+)\)"
TL_LON_REGEX = r"TL\(lon:([-0-9.]+)-lat:"


def boxstr_to_tl_lat_lon(
    boxstr: str, to_float=False
) -> Tuple[Union[str, int], Union[str, int]]:
    fn = float if to_float else (lambda x: x)
    return fn(re.findall(TL_LAT_REGEX, boxstr)[0]), fn(
        re.findall(TL_LON_REGEX, boxstr)[0]
    )


def boxstr_to_lats(boxstr) -> List[str]:
    return re.findall(TL_LAT_REGEX, boxstr)


def boxstr_to_lons(boxstr) -> List[str]:
    return re.findall(TL_LON_REGEX, boxstr)


DATE_REGEX = r"somalia_m[3|4]_([0-9-]+)_"
MONTH_REGEX = r"somalia_m(3|4)_"
DATE_FORMAT = "%Y-%m-%d"


def path_to_end_date(path: str, ftstr=False) -> Union[str, datetime]:
    m = int(re.search(MONTH_REGEX, path).group(1))
    d = re.search(DATE_REGEX, path).group(1)
    dt = datetime.strptime(d, DATE_FORMAT) + relativedelta(months=m)
    return dt if not ftstr else dt.strftime(DATE_FORMAT)


def date_to_season(date: str) -> int:
    return (int(date.replace("-01", "")[-2:]) - 1) // 3


def dates_to_seasons(dates: List[str]) -> List[int]:
    return [date_to_season(d) for d in dates]


def sort_paths_by_date(paths: List[str]) -> List[str]:
    strd_paths, _ = zip(
        *sorted([(p1, path_to_end_date(p1)) for p1 in paths], key=lambda x: x[1])
    )
    return strd_paths


def filter_paths_by_date(paths: List[str], dates: List[str]) -> List[str]:
    return [p for p in paths if path_to_end_date(p, ftstr=True) in dates]


class Constants:
    MAX_TIME_DIFFERENCE = 84
    TILE_SIZE = 145
    CHANNEL_MEANS = torch.tensor(
        [0.0707, 0.0919, 0.1545, 0.2256, 0.3408, 0.4111, 0.3152]
    )
    CHANNEL_STDS = torch.tensor(
        [0.0377, 0.0431, 0.0611, 0.0912, 0.0895, 0.1145, 0.1027]
    )
    # https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
    RGB_BANDS = [3, 2, 1]
    ADMIN_NAME_MAP = {  # map the names in the .csv file to the names in the .shp files
        "Baydhaba": "Baydhabo",
        "Bulo Burto": "Bulo-Burte",
        "Tayeeglow": "Tiyeglow",
        "Sheikh": "Sheekh",
        "Caynabo": "Caynaba",
        "Taleex": "Talex",
        "Burtinle": "Butinle",
        "Belet Weyne": "Beledweyn",
        "Lughaye": "Lughaya",
        "Rab Dhuure": "Rabdhuure",
        "Bandarbeyla": "Bandar Beyla",
        "Bossaso": "Bosaaso",
        "Adan Yabaal": "Aadan Yabaal",
        "Owdweyne": "Oodweyne",
        "Gebiley": "Gabiley",
        "Laasqoray": "Badhan",
        "Banadir": "Mogadishu",
        "Belet Xaawo": "Beled-Xaawo",
        "Galdogob": "Goldogob",
    }
    START_DATE = "2013-05-01"
    END_DATE_3M = "2015-11-01"
    END_DATE = "2020-03-01"
    NB_IPC_SCORES = (
        4  # actually 5 but the worst score does not occur in the considered period
    )
    IPC_CLASS_WEIGHTS = torch.tensor([0.09592509, 0.01822905, 0.08934612, 0.79649974])
    IPC_CLASS_WEIGHTS_BIN = torch.tensor([0.86588481, 0.13411519])


def binarize_ipcs(ipcs: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    """Expects input to be in [0, 4], not [1, 5] (as found in fews net csv)"""
    # copy works for np.ndarray and pd.Series
    binary_ipcs = ipcs.clone() if isinstance(ipcs, torch.Tensor) else ipcs.copy()
    binary_ipcs[ipcs <= 1] = 0
    binary_ipcs[ipcs > 1] = 1
    return binary_ipcs


def get_maj_class(ipcs: Tensor, return_binary=False) -> Tuple[Tensor, Optional[Tensor]]:
    _, ipc_counts = torch.unique(ipcs, return_counts=True)
    maj_class = ipc_counts.argmax()
    if return_binary:
        _, bin_counts = torch.unique(binarize_ipcs(ipcs), return_counts=True)
        bin_maj_class = bin_counts.argmax()
        return maj_class, bin_maj_class
    return maj_class


def tensor(array: Union[np.ndarray, Collection, Tensor]) -> Tensor:
    array = copy.deepcopy(array)
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    elif isinstance(array, Tensor):
        return array
    else:
        return torch.tensor(array)


def preds_to_cpu(pdict):
    return {
        k: (
            v.cpu()
            if isinstance(v, torch.Tensor)
            else (
                {
                    k2: (
                        v2.cpu()
                        if isinstance(v2, torch.Tensor)
                        else (
                            [e.cpu() if isinstance(e, torch.Tensor) else e for e in v2]
                            if isinstance(v2, List)
                            else v2
                        )
                    )
                    for k2, v2 in v.items()
                }
                if isinstance(v, Dict)
                else v
            )
        )
        for k, v in pdict.items()
    }


def plot_tile_loc(
    tiles: List,
    shape: Optional[gpd.GeoDataFrame] = None,
    tiles_name: Optional[str] = None,
    path_to_shp: Optional[str] = None,
    mark_regions: Optional[List[List[str]]] = None,
    mark_region_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    show: bool = True,
    save: bool = False,
):
    """
    utils.plot_tile_loc([utils.boxstr_to_tl_lat_lon(b) for b in val['boxes']],
                    path_to_shp='data/SO_Admin2_1990/SO_Admin2_1990.shp',
                    mark_regions=[ood['regions']], mark_region_labels=['Out-of-domain'], title='')
    """
    # https://geopandas.org/en/stable/docs/user_guide/mapping.html
    # https://geopandas.org/en/stable/gallery/create_geopandas_from_pandas.html
    plt.close()
    if shape is None:
        shape = gpd.read_file(path_to_shp)
    # logger.info('Converting shapefile CRS from %s to first tile CRS %s' % (shapefile.crs, first_file.crs))
    # shapefile.geometry = shapefile.geometry.to_crs(first_file.crs)
    fig, ax = plt.subplots(figsize=(5, 6))

    colors = ["white"]
    extra_colors = 0
    legend = ["Train"]

    colors += sns.color_palette("hls", extra_colors + 1)
    pmarks = []
    from matplotlib.patches import Patch

    if mark_regions:

        def region_fn(admin):
            for i, lst in enumerate(mark_regions):
                if admin in lst:
                    return i + 1
            return 0

        shape["mark_regions"] = shape.ADMIN2.apply(region_fn)

        extra_colors += max(shape["mark_regions"])
        legend += mark_region_labels

        for m, data in shape.groupby("mark_regions"):
            data.plot(ax=ax, edgecolor="black", color=colors[m])
            pmarks.append(
                Patch(facecolor=colors[m], label=legend[m], edgecolor="black")
            )

    if title:
        ax.set_title(title, fontsize=14)

    if tiles:
        lats, lons = zip(*tiles)
        df = pd.DataFrame({"lat": lats, "lon": lons})
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
        gdf.plot(ax=ax, color=colors[-1], markersize=2)
        legend += [tiles_name]
        pmarks.append(Patch(facecolor=colors[-1], label=legend[-1], edgecolor="black"))

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, *pmarks], loc="lower right", fontsize=14)

    plt.tight_layout()
    plt.axis("off")

    if save:
        plt.savefig("imgs/%s" % filename)
    if show:
        plt.show()
    plt.close()


def get_df_from_wandb(project="<organization>/SSSL-IPC"):
    api = wandb.Api(timeout=40)
    # Project is specified by <entity/project-name>
    runs = api.runs(project, per_page=3000)
    summary_list, config_list, name_list, time_list, dur_list, state_lst = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        # .name is the human-readable name of the run.
        name_list.append(run.name)
        time_list.append(run._attrs["createdAt"])
        dur_list.append(run._summary.get("_runtime", None))
        state_lst.append(run.state)

    _df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "createdAt": time_list,
            "runtime": dur_list,
            "state": state_lst,
        }
    )
    df = pd.concat(
        [
            _df.drop(["config", "summary"], axis=1),
            _df["config"].apply(pd.Series),
            _df["summary"].apply(pd.Series),
        ],
        axis=1,
    )
    return df
