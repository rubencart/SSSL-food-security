import datetime
import json
import random
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import yaml
from dateutil.relativedelta import relativedelta
from matplotlib import cm
from matplotlib import pyplot as plt
from sssl import utils
from sssl.config import Config
from sssl.data.ipc import IPCScoreDataset
from tqdm import tqdm


def plot_tiles_per_ipc():
    for ipc, zds in tqdm(ipc_ds.ipc_2_zd.items()):
        for z, d in tqdm(random.sample(zds, k=len(zds))[:5]):
            box_dict: Dict[str, List[str]] = ipc_ds.f.zone2box2p[ipc_ds.f.all_admins[z]]
            box_list = random.sample(list(box_dict.values()), k=len(box_dict))[:2]
            paths: List[str] = [pathlist[d] for pathlist in box_list]
            for p in tqdm(paths):
                p_cor = Path(cfg.tiles_dir) / p
                ds = rio.open(p_cor)
                # 1.3 * un_img ** (1 / 1.6)
                array = ((np.nan_to_num(ds.read()) ** (1 / 1.6)) * 1.3 * 256).astype(np.uint8)
                bounds = ds.bounds
                plt.close()
                plt.imshow(array[utils.Constants.RGB_BANDS].transpose(1, 2, 0))
                plt.tight_layout()
                plt.savefig(
                    f"imgs/tiles/tile_lon{bounds.left:.2f}_lat{bounds.top:.2f}_ipc{ipc}.pdf"
                )
                plt.show()
                plt.close()


if __name__ == "__main__":
    with open("config/ipc/debug.yaml", "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Config()
    cfg.from_dict(dict_cfg)
    cfg.process_args()
    cfg.finetune.temporally_separated = False

    ipc_ds = IPCScoreDataset(cfg, "val")

    cmap = cm.get_cmap("Spectral")
    df = pd.read_csv("data/predicting_food_crises_data.csv")

    col = df["year_month"].apply(
        lambda x: (datetime.datetime.strptime(x, "%Y_%m") + relativedelta(months=1)).strftime(
            "%Y-%m-%d"
        )
    )
    df = df.assign(ymd=col.values)

    # plot_tiles_per_ipc()

    import seaborn as sns

    colors = sns.color_palette("rocket", 5)
    df.loc[df["country"] == "Democratic Republic of Congo", "country"] = "DR of Congo"

    lfs = 17
    plt.close()
    df[df["ymd"] >= "2013-08-01"].groupby(["country"])["fews_ipc"].value_counts(
        normalize=True
    ).unstack().plot(
        kind="bar", stacked=True, cmap=cmap, figsize=(10, 6), color=colors, fontsize=lfs
    )
    for col, lab in zip(colors, [1, 2, 3, 4, 5]):
        plt.axvspan(0, 0, fc=col, label=lab)
    legend = plt.legend([1, 2, 3, 4, 5], title="IPC", bbox_to_anchor=(1.02, 1), fontsize=lfs)
    legend.get_title().set_fontsize(lfs)
    plt.yticks(fontsize=lfs - 1)
    plt.xticks(fontsize=lfs + 1)
    plt.xlabel("Country", fontsize=lfs)
    plt.ylabel("Frequency", fontsize=lfs)
    plt.title("IPC distribution 2013-2020", fontsize=lfs + 1)
    plt.tight_layout()
    plt.savefig(f"imgs/ipc_distrib_2013.pdf")
    plt.show()
    plt.close()

    plt.close()
    df.groupby(["country"])["fews_ipc"].value_counts(normalize=True).unstack().plot(
        kind="bar", stacked=True, cmap=cmap, figsize=(10, 6), color=colors, fontsize=lfs
    )
    for col, lab in zip(colors, [1, 2, 3, 4, 5]):
        plt.axvspan(0, 0, fc=col, label=lab)
    legend = plt.legend([1, 2, 3, 4, 5], title="IPC", bbox_to_anchor=(1.02, 1), fontsize=lfs)
    legend.get_title().set_fontsize(lfs)
    plt.yticks(fontsize=lfs - 1)
    plt.xticks(fontsize=lfs + 1)
    plt.xlabel("Country", fontsize=lfs)
    plt.ylabel("Frequency", fontsize=lfs)
    plt.title("IPC distribution 2009-2020", fontsize=lfs + 1)
    plt.tight_layout()
    plt.savefig(f"imgs/ipc_distrib_2009.pdf")
    plt.show()
    plt.close()

    plt.close()
    df[df["country"] == "Somalia"].groupby(["year"])["fews_ipc"].value_counts(
        normalize=True
    ).unstack().plot(
        kind="bar",
        stacked=True,
        cmap=cmap,
        figsize=(7, 6),
        color=colors,
        fontsize=lfs,
    )
    for col, lab in zip(colors, [1, 2, 3, 4, 5]):
        plt.axvspan(0, 0, fc=col, label=lab)
    plt.yticks(fontsize=lfs - 1)
    plt.xticks(fontsize=lfs + 1)
    plt.xlabel("Year", fontsize=lfs)
    plt.ylabel("Frequency", fontsize=lfs)
    plt.title("IPC distribution Somalia 2009-2020", fontsize=lfs + 1)
    legend = plt.legend([1, 2, 3, 4, 5], title="IPC", bbox_to_anchor=(1.02, 1), fontsize=lfs)
    legend.get_title().set_fontsize(lfs)
    plt.tight_layout()
    plt.savefig(f"imgs/ipc_distrib_somalia_2009_2.pdf")
    plt.show()
    plt.close()

    plt.close()
    df[(df["country"] == "Somalia") & (df["ymd"] >= "2013-08-01")].groupby(["year"])[
        "fews_ipc"
    ].value_counts(normalize=True).unstack().plot(
        kind="bar",
        stacked=True,
        cmap=cmap,
        figsize=(7, 6),
        color=colors,
        fontsize=lfs,
    )
    for col, lab in zip(colors, [1, 2, 3, 4, 5]):
        plt.axvspan(0, 0, fc=col, label=lab)
    plt.yticks(fontsize=lfs - 1)
    plt.xticks(fontsize=lfs + 1)
    plt.xlabel("Year", fontsize=lfs)
    plt.ylabel("Frequency", fontsize=lfs)
    plt.title("IPC distribution Somalia 2013-2020", fontsize=lfs + 1)
    legend = plt.legend([1, 2, 3, 4], title="IPC", bbox_to_anchor=(1.02, 1), fontsize=lfs)
    legend.get_title().set_fontsize(lfs)
    plt.tight_layout()
    plt.savefig(f"imgs/ipc_distrib_somalia_2013_2.pdf")
    plt.show()
    plt.close()

    ood_splits_path: str = "data/indices/ood_splits.json"
    val_splits_path: str = "data/indices/val_splits.json"
    test_splits_path: str = "data/indices/test_splits.json"
    train_splits_path: str = "data/indices/train_splits.json"
    downstr_splits_path: str = "data/indices/downstr_splits_incl_small.json"

    with open(ood_splits_path, "r") as f:
        ood = json.load(f)
    with open(val_splits_path, "r") as f:
        val = json.load(f)
    with open(test_splits_path, "r") as f:
        test = json.load(f)
    with open(train_splits_path, "r") as f:
        train = json.load(f)
    with open(downstr_splits_path, "r") as f:
        downstr = json.load(f)

    shp = "data/SO_Admin2_1990/SO_Admin2_1990.shp"
    shapefile = gpd.read_file(shp)

    utils.plot_tile_loc(
        tiles=[utils.boxstr_to_tl_lat_lon(b) for b in val["boxes"]],
        tiles_name="Validation",
        path_to_shp="data/SO_Admin2_1990/SO_Admin2_1990.shp",
        # shape=shapefile,
        mark_regions=[ood["regions"]],
        mark_region_labels=["Out-of-domain"],
        title="Pre-train splits",
        save=True,
        filename="splits_pretrain.pdf",
    )

    utils.plot_tile_loc(
        tiles=[],
        tiles_name="",
        path_to_shp="data/SO_Admin2_1990/SO_Admin2_1990.shp",
        # shape=shapefile,
        mark_regions=[ood["regions"], downstr["val_regions"], downstr["test_regions"]],
        mark_region_labels=["Out-of-domain", "Validation", "In-domain test"],
        title="Downstream splits",
        save=True,
        filename="splits_downstream.pdf",
    )
