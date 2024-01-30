import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

import dateutil.tz
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import wandb
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)

from sssl import utils

logger = logging.getLogger("__main__")


def admin_name_convert(name: str) -> str:
    return utils.Constants.ADMIN_NAME_MAP.get(name, name)


def run(args: argparse.Namespace, output_dir: str):
    for fut in (
        (0, 1, 2, 3)
        if args.iterate_future and args.temporally_separated
        else (args.n_steps_in_future,)
    ):
        logger.info("Running %s steps into the future" % fut)
        for bal in (
            ("balanced", None) if args.iterate_balanced else (args.class_weight,)
        ):
            logger.info(
                "Running with %s class weights" % {None: "unbalanced"}.get(bal, bal)
            )
            args.n_steps_in_future = fut
            args.class_weight = bal
            run_name = (
                args.run_name
                + ("_bal" if bal else "")
                + (f"_{fut}fut_after202003" if args.temporally_separated else "")
            )

            cfg = {f"random_forest/{k}": v for k, v in vars(args).items()}
            cfg.update({"cfg_name": run_name, "debug": args.debug})
            kwargs = {
                "name": output_dir.replace("output/", "").replace(
                    args.run_name, run_name
                ),
                "entity": "",  # todo
                "project": "SSSL-IPC",
                "dir": output_dir,
                "mode": "online" if not args.debug else "offline",
                "save_code": True,
                "tags": [str(args.seed), "RF"],
                "config": cfg,
            }
            wandb.init(**kwargs)

            inp_feat_names = [
                "centx",
                "centy",
                "area",
                "pop",
                "ruggedness_mean",
                "pasture_pct",
                "cropland_pct",
                "ndvi_mean",
                "rain_mean",
                "et_mean",
                "p_staple_food",
                "acled_count",
                "ndvi_anom",
                "rain_anom",
                "et_anom",
                "acled_fatalities",
            ]

            logger.info("Reading csv")
            df = pd.read_csv(args.food_crises_csv)
            df.admin_name = df.admin_name.apply(
                lambda x: utils.Constants.ADMIN_NAME_MAP.get(x, x)
            )

            if not args.include_05_2013:
                df = df[df.ymd > "2013-05-01"]

            train_dates = val_dates = test_dates = all_dates = df.sort_values("ymd")[
                "ymd"
            ].unique()

            logger.info("Reading split dict")
            with open(args.split_dict) as f:
                splits = json.load(f)
            with open(args.dicts) as f:
                dicts = json.load(f)

            if args.temporally_separated:
                shp = gpd.read_file(args.future_shp)
                fut_df = pd.DataFrame(
                    data={
                        "admin_name": shp.ADMIN2,
                        "fews_ipc": shp.CS,
                        # "year_month": shp.report_mon,
                    }
                )
                col = shp.report_mon.apply(
                    lambda x: (
                        datetime.strptime(x, "%m-%Y") + relativedelta(months=1)
                    ).strftime("%Y-%m-%d")
                )
                fut_df = fut_df.assign(ymd=col.values)
                df = pd.concat([df, fut_df], ignore_index=True)

                fut_date = fut_df["ymd"].unique()[0]
                # 2 -2
                # 2 -2
                # 1 -3
                # 0 -4
                train_dates = all_dates[min(3 - fut, 2) : -max(fut + 1, 2)]
                val_dates = [all_dates[-max(fut + 1, 2)]]
                test_dates = [all_dates[-max(fut, 1)]]

                if args.n_steps_in_future > 0:
                    df["fews_ipc_backup"] = df["fews_ipc"].copy()
                    df["fews_ipc"] = (
                        df.sort_values("ymd")
                        .groupby("admin_name")["fews_ipc"]
                        .shift(args.n_steps_in_future)
                    )
                    df = df.dropna(subset=["fews_ipc"])

                # define masks based on time
                ood_reg_mask = df.admin_name.isin(splits["ood_regions"])
                test_mask = df.ymd.isin(test_dates) & ~ood_reg_mask
                val_mask = df.ymd.isin(val_dates) & ~ood_reg_mask

            else:
                # define masks based on geography
                ood_reg_mask = df.admin_name.isin(splits["ood_regions"])
                test_mask = df.admin_name.isin(splits["test_regions"]) & ~ood_reg_mask
                val_mask = df.admin_name.isin(splits["val_regions"]) & ~ood_reg_mask

            ood_mask = ood_reg_mask & df.ymd.isin(test_dates)

            if args.sssl_predictions != "":
                preds_df, col_names = get_preds_df(args, dicts)
                df = df.merge(preds_df, on=["admin_name", "ymd"], how="left")
                logger.info(
                    "Dropping %s rows that do not have an ipc prediction"
                    % df.dropna(subset=col_names).shape[0]
                )
                df = df.dropna(subset=col_names)
                inp_feat_names += col_names

            # [1, 5] (5 does not occur but it could)
            Y = df.fews_ipc.to_numpy(dtype=int)
            Y -= 1  # [0, 4]
            # https://openknowledge.worldbank.org/handle/10986/34510 page 4
            Y_bin = utils.binarize_ipcs(Y)
            X = df[inp_feat_names]

            logger.info("Splitting data into train/val/test/ood")
            training_data = {}
            for perc in (100, 70, 50, 20, 5, 1) if args.iterate_percentages else (100,):
                if perc == 100 and args.temporally_separated:
                    mask = df.ymd.isin(train_dates) & ~ood_reg_mask
                    regs = [
                        n
                        for n in df.admin_name.unique()
                        if n not in splits["ood_regions"]
                    ]
                else:
                    regs = [
                        admin_name_convert(n) for n in splits[f"train_regions_{perc}"]
                    ]
                    mask = df.admin_name.isin(regs)
                training_data[perc] = (X[mask], Y[mask], Y_bin[mask], regs)

            df_tst = df[test_mask]
            X_tst, Y_tst = (X[test_mask], Y[test_mask])
            Y_tst_bin = Y_bin[test_mask]
            df_val = df[val_mask]
            X_val, Y_val = (X[val_mask], Y[val_mask])
            Y_val_bin = Y_bin[val_mask]
            df_ood = df[ood_mask]
            X_ood, Y_ood = X[ood_mask], Y[ood_mask]
            Y_ood_bin = Y_bin[ood_mask]

            output = {}
            for perc, (X_tr, Y_tr, Y_tr_bin, train_regions) in training_data.items():
                logger.info(
                    "Running for %s of data, regions: %s" % (perc, train_regions)
                )

                logger.info("Defining classifiers")
                clf = build_rf_clf(args)
                clf_bin = build_rf_clf(args)

                logger.info("Fitting classifiers")
                clf.fit(X_tr, Y_tr)
                clf_bin.fit(X_tr, Y_tr_bin)

                # check https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation
                #  for feature importance evaluation

                logger.info("Making predictions")
                pred_tst = clf.predict(X_tst)
                pred_tst_bin = clf_bin.predict(X_tst)
                pred_val = clf.predict(X_val)
                pred_val_bin = clf_bin.predict(X_val)
                pred_ood = clf.predict(X_ood)
                pred_ood_bin = clf_bin.predict(X_ood)

                logger.info("Computing majority class prediction baselines")
                _, counts = np.unique(Y_tr, return_counts=True)
                maj_class = counts.argmax()
                _, bin_counts = np.unique(Y_tr_bin, return_counts=True)
                bin_maj_class = bin_counts.argmax()

                logger.info("Computing metrics and gathering predictions")
                result_dict = {}
                preds_dict = {}
                for (name, preds, tgt, df_eval, num_classes, maj_cl) in (
                    (
                        "test",
                        pred_tst,
                        Y_tst,
                        df_tst,
                        utils.Constants.NB_IPC_SCORES,
                        maj_class,
                    ),
                    (
                        "val",
                        pred_val,
                        Y_val,
                        df_val,
                        utils.Constants.NB_IPC_SCORES,
                        maj_class,
                    ),
                    (
                        "ood",
                        pred_ood,
                        Y_ood,
                        df_ood,
                        utils.Constants.NB_IPC_SCORES,
                        maj_class,
                    ),
                    ("test_bin", pred_tst_bin, Y_tst_bin, df_tst, 2, bin_maj_class),
                    ("val_bin", pred_val_bin, Y_val_bin, df_val, 2, bin_maj_class),
                    ("ood_bin", pred_ood_bin, Y_ood_bin, df_ood, 2, bin_maj_class),
                ):
                    for ipc, row in zip(preds, df_eval.to_dict(orient="records")):
                        preds_dict.setdefault(name, {}).setdefault(
                            row["admin_name"], {}
                        )[row["ymd"]] = int(ipc)

                    preds, tgt = torch.from_numpy(preds), torch.from_numpy(tgt)

                    # remap in case some labels don't occur in preds/tgt
                    pred_bins = (
                        torch.bincount(preds, minlength=num_classes) / preds.shape[0]
                    )
                    gt_bins = torch.bincount(tgt, minlength=num_classes) / tgt.shape[0]
                    cum_tags = (gt_bins + pred_bins).gt(0).long().cumsum(-1) - 1
                    mapped_preds = torch.tensor([cum_tags[v] for v in preds])
                    mapped_tgt = torch.tensor([cum_tags[v] for v in tgt])
                    mapped_nc = int(cum_tags.max()) + 1

                    maj_class_preds = torch.full(preds.shape, fill_value=maj_cl)
                    for avg in ("weighted", "micro", "macro"):

                        result_dict.update(
                            {
                                f"{name}_f1_{avg}": multiclass_f1_score(
                                    preds, tgt, num_classes=num_classes, average=avg
                                ).item(),
                                f"{name}_acc_{avg}": multiclass_accuracy(
                                    preds, tgt, num_classes=num_classes, average=avg
                                ).item(),
                                f"{name}_f1_{avg}_mapped": multiclass_f1_score(
                                    mapped_preds,
                                    mapped_tgt,
                                    num_classes=mapped_nc,
                                    average=avg,
                                ).item()
                                if mapped_nc > 1
                                else 1.0,
                                f"{name}_acc_{avg}_mapped": multiclass_accuracy(
                                    mapped_preds,
                                    mapped_tgt,
                                    num_classes=mapped_nc,
                                    average=avg,
                                ).item()
                                if mapped_nc > 1
                                else 1.0,
                                **(
                                    compute_changed_in_future(
                                        df_eval, preds, tgt, name, avg, num_classes
                                    )
                                    if fut > 0
                                    else {}
                                ),
                                f"{name}_maj_baseline_f1_{avg}": multiclass_f1_score(
                                    maj_class_preds,
                                    tgt,
                                    num_classes=num_classes,
                                    average=avg,
                                ).item(),
                                f"{name}_maj_baseline_acc_{avg}": multiclass_accuracy(
                                    maj_class_preds,
                                    tgt,
                                    num_classes=num_classes,
                                    average=avg,
                                ).item(),
                                f"{name}_maj_baseline_f1_{avg}_mapped": multiclass_f1_score(
                                    maj_class_preds,
                                    tgt,
                                    num_classes=mapped_nc,
                                    average=avg,
                                ).item()
                                if mapped_nc > 1
                                else 1.0,
                                f"{name}_maj_baseline_acc_{avg}_mapped": multiclass_accuracy(
                                    maj_class_preds,
                                    tgt,
                                    num_classes=mapped_nc,
                                    average=avg,
                                ).item()
                                if mapped_nc > 1
                                else 1.0,
                            }
                        )
                output[perc] = {"predictions": preds_dict, "results": result_dict}
                logger.info("Results for %s%% of data: " % perc)
                logger.info(json.dumps(result_dict, indent=2))
                wandb.log(
                    {
                        "%s%s" % (k, f"_{perc}" if perc != 100 else ""): v
                        for k, v in result_dict.items()
                    }
                )

            logger.info("Gathering ground-truths")
            zone_date_2_ipc = {}
            for ipc, row in zip(Y, df.to_dict(orient="records")):
                zone_date_2_ipc.setdefault(row["admin_name"], {})[row["ymd"]] = int(ipc)
            output["gold"] = zone_date_2_ipc

            logger.info(
                "Saving results to %s"
                % os.path.join(output_dir, f"output_{run_name}.json")
            )
            with open(os.path.join(output_dir, f"output_{run_name}.json"), "w") as f:
                json.dump(output, f)
            logger.info("Done")

            wandb.finish()
            time.sleep(5)


def compute_changed_in_future(df_eval, preds, tgt, split, avg, num_classes):
    nc = num_classes

    changed_mask = df_eval["fews_ipc"] != df_eval["fews_ipc_backup"]
    changed_mask = changed_mask.to_numpy()
    changed_tgt = tgt[changed_mask]
    changed_preds = preds[changed_mask]

    # remap in case some labels don't occur in preds/tgt
    pred_bins = (
        torch.bincount(changed_preds, minlength=nc).cpu() / changed_preds.shape[0]
    )
    gt_bins = torch.bincount(changed_tgt, minlength=nc).cpu() / changed_tgt.shape[0]
    cum_tags = (gt_bins + pred_bins).gt(0).long().cumsum(-1) - 1
    mapped_nc = int(cum_tags.max()) + 1
    mapped_preds = torch.tensor([cum_tags[v] for v in changed_preds])
    mapped_tgt = torch.tensor([cum_tags[v] for v in changed_tgt])

    result_dict = {
        f"{split}_f1_{avg}_changed": multiclass_f1_score(
            changed_preds, changed_tgt, num_classes=nc, average=avg
        ).item(),
        f"{split}_acc_{avg}_changed": multiclass_accuracy(
            changed_preds, changed_tgt, num_classes=nc, average=avg
        ).item(),
        f"{split}_f1_{avg}_changed_mapped": multiclass_f1_score(
            mapped_preds, mapped_tgt, num_classes=mapped_nc, average=avg
        ).item()
        if mapped_nc > 1
        else 1.0,
        f"{split}_acc_{avg}_changed_mapped": multiclass_accuracy(
            mapped_preds, mapped_tgt, num_classes=mapped_nc, average=avg
        ).item()
        if mapped_nc > 1
        else 1.0,
    }
    return result_dict


def get_preds_df(args, dicts):
    sssl_preds = torch.load(args.sssl_predictions, map_location=torch.device("cpu"))
    if args.use_sssl_preds_as == "maj_votes":
        preds_df = pd.DataFrame(
            [
                (dicts["all_admins"][a], dicts["all_end_dates"][d], pred_ipc)
                for pred_dict in sssl_preds.values()
                for (a, d), pred_ipc in pred_dict["zone_date_2_maj_vote"].items()
            ],
            columns=["admin_name", "ymd", "nn_ipc"],
        )
        col_names = ["nn_ipc"]
    else:
        assert args.use_sssl_preds_as == "scores"

        def logits_2_votes(logits):
            votes = torch.stack(logits).argmax(dim=-1).bincount(minlength=4)
            return votes / votes.sum()

        col_names = ["nn_v1", "nn_v2", "nn_v3", "nn_v4"]
        preds_df = pd.DataFrame(
            data=[
                (
                    dicts["all_admins"][z],
                    dicts["all_end_dates"][d],
                    *logits_2_votes(logits).tolist(),
                )
                for pred_dict in sssl_preds.values()
                for ((z, d), logits) in pred_dict["zone_date_2_logits"].items()
            ],
            columns=["admin_name", "ymd", *col_names],
        )

    return preds_df, col_names


def build_rf_clf(args):
    clf = RandomForestClassifier(
        bootstrap=True,
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        max_depth=args.max_depth,
        max_leaf_nodes=args.max_leaf_nodes,
        class_weight=args.class_weight,
    )
    logger.info("Defined RandomForestClassifier")
    logger.info(clf)
    return clf


def define_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--food_crises_csv",
        type=str,
        default="data/predicting_food_crises_data_somalia_from2013-05-01.csv",
    )
    parser.add_argument(
        "--split_dict",
        type=str,
        default="data/indices6_/downstr_splits_incl_small.json",
    )
    parser.add_argument("--dicts", type=str, default="data/indices/dicts.json")
    #     future_ipc_shp: List[str] = [
    #         "data/SO_202006/SO_202006_CS.shp",
    #         "data/SO_202010/SO_202010_CS.shp",
    #         "data/SO_202102/SO_202102_CS.shp",  # don't use, all test data has same ipc score
    #     ]
    parser.add_argument(
        "--future_shp", type=str, default="data/SO_202006/SO_202006_CS.shp"
    )

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--min_samples_split", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=3)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--max_leaf_nodes", type=int, default=None)

    parser.add_argument("--n_steps_in_future", type=int, default=0)
    parser.add_argument("--temporally_separated", action="store_true")
    parser.add_argument(
        "--use_sssl_preds_as",
        type=str,
        default="scores",
        choices=["scores", "maj_votes"],
    )
    parser.add_argument("--sssl_predictions", type=str, default="")

    # use "balanced" for better macro scores, None for better weighted/micro
    parser.add_argument("--class_weight", type=str, default="balanced")
    parser.add_argument(
        "--run_name",
        type=str,
        default="random_forest_tempsep_changed",
    )

    # To include training and eval on first IPC score date (which has no corresponding tiles in the NN dataset)
    parser.add_argument("--include_05_2013", action="store_true")
    parser.add_argument("--iterate_future", action="store_true")
    parser.add_argument("--iterate_balanced", action="store_true")
    parser.add_argument("--iterate_percentages", action="store_true")
    return parser


if __name__ == "__main__":
    parser = define_args()
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.run_name += f"_s{args.seed}"

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = "output/%s_%s" % (timestamp, args.run_name)

    utils.mkdir_p(output_dir)
    utils.initialize_logging(output_dir, to_file=True, logger_name="__main__")
    logger.info(vars(args))

    run(args, output_dir)
