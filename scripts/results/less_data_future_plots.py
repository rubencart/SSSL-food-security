import itertools
from datetime import datetime

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sssl import utils

RUNS = [
    "sssl_resnet18_t1_s04_ALL",
    "tile2vec_resnet18_t1_s015_ALL",
    "relres_resnet18_ALL",
]

NAME_DICT = {
    "ImageNet": "ImageNet",
    "random": "Random init.",
    "random_forest_bal": "Random forest",
    "sssl_resnet18_t1_s04_ALL": "SSSL",
    "tile2vec_resnet18_t1_s015_ALL": "Tile2Vec",
    "relres_resnet18_ALL": "Data aug.",
}


def data_plot(
    df,
    groupby_cols,
    title,
    metric,
    m_name,
    rf=None,
    maj_baseline=None,
    frozen_dashed=True,
    legend=True,
):
    plt.close()
    # fig, ax = plt.subplots(figsize=(8, 6))
    fig = plt.figure(figsize=(9 if legend else 7, 6))
    ax = fig.gca()

    groups = df.groupby(groupby_cols, dropna=False)
    colors = sns.color_palette("hls", len(groups) + 2)
    ax.set_prop_cycle("color", colors)
    markers = itertools.cycle(("o", "v", "^", ">", "<", "s", "P", "d"))
    ticks = [100, 70, 50, 20, 5, 1]

    for (pt_on, relres, cfg_name), gdf in groups:
        color = next(ax._get_lines.prop_cycler)["color"]
        marker = next(markers)
        frz_groups = gdf.groupby(["finetune/freeze_backbone"], dropna=False)

        for frz, fdf in frz_groups:
            fdf = fdf.sort_values(
                "finetune/percentage_of_training_data", ascending=False
            )
            print(fdf[["name", "finetune/percentage_of_training_data"]])
            style = "-" if (not frz or not frozen_dashed) else "--"
            fdf.plot(
                x="finetune/percentage_of_training_data",
                y=metric,
                ax=ax,
                legend=False,
                color=color,
                style=style,
                label=NAME_DICT[cfg_name if pd.notna(cfg_name) else pt_on]
                + (f' {"frz" if frz else "ft"}' if frozen_dashed else ""),
                marker=marker,
            )

    if rf is not None:
        color = next(ax._get_lines.prop_cycler)["color"]
        for rec in rf:
            ax.plot(
                ticks,
                rec["scores"],
                color=color,
                linestyle="--" if pd.isna(rec["random_forest/class_weight"]) else "-",
                marker=next(markers),
                label="Random forest"
                if pd.isna(rec["random_forest/class_weight"])
                else "Random forest",
            )

    if maj_baseline is not None:
        plt.axhline(
            y=maj_baseline,
            linestyle="dotted",
            label="Maj. baseline",
        )

    plt.xticks(ticks, ticks, fontsize=14)
    plt.xlabel("Percentage of training data", fontsize=16)

    if not legend:
        plt.yticks(fontsize=14)
        plt.ylabel(m_name, fontsize=16)
    else:
        plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=6 * [""])
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=15)

    plt.ylim(0.0, 1.0)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig("imgs/%s_lrg.pdf" % title.replace(" ", "_").lower())
    plt.show()
    plt.close()


def fut_plot(
    df,
    groupby_cols,
    title,
    metric,
    m_name,
    rf=None,
    rf_metric=None,
    maj_baseline=None,
    frozen_dashed=True,
    legend=True,
):
    plt.close()
    fig = plt.figure(figsize=(9 if legend else 7, 6))
    ax = fig.gca()

    groups = df.groupby(groupby_cols, dropna=False)
    colors = sns.color_palette("hls", len(groups) + 2)
    ax.set_prop_cycle("color", colors)
    markers = itertools.cycle(("o", "v", "^", ">", "<", "s", "P", "d"))
    ticks = [0, 1, 2, 3]

    for (pt_on, relres, cfg_name), gdf in groups:
        color = next(ax._get_lines.prop_cycler)["color"]
        marker = next(markers)
        frz_groups = gdf.groupby(["finetune/freeze_backbone"], dropna=False)

        for frz, fdf in frz_groups:
            fdf = fdf.sort_values("finetune/n_steps_in_future", ascending=False)
            print(fdf[["name", "finetune/n_steps_in_future"]])
            style = "-" if (not frz or not frozen_dashed) else "--"
            fdf.plot(
                x="finetune/n_steps_in_future",
                y=metric,
                ax=ax,
                legend=False,
                color=color,
                style=style,
                label=NAME_DICT[cfg_name if pd.notna(cfg_name) else pt_on]
                + (f' {"frz" if frz else "ft"}' if frozen_dashed else ""),
                # label=f'{cfg_name if pd.notna(cfg_name) else pt_on}_{"frz" if frz else "ft"}',
                marker=marker,
            )

    if rf is not None:
        color = next(ax._get_lines.prop_cycler)["color"]
        for bal, rf_df in rf:
            rf_df.plot(
                x="random_forest/n_steps_in_future",
                y=rf_metric,
                ax=ax,
                legend=False,
                color=color,
                style="--" if pd.isna(bal) else "-",
                # label='random_forest/run_name',
                label="Random forest" if not pd.isna(bal) else "Random forest",
                marker=next(markers),
            )

    if maj_baseline is not None:
        plt.axhline(
            y=maj_baseline,
            # color=colors[-1],
            linestyle="dotted",
            label="Maj. baseline",
        )

    if not legend:
        plt.yticks(fontsize=14)
        plt.ylabel(m_name, fontsize=16)
    else:
        plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=6 * [""])
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=15)

    plt.xticks(ticks, ticks, fontsize=14)
    plt.xlabel("N steps into the future", fontsize=16)
    plt.ylim(0.0, 1.0)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    fn = "imgs/%s_after202003_mapped_lrg.pdf" % title.replace(" ", "_").lower()
    print("Saving to %s" % fn)
    plt.savefig(fn)
    plt.show()
    plt.close()


def run():
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 500)

    df = utils.get_df_from_wandb()
    run_df = preprocess_df(df)

    groupby_cols = ["finetune/pretrained_on", "pretrain_relres", "pretrain_cfg_name"]
    metrics = [
        "test_id_maj_vote_f1_macro_mapped",
        "test_ood_maj_vote_f1_macro_mapped",
        # "test_id_maj_vote_f1_macro_changed_mapped",
        # "test_ood_maj_vote_f1_macro_changed_mapped",
        "test_id_maj_vote_f1_macro",
        "test_ood_maj_vote_f1_macro",
        # "test_id_maj_vote_f1_weighted",
        # "test_ood_maj_vote_f1_weighted",
    ]
    rf_metrics = [
        "test_f1_macro_mapped",
        "ood_f1_macro_mapped",
        # "test_f1_macro_changed_mapped",
        # "ood_f1_macro_changed_mapped",
        "test_f1_macro",
        "ood_f1_macro",
        # "test_f1_weighted",
        # "ood_f1_weighted",
    ]
    rf_maj_baseline_metrics = [
        "test_maj_baseline_f1_macro",
        "ood_maj_baseline_f1_macro",
        "test_maj_baseline_f1_macro",
        "ood_maj_baseline_f1_macro",
        "test_maj_baseline_f1_weighted",
        "ood_maj_baseline_f1_weighted",
    ]
    dset_names = 2 * ["Test", "OOD"]
    metric_names = [
        "Macro F1",
        "Macro F1",
        "Macro F1",
        "Macro F1",
        # "Macro F1 on changed zones",
        # "Macro F1 on changed zones",
        # "Weighted F1",
        # "Weighted F1",
        # "Accuracy",
        # "Accuracy",
    ]
    rf_run_names = [
        "2023_09_20_23_49_21_random_forest_tempsep_changed_s42_bal",  # future
        "2023_09_20_23_49_21_random_forest_tempsep_changed_s42_bal",  # future
        "2023_02_10_16_02_59_random_forest_s42_bal",  # decreasing data
        "2023_02_10_16_02_59_random_forest_s42_bal",  # decreasing data
    ]

    for metric, rf_metric, rf_mb_metric, dset_name, m_name in zip(
        metrics, rf_metrics, rf_maj_baseline_metrics, dset_names, metric_names
    ):

        rf_df = df[df.name.str.contains(rf_run_names[-1])]
        maj_bas_score, records = get_rf_data_df(
            rf_df, rf_mb_metric, rf_metric, rf_run_names
        )

        # maj_bas_score = rf_df[rf_df["random_forest/n_steps_in_future"] == 0][
        #     "test_maj_baseline_f1_macro"
        # ].item()

        fltrd_data_df = get_data_df(groupby_cols, metric, run_df)

        for frz in (True, False):
            data_plot(
                fltrd_data_df[fltrd_data_df["finetune/freeze_backbone"] == frz],
                groupby_cols,
                f'Training set size vs {dset_name} {m_name} with {"frozen" if frz else "unfrozen"} weights',
                metric,
                m_name,
                rf=records,
                maj_baseline=maj_bas_score,
                frozen_dashed=False,
                legend=not frz,
            )

        fltrd_fut_df = get_fut_df(groupby_cols, metric, run_df)
        rf_fut_grps = get_rf_fut_grps(rf_df)

        for frz in (True, False):
            fut_plot(
                fltrd_fut_df[fltrd_fut_df["finetune/freeze_backbone"] == frz],
                groupby_cols,
                f'Future {dset_name} {m_name} with {"frozen" if frz else "unfrozen"} weights',
                metric,
                m_name,
                rf=rf_fut_grps,
                rf_metric=rf_metric,
                maj_baseline=maj_bas_score,
                frozen_dashed=False,
                legend=not frz,
            )
    print("ok")


def get_rf_fut_grps(rf_df):
    rf_fut_df = rf_df.dropna(subset=["random_forest/n_steps_in_future"])
    rf_fut_df = rf_fut_df[
        pd.isna(rf_fut_df["random_forest/sssl_predictions"])
        | (rf_fut_df["random_forest/sssl_predictions"] == "")
    ]
    rf_fut_df = rf_fut_df[pd.notna(rf_fut_df["random_forest/class_weight"])]
    rf_fut_df = rf_fut_df.sort_values("random_forest/n_steps_in_future")
    rf_fut_grps = rf_fut_df.groupby(["random_forest/class_weight"], dropna=False)
    return rf_fut_grps


def get_rf_data_df(rf_df, rf_mb_metric, rf_metric, rf_run_names):
    rf_data_df = rf_df[rf_df.name.isin(rf_run_names)]
    # make sure not in future
    maj_bas_score = rf_data_df[pd.notna(rf_data_df["random_forest/class_weight"])][
        rf_mb_metric
    ].iloc[0]
    rf_data_ms = [
        "%s%s" % (rf_metric, f"_{perc}" if perc != 100 else "")
        for perc in (100, 70, 50, 20, 5, 1)
    ]
    records = rf_data_df[
        ["random_forest/run_name", "random_forest/class_weight"] + rf_data_ms
    ].to_dict(orient="records")
    records = [{**d, "scores": [d[m] for m in rf_data_ms]} for d in records]
    return maj_bas_score, records


def get_fut_df(groupby_cols, metric, run_df):
    all_data_mask = pd.isna(run_df["finetune/percentage_of_training_data"]) | (
        run_df["finetune/percentage_of_training_data"] == 100
    )
    fut_df = run_df[all_data_mask].copy()
    fut_df.loc[
        pd.isna(fut_df["finetune/n_steps_in_future"]), "finetune/n_steps_in_future"
    ] = 0
    temp_sep_mask = fut_df["finetune/temporally_separated"] == True
    best_mask = fut_df["finetune/all_backbone_ckpts_in_dir"].isin(["", None])
    best_fut_df = fut_df[best_mask & temp_sep_mask]
    # filter out doubles
    groupby_cols_frz = groupby_cols + [
        "finetune/freeze_backbone",
        "finetune/n_steps_in_future",
    ]
    fltrd_fut_df = best_fut_df.sort_values(
        by=["finetune/n_steps_in_future", "createdAt"], ascending=[False, False]
    ).drop_duplicates(groupby_cols_frz)
    return fltrd_fut_df


def get_data_df(groupby_cols, metric, run_df, downstr_splits=None):
    not_fut_mask = (
        pd.isna(run_df["finetune/n_steps_in_future"])
        | (run_df["finetune/n_steps_in_future"] == 0)
    ) & (
        pd.isna(run_df["finetune/temporally_separated"])
        | (run_df["finetune/temporally_separated"] is False)
    )
    data_df = run_df[not_fut_mask].copy()
    if downstr_splits:
        data_df = data_df[
            data_df["downstr_splits_path"]
            == f'downstr_splits_incl_small{"_v2" if downstr_splits == "v2" else ""}.json'
        ]
    best_mask = data_df["finetune/all_backbone_ckpts_in_dir"].isin(["", None])
    best_data_df = data_df[best_mask]
    best_pt_frz_df = data_df[~best_mask].loc[
        data_df[~best_mask]
        .groupby(groupby_cols + ["seed"], dropna=False)[metric]
        .idxmax()
    ]
    best_data_df = pd.concat([best_data_df, best_pt_frz_df])
    # filter out doubles
    groupby_cols_frz = groupby_cols + [
        "finetune/freeze_backbone",
        "finetune/percentage_of_training_data",
        "seed",
    ]
    fltrd_data_df = best_data_df.sort_values(
        by=["finetune/percentage_of_training_data", "createdAt"],
        ascending=[False, False],
    ).drop_duplicates(groupby_cols_frz)
    return fltrd_data_df


def preprocess_df(df, filter_runs=True):
    m1 = df["finetune/clf_head"] == "mlp"
    m2 = df["landsat8_bands"] == "ALL"
    m6 = pd.notna(df["test_id_maj_vote_f1_weighted"])

    min_date = datetime.strptime("2023-01-01", "%Y-%m-%d")  # todo adjust
    m8 = pd.to_datetime(df["createdAt"], format="%Y-%m-%dT%H:%M:%S") > min_date
    m12 = pd.isna(df["finetune/binarize_ipc"]) | (df["finetune/binarize_ipc"] == False)
    m14 = ~df.cfg_name.str.contains("_lr_")
    all_df = df.loc[m1 & m2 & m6 & m8 & m12 & m14].copy()
    all_df["pretrain_relres"] = (
        all_df["finetune/pretrained_ckpt_path"].str.contains("relres").astype("boolean")
    )
    all_df["pretrain_loss_type_sssl"] = (
        all_df["finetune/pretrained_ckpt_path"].str.contains("sssl").astype("boolean")
    )
    all_df["pretrain_cfg_name"] = all_df["finetune/pretrained_ckpt_path"].str.extract(
        r"output\/[0-9_]+s[0-9]+_([0-9a-zA-Z_]+)\/", expand=False
    )
    run_df = all_df[
        all_df["finetune/pretrained_on"].isin(["random", "ImageNet"])
        | (
            (all_df["finetune/pretrained_on"] == "own")
            & pd.notna(all_df["finetune/pretrained_ckpt_path"])
            # & all_df['finetune/pretrained_ckpt_path'].apply(lambda x: any(r in x for r in RUNS))
        )
    ]
    if filter_runs:
        run_df = run_df[
            run_df["finetune/pretrained_on"].isin(["random", "ImageNet"])
            | (
                (run_df["finetune/pretrained_on"] == "own")
                & run_df["finetune/pretrained_ckpt_path"].apply(
                    lambda x: any(r in x for r in RUNS)
                )
            )
        ]
    return run_df


if __name__ == "__main__":
    run()
