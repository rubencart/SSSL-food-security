from datetime import datetime

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sssl import utils


def pretrain_run_plot(pt_run_grps, metric, title, rf_score_bal, rf_score, maj_bas_score):
    plt.close()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.gca()
    colors = sns.color_palette("hls", len(pt_run_grps))
    ax.set_prop_cycle("color", colors)
    for (dirn, _, head, is_sssl), grp in pt_run_grps:
        grp = grp.sort_values("finetune/pretrained_epoch")
        linestyle = "solid" if is_sssl else "dashed"
        grp.plot(
            x="finetune/pretrained_epoch",
            y=metric,
            ax=ax,
            marker=".",
            label=grp.cfg_name.iloc[0],
            legend=False,
            linestyle=linestyle,
        )
    plt.axhline(
        y=rf_score_bal.item(),
        # color=next(ax._get_lines.prop_cycler)["color"],  # no longer supported in matplotlib
        linestyle="dotted",
        label="rf_score_bal",
    )
    plt.axhline(
        y=rf_score.item(),
        # color=next(ax._get_lines.prop_cycler)["color"],
        linestyle="dotted",
        label="rf_score",
    )
    plt.axhline(
        y=maj_bas_score.item(),
        # color=next(ax._get_lines.prop_cycler)["color"],
        linestyle="dotted",
        label="maj_bas_score",
    )
    plt.title("%s_%s" % (metric, title))
    # https://stackoverflow.com/a/43439132/2332296
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("imgs/%s_%s.pdf" % (metric, title))
    plt.show()


def loss_type_plot(loss_grps, metric, frz, title, rf_score_bal, rf_score, maj_bas_score, clf_head):
    plt.close()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.gca()
    colors = sns.color_palette("hls", 2 + 3)
    ax.set_prop_cycle("color", colors)

    loss_means = loss_grps[metric].mean()
    loss_stds = loss_grps[metric].std()

    for is_sssl in (True, False):
        #                     freeze=True
        mns = loss_means.loc[(clf_head, is_sssl)]
        stds = loss_stds.loc[(clf_head, is_sssl)]

        # linestyle = 'solid' if not is_sssl else 'dashed'
        label = "%s_%s" % (
            "sssl" if is_sssl else "tile2vec",
            "frz" if frz else "ft",
        )
        plt.plot(
            mns.index,
            mns,
            marker=".",
            label=label,
            # linestyle=linestyle,
        )
        plt.fill_between(mns.index, mns - stds, mns + stds, alpha=0.25)

    plt.axhline(
        y=rf_score_bal.item(),
        # color=next(ax._get_lines.prop_cycler)["color"],
        linestyle="dotted",
        label="rf_score_bal",
    )
    plt.axhline(
        y=rf_score.item(),
        # color=next(ax._get_lines.prop_cycler)["color"],
        linestyle="dotted",
        label="rf_score",
    )
    plt.axhline(
        y=maj_bas_score.item(),
        # color=next(ax._get_lines.prop_cycler)["color"],
        linestyle="dotted",
        label="maj_bas_score",
    )

    plt.xlabel("finetune/pretrained_epoch")
    plt.title("%s_%s" % (metric, title))
    # https://stackoverflow.com/a/43439132/2332296
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("imgs/lt_%s_%s.pdf" % (metric, title))
    plt.show()


METRIC_NAME_DICT = {
    "test_val_f1_macro": "Macro F1",
    "test_id_f1_macro": "Macro F1",
}


def limit_plot(df, metrics, glob_metric, fn, title, maj_bas_score):
    plt.close()
    fs = 17
    fig = plt.figure(figsize=(13, 6))
    ax = fig.gca()
    colors = sns.color_palette("hls", len(metrics) + 1)
    ax.set_prop_cycle("color", colors)
    sum_df = df.groupby(["pretrain_tlimit", "pretrain_slimit"], dropna=False)[metrics].max()

    ax = sum_df.plot.bar(ax=ax, color=colors, rot=45, width=0.65)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.axhline(
        y=maj_bas_score.item(),
        color=colors[-1],
        linestyle="dashed",
        label="Maj. baseline",
    )
    plt.ylabel(METRIC_NAME_DICT.get(glob_metric, glob_metric), fontsize=fs)
    plt.xlabel("$(D_t, D_g)$ in months and degrees of lat/lon", fontsize=fs)
    plt.title(title.replace("test_", "").replace("_", " "), fontsize=fs)
    plt.legend(
        [
            "Majority baseline",
            "Majority voting",
            "Maximum voting",
            "Individual tiles",
        ],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=fs,
    )
    plt.tight_layout()
    plt.savefig("imgs/limit_%s_lrg.pdf" % fn)
    plt.show()


def run():
    # from sessions.wandb_epoch_plot import limit_plot, pretrain_run_plot, loss_type_plot
    df = utils.get_df_from_wandb()

    m2 = df["finetune/pretrained_on"] == "own"
    m3 = ~df.debug
    m4 = pd.notna(df["finetune/pretrained_epoch"])
    m5 = pd.notna(df["finetune/all_backbone_ckpts_in_dir"])
    m6 = pd.notna(df["test_id_maj_vote_f1_weighted"])
    m7 = df["finetune/pretrained_epoch"] > -1
    min_date = datetime.strptime("2023-01-01", "%Y-%m-%d")
    m8 = pd.to_datetime(df["createdAt"], format="ISO8601").dt.date > min_date.date()
    # m9 = df['landsat8_bands'] == 'RGB'
    # m11 = ~df['finetune/all_backbone_ckpts_in_dir'].str.contains('relres')
    m12 = pd.isna(df["finetune/binarize_ipc"]) | (df["finetune/binarize_ipc"] == False)
    m13 = pd.isna(df["finetune/n_steps_in_future"]) | (df["finetune/n_steps_in_future"] == 0)
    m14 = ~df["cfg_name"].str.contains("_best")

    totmask = m2 & m3 & m4 & m5 & m6 & m7 & m8 & m12 & m13 & m14

    all_df = df.loc[totmask].copy()
    all_df = all_df[
        ~all_df["finetune/all_backbone_ckpts_in_dir"].str.contains("relres").astype("boolean")
    ]
    all_df["pretrain_loss_type_sssl"] = all_df["finetune/all_backbone_ckpts_in_dir"].str.contains(
        "sssl"
    )
    all_df["pretrain_cfg_name"] = all_df["finetune/all_backbone_ckpts_in_dir"].str.extract(
        r"output\/[0-9_]+s[0-9]+_([0-9a-zA-Z_]+)\/", expand=False
    )
    all_df["finetune/all_backbone_ckpts_in_dir"] = all_df[
        "finetune/all_backbone_ckpts_in_dir"
    ].str.rstrip("2/")
    all_df["pretrain_tlimit"] = (
        all_df["finetune/all_backbone_ckpts_in_dir"]
        .str.extract(r"_t([0-9]+)_", expand=False)
        .astype("int")
    )
    all_df["pretrain_slimit"] = all_df["finetune/all_backbone_ckpts_in_dir"].str.extract(
        r"_s([0-9]+)_(?:RGB|ALL)", expand=False
    )
    not_admin_mask = pd.notna(all_df["pretrain_slimit"])
    all_df.loc[not_admin_mask, "pretrain_slimit"] = all_df.loc[
        not_admin_mask, "pretrain_slimit"
    ].str.replace("0", "0.")
    admin_mask = all_df["finetune/all_backbone_ckpts_in_dir"].str.contains("_admin_")
    all_df["pretrain_stype"] = admin_mask
    all_df.loc[admin_mask, "pretrain_slimit"] = "admin"
    assert all(admin_mask == ~not_admin_mask)

    rf_df = df[df.cfg_name.str.contains("random_forest")]

    frz, bands = True, "RGB"
    for frz in (True,):  # (True, False):
        for bands in ("ALL", "RGB"):
            metrics, rf_metric, maj_baseline_metric = (
                [
                    "test_val_maj_vote_f1_macro",
                    "test_val_max_vote_f1_macro",
                    "test_val_tile_f1_macro",
                ],
                "val_f1_macro",
                "val_maj_baseline_f1_macro",
            )
            for metrics, rf_metric, maj_baseline_metric in (
                (
                    [
                        "test_val_maj_vote_f1_macro",
                        "test_val_max_vote_f1_macro",
                        "test_val_tile_f1_macro",
                    ],
                    "val_f1_macro",
                    "val_maj_baseline_f1_macro",
                ),
                (
                    [
                        "test_id_maj_vote_f1_macro",
                        "test_id_max_vote_f1_macro",
                        "test_id_tile_f1_macro",
                    ],
                    "test_f1_macro",
                    "test_maj_baseline_f1_macro",
                ),
                (
                    [
                        "test_id_maj_vote_f1_weighted",
                        "test_id_max_vote_f1_weighted",
                        "test_id_tile_f1_weighted",
                    ],
                    "test_f1_weighted",
                    "test_maj_baseline_f1_weighted",
                ),
                (
                    [
                        "test_val_maj_vote_f1_weighted",
                        "test_val_max_vote_f1_weighted",
                        "test_val_tile_f1_weighted",
                    ],
                    "val_f1_weighted",
                    "val_maj_baseline_f1_weighted",
                ),
                (
                    "test_id_bin_maj_vote_f1_weighted",
                    "test_bin_f1_weighted",
                    "test_bin_maj_baseline_f1_weighted",
                ),
                (
                    "test_id_bin_maj_vote_f1_macro",
                    "test_bin_f1_macro",
                    "test_bin_maj_baseline_f1_macro",
                ),
            ):
                frz_mask = all_df["finetune/freeze_backbone"] == frz
                bnd_mask = all_df["landsat8_bands"] == bands

                rf_df = rf_df[
                    pd.isna(rf_df["random_forest/n_steps_in_future"])
                    | rf_df["random_forest/n_steps_in_future"]
                    == 0
                ]
                rf_df = rf_df[
                    pd.isna(rf_df["random_forest/temporally_separated"])
                    | ~rf_df["random_forest/temporally_separated"]
                ]
                rf_df = rf_df[
                    pd.isna(rf_df["random_forest/sssl_predictions"])
                    | (rf_df["random_forest/sssl_predictions"] == "")
                ]
                rf_score_bal = rf_df[rf_df["random_forest/class_weight"] == "balanced"][
                    rf_metric
                ].iloc[0]
                rf_score = rf_df[rf_df["random_forest/class_weight"].isnull()][rf_metric].iloc[0]
                maj_bas_score = rf_df[rf_df["random_forest/class_weight"].isnull()][
                    maj_baseline_metric
                ].iloc[0]

                ft_df = all_df[frz_mask & bnd_mask]
                for ltype in (
                    "Tile2Vec",
                    "SSSL",
                ):
                    try:
                        mask = (
                            ft_df["pretrain_loss_type_sssl"]
                            if ltype == "SSSL"
                            else ~ft_df["pretrain_loss_type_sssl"]
                        )
                        ltype_df = ft_df[mask]
                        # if ltype == 'sssl':  # exclude relres
                        glob_metric = metrics[-1].replace("_tile", "")
                        title = f'{glob_metric}_{"frz" if frz else "ft"}_{ltype.lower()}_{bands}'

                        limit_plot(
                            ltype_df.copy(),
                            metrics,
                            glob_metric,
                            title,
                            f"Thresholds & aggregation: {ltype}",
                            maj_bas_score,
                        )
                    except IndexError:
                        print(
                            "Not enough completed runs for pairs plot for frz %s, loss type %s,"
                            " bands %s, metric %s" % (frz, ltype, bands, metrics[0])
                        )

                for metric in metrics:
                    for head in ("linear", "mlp"):
                        try:
                            loss_grps = ft_df.groupby([
                                "finetune/clf_head",
                                # 'pretrain/loss_type',
                                "pretrain_loss_type_sssl",
                                "finetune/pretrained_epoch",
                            ])
                            title = f'{"frz" if frz else "ft"}_{bands}_{head}'
                            loss_type_plot(
                                loss_grps,
                                metric,
                                frz,
                                title,
                                rf_score_bal,
                                rf_score,
                                maj_bas_score,
                                head,
                            )

                        except (StopIteration, KeyError):
                            print(
                                "Not enough completed runs for loss_type plot for frz %s, metric"
                                " %s, bands %s, head %s" % (frz, metric, bands, head)
                            )

                    for tlimit in (1, 4, 12, 36, 84):
                        try:
                            pt_epoch_df = all_df[
                                frz_mask
                                & bnd_mask
                                & all_df["cfg_name"].str.contains(f"_t{tlimit}_")
                            ]
                            pt_run_grps = pt_epoch_df.groupby([
                                # 'finetune/all_backbone_ckpts_in_dir',
                                "pretrain_cfg_name",
                                "finetune/freeze_backbone",
                                "finetune/clf_head",
                                # 'pretrain/loss_type',
                                "pretrain_loss_type_sssl",
                            ])
                            title = f'{"frz" if frz else "ft"}_t{tlimit}_{bands}'
                            pretrain_run_plot(
                                pt_run_grps,
                                metric,
                                title,
                                rf_score_bal,
                                rf_score,
                                maj_bas_score,
                            )

                        except StopIteration:
                            print(
                                "Not enough completed runs for pt run plot for frz %s, tlimit %s,"
                                " metric %s, bands %s" % (frz, tlimit, metric, bands)
                            )


if __name__ == "__main__":
    run()
