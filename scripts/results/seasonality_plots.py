import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sssl import utils
from sssl.config import Config
from sssl.data.ipc import IPCScoreDataset
from torchmetrics.functional.classification import multiclass_f1_score as mf1

PREDS = [
    (
        "SSSL",
        "/path/to/s42_sssl_resnet18_t1_s04_ALL_#_ft_mlp_100/test_preds.pkl",
    ),
]


def plot(
    d_ipc,
    d_preds,
    title,
    boxplot=False,
    violinplot=True,
    plot_seasons=True,
    plot_f1=True,
    metric_avg="macro",
    gt_distr_only=True,
    dates=None,
    fn=None,
):
    plt.close()
    fig = plt.figure(figsize=(13 if dates else 11, 6))
    ax = fig.gca()
    if dates is not None:
        ticks = [d.replace("-01", "") for d in dates]
    else:  # if seasons is not None:
        ticks = ["Winter", "Spring", "Summer", "Autumn"]

    if boxplot:

        def set_box_color(bp, color):
            plt.setp(bp["boxes"], color=color)
            plt.setp(bp["whiskers"], color=color)
            plt.setp(bp["caps"], color=color)
            plt.setp(bp["medians"], color=color)

        col_fn, plot_fn = set_box_color, ax.boxplot
        kwargs = {"showmeans": True, "meanline": True}

    if violinplot:

        def set_violin_color(bp, color):
            for partname in ("cbars", "cmins", "cmaxes"):  # , 'cmeans', 'cmedians',
                vp = bp[partname]
                vp.set_linewidth(0.0)

            # Make the violin body blue with a red border:
            for vp in bp["bodies"]:
                vp.set_facecolor(color)
                vp.set_alpha(1.0)

        col_fn, plot_fn = set_violin_color, ax.violinplot
        kwargs = {}

    width_factor = 2.5 if not gt_distr_only else 1
    w = 0.4
    dev = 0.5
    xtick_coords = np.array(range(0, len(ticks), 1)) * width_factor

    if plot_seasons:
        alph = 0.25
        colors = [
            {"facecolor": "cornflowerblue", "alpha": alph},
            {"facecolor": "springgreen", "alpha": alph},
            {"facecolor": "gold", "alpha": alph},
            {"facecolor": "peru", "alpha": alph},
        ]
        seasons = (
            [(int(d[-2:]) - 1) // 3 for d in ticks] if dates is not None else range(4)
        )
        for i, season in enumerate(seasons):
            mp = i * width_factor
            ax.axvspan(
                xmin=mp - width_factor / 2, xmax=mp + width_factor / 2, **colors[season]
            )

    shapes = []
    if boxplot or violinplot:
        bpl = plot_fn(
            [
                d_ipc[i] + torch.rand_like(d_ipc[i].float()) / 20
                for (i, d) in enumerate(ticks)
            ],
            **(
                {"positions": xtick_coords, "widths": w * (1.5 if dates else 1.0)}
                if gt_distr_only
                else {"positions": xtick_coords - dev, "widths": w}
            ),
            **kwargs,
        )
        col_fn(bpl, "#D7191C")  # colors are from http://colorbrewer2.org/
        colors_labels = [
            ("#D7191C", "Ground truth IPC distribution\n(left y-axis)"),
        ]
        if not gt_distr_only:
            bpr = plot_fn(
                [
                    d_preds[i] + torch.rand_like(d_preds[i].float()) / 20
                    for (i, d) in enumerate(ticks)
                ],
                positions=xtick_coords + dev,
                widths=w,
                **kwargs,
            )
            # shapes.append(bpr)
            col_fn(bpr, "#2C7BB6")
            colors_labels.append(
                ("#2C7BB6", "Predicted IPC distribution\n(left y-axis)")
            )

        # draw temporary red and blue lines and use them to create a legend
        for col, lab in colors_labels:
            ax.scatter([], [], c=col, label=lab)

        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks(
            range(utils.Constants.NB_IPC_SCORES),
            range(1, utils.Constants.NB_IPC_SCORES + 1),
            fontsize=12,
        )
        ax.set_ylabel("IPC", fontsize=13)

    ax.set_xticks(xtick_coords, ticks, rotation="vertical", fontsize=12)
    ax.set_xlim(-width_factor / 2, len(ticks) * width_factor - (width_factor / 2))
    ax.set_xlabel("Date" if dates else "Season", fontsize=13)
    lines, labels = ax.get_legend_handles_labels()

    if plot_f1:
        ax2 = ax.twinx()
        scores = [
            mf1(
                d_preds[i],
                d_ipc[i],
                # num_classes=utils.Constants.NB_IPC_SCORES,
                num_classes=len(torch.unique(torch.cat((d_ipc[i], d_preds[i])))),
                average=metric_avg,
            )
            for (i, d) in enumerate(ticks)
        ]
        line = ax2.plot(
            xtick_coords,
            scores,
            c="magenta",
            label="Macro F1 of best SSSL model\n(right y-axis)",
            marker="^",
            mfc="magenta",
        )
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Macro F1", fontsize=13)
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines, labels = lines + lines2, labels + labels2

    # ax.legend(shapes, [s.get_label() for s in shapes],
    ax.legend(
        lines,
        labels,
        loc="upper left",
        fontsize=12,
        bbox_to_anchor=(1.04 if dates else 1.08, 1),
    )
    plt.title("%s" % title, fontsize=14)
    plt.tight_layout()
    plt.savefig("imgs/%s.pdf" % (fn if fn else title.replace(" ", "_").lower()))
    plt.show()
    plt.close()


def run():
    with open("config/ipc/debug.yaml", "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Config()
    cfg.from_dict(dict_cfg)
    cfg.process_args()

    tds = IPCScoreDataset(cfg, "test")
    oodds = IPCScoreDataset(cfg, "ood")

    preds = {}
    for name, path in PREDS:
        with open(path, "rb") as f:
            # preds[name] = utils.preds_to_cpu(pickle.load(f))
            preds[name] = utils.preds_to_cpu(torch.load(f))

    print("done")
    for run_name in preds.keys():
        for test_set in preds[run_name].keys():
            zd_ipc = preds[run_name][test_set]["zone_date_2_ipc"]
            zd_mv = preds[run_name][test_set]["zone_date_2_maj_vote"]
            zds = list(zd_ipc.keys())
            zd_logits = preds[run_name][test_set]["zone_date_2_logits"]

            # tile
            date2zone2ipc, date2zone2preds = {}, {}
            for (z, d) in zds:
                p = zd_logits[(z, d)]
                date2zone2ipc.setdefault(d, []).append(
                    torch.full((len(p),), zd_ipc[(z, d)])
                )
                date2zone2preds.setdefault(d, []).append(p)

            def dict_cat(dct):
                return {k: torch.cat(v) for k, v in dct.items()}

            date2ipc_all = dict_cat(date2zone2ipc)
            date2preds_all = {
                d: torch.cat([torch.stack(tl).argmax(-1) for tl in rl])
                for d, rl in date2zone2preds.items()
            }
            # plot(
            #     date2ipc_all,
            #     date2preds_all,
            #     f"dt_{run_name}-{test_set}-tile_avg",
            #     dates=tds.f.all_end_dates,
            # )

            # zone maj vote
            date2ipc_z, date2preds_z = {}, {}
            for (z, d) in zds:
                date2ipc_z.setdefault(d, []).append(zd_ipc[(z, d)])
                date2preds_z.setdefault(d, []).append(zd_mv[(z, d)])
            date2ipc_z = {k: torch.tensor(v) for (k, v) in date2ipc_z.items()}
            date2preds_z = {k: torch.tensor(v) for (k, v) in date2preds_z.items()}

            # plot(
            #     date2ipc_z,
            #     date2preds_z,
            #     f"dt_{run_name}-{test_set}-zone_avg",
            #     dates=tds.f.all_end_dates,
            # )

            # tile season
            season2ipc_all, season2preds_all = {}, {}
            for (z, d) in zds:
                sn = utils.date_to_season(tds.f.all_end_dates[d])
                season2ipc_all.setdefault(sn, []).append(date2ipc_all[d])
                season2preds_all.setdefault(sn, []).append(date2preds_all[d])
            season2ipc_all = dict_cat(season2ipc_all)
            season2preds_all = dict_cat(season2preds_all)
            # plot(
            #     season2ipc_all,
            #     season2preds_all,
            #     f"{run_name} macro F1 vs. season",
            #     fn=f"sn_{run_name}-{test_set}-season_tile_avg",
            # )

            # tile season
            season2ipc_z, season2preds_z = {}, {}
            for (z, d) in zds:
                sn = utils.date_to_season(tds.f.all_end_dates[d])
                season2ipc_z.setdefault(sn, []).append(zd_ipc[(z, d)])
                season2preds_z.setdefault(sn, []).append(zd_mv[(z, d)])
            season2ipc_z = {k: torch.tensor(v) for (k, v) in season2ipc_z.items()}
            season2preds_z = {k: torch.tensor(v) for (k, v) in season2preds_z.items()}
            plot(
                season2ipc_z,
                season2preds_z,
                f"{run_name} macro F1 & IPC distribution per season",
                fn=f"sn_{run_name}-{test_set}-season_zone_avg",
                gt_distr_only=False,
            )


if __name__ == "__main__":
    run()
