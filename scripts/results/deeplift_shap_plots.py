import itertools
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import shap
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from pytorch_lightning.utilities.seed import seed_everything
from shap import Explanation, kmeans
from shap.plots import colors
from sssl import utils
from sssl.config import Config
from sssl.data.ipc import IPCBatch, IPCScoreDataset
from sssl.model.ipc_module import IPCModule
from torch import Tensor, nn
from tqdm import tqdm


def image_plot(
    shap_values: Explanation or np.ndarray,
    pixel_values: Optional[np.ndarray] = None,
    labels: Optional[list or np.ndarray] = None,
    true_labels: Optional[list] = None,
    width: Optional[int] = 20,
    aspect: Optional[float] = 0.2,
    hspace: Optional[float] = 0.2,
    labelpad: Optional[float] = None,
    cmap: Optional[str or Colormap] = colors.red_transparent_blue,
    show: Optional[bool] = True,
):
    """
    From shap library.
    Plots SHAP values for image inputs.

    Parameters
    ----------
    shap_values : [numpy.array]
        List of arrays of SHAP values. Each array has the shape
        (# samples x width x height x channels), and the
        length of the list is equal to the number of model outputs that are being
        explained.

    pixel_values : numpy.array
        Matrix of pixel values (# samples x width x height x channels) for each image.
        It should be the same
        shape as each array in the ``shap_values`` list of arrays.

    labels : list or np.ndarray
        List or ``np.ndarray`` (# samples x top_k classes) of names for each of the
        model outputs that are being explained.

    true_labels: list
        List of a true image labels to plot.

    width : float
        The width of the produced matplotlib plot.

    labelpad : float
        How much padding to use around the model output labels.

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    Examples
    --------

    See `image plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/image.html>`_.

    """

    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        # feature_names = [shap_exp.feature_names]
        # ind = 0
        if len(shap_exp.output_dims) == 1:
            shap_values = [
                shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])
            ]
        elif len(shap_exp.output_dims) == 0:
            shap_values = shap_exp.values
        else:
            raise Exception(
                "Number of outputs needs to have support added!! (probably a simple fix)"
            )
        if pixel_values is None:
            pixel_values = shap_exp.data
        if labels is None:
            labels = shap_exp.output_names

    # multi_output = True
    if not isinstance(shap_values, list):
        # multi_output = False
        shap_values = [shap_values]

    if len(shap_values[0].shape) == 3:
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    # labels: (rows (images) x columns (top_k classes) )
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels).reshape(1, -1)

    label_kwargs = {} if labelpad is None else {"pad": labelpad}

    # plot our explanations
    x = pixel_values
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = plt.subplots(
        nrows=x.shape[0] + 1, ncols=len(shap_values) + 1, figsize=fig_size
    )
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure we have a 2D array for grayscale
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                0.2989 * x_curr[:, :, 0]
                + 0.5870 * x_curr[:, :, 1]
                + 0.1140 * x_curr[:, :, 2]
            )  # rgb to gray
            x_curr_disp = x_curr
        elif len(x_curr.shape) == 3:
            x_curr_gray = x_curr.mean(2)

            # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
            flat_vals = x_curr.reshape(
                [x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]
            ).T
            flat_vals = (flat_vals.T - flat_vals.mean(1)).T
            means = kmeans(flat_vals, 3, round_values=False).data.T.reshape(
                [x_curr.shape[0], x_curr.shape[1], 3]
            )
            x_curr_disp = (means - np.percentile(means, 0.5, (0, 1))) / (
                np.percentile(means, 99.5, (0, 1)) - np.percentile(means, 1, (0, 1))
            )
            x_curr_disp[x_curr_disp > 1] = 1
            x_curr_disp[x_curr_disp < 0] = 0
        else:
            x_curr_gray = x_curr
            x_curr_disp = x_curr

        axes[row, 0].imshow(x_curr_disp, cmap=plt.get_cmap("gray"))
        if true_labels:
            axes[row, 0].set_title(true_labels[row], **label_kwargs)
        axes[row, 0].axis("off")
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack(
                [np.abs(shap_values[i]) for i in range(len(shap_values))], 0
            ).flatten()
        else:
            abs_vals = np.stack(
                [np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0
            ).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
        for i in range(len(shap_values)):
            if labels is not None:
                axes[row, i + 1].set_title(labels[row, i], fontsize=18, **label_kwargs)
            sv = (
                shap_values[i][row]
                if len(shap_values[i][row].shape) == 2
                else shap_values[i][row].sum(-1)
            )
            axes[row, i + 1].imshow(
                x_curr_gray,
                cmap=plt.get_cmap("gray"),
                alpha=0.15,
                extent=(-1, sv.shape[1], sv.shape[0], -1),
            )
            im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
            axes[row, i + 1].axis("off")
    for ax in axes[-1, :]:
        ax.axis("off")
    if hspace == "auto":
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(
        im,
        ax=axes[-1, :],
        orientation="horizontal",
        aspect=fig_size[0] / aspect,
    )
    cb.outline.set_visible(False)
    cb.set_label(label="SHAP value", size=20)
    if show:
        plt.show()


class IPCNetForShap(nn.Module):
    def __init__(self, ipc_module: IPCModule):
        super().__init__()
        self.ipc_module = ipc_module

    def forward(self, tiles: Tensor) -> Tensor:
        cnn_feats = self.ipc_module.backbone.model(tiles)
        logits = self.ipc_module.classifier.head(cnn_feats.squeeze(1))
        return nn.functional.softmax(logits, dim=-1)


if __name__ == "__main__":
    num_samples = 100
    num_bg_samples = 700
    num_bg_mean = 10
    seed = 42

    seed_everything(seed)

    with open("config/ipc/debug.yaml", "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Config()
    cfg.from_dict(dict_cfg)
    cfg.process_args()
    cfg.finetune.n_steps_in_future = 0
    cfg.finetune.temporally_separated = False

    ipc_ds = IPCScoreDataset(cfg, "val")
    train_ipc_ds = IPCScoreDataset(cfg, "train")

    p = "/path/to/s42_sssl_resnet18_t1_s04_ALL/checkpoints/<checkpoint>.ckpt"
    cfg.finetune.pretrained_on = "own"
    cfg.finetune.pretrained_ckpt_path = p
    module = IPCModule(cfg=cfg)
    model = IPCNetForShap(module)

    background_samples = []
    samples = []
    samples_per_ipc = [
        len(
            [
                pathlist[d]
                for (z, d) in train_ipc_ds.ipc_2_zd[ipc]
                for pathlist in train_ipc_ds.f.zone2box2p[
                    train_ipc_ds.f.all_admins[z]
                ].values()
                if len(pathlist) > d
            ]
        )
        for ipc in [0, 1, 2, 3]
    ]
    bg_samples_per_ipc = min(num_bg_samples, *samples_per_ipc)
    for ipc in [0, 1, 2, 3]:
        background_samples += itertools.islice(
            train_ipc_ds.sample_from_ipc_class(ipc, shuffle=True), bg_samples_per_ipc
        )
        samples += itertools.islice(
            ipc_ds.sample_from_ipc_class(ipc, shuffle=True), num_samples
        )

    bg_batch = ipc_ds.collate(background_samples)
    batch = ipc_ds.collate(samples)
    bg_tiles = bg_batch.tiles.to("cuda:0")
    bg_tiles = bg_tiles.reshape(
        num_bg_mean, (4 * num_bg_samples) // num_bg_mean, *bg_tiles.shape[1:]
    ).mean(0)
    tiles = batch.tiles.to("cuda:0")
    model = model.to("cuda:0")

    e = shap.DeepExplainer(model, bg_tiles)
    shap_values = e.shap_values(tiles)
    p = "output/2023_09_25_22_31_00_shap_values/shap_values.npy"
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    np.save(p, shap_values)
    p2 = "output/2023_09_25_22_31_00_shap_values/tiles.npy"
    np.save(p2, tiles.cpu())
    print("Computed and saved shap values")

    shap_values = np.array(shap_values)
    shap_per_band = (
        shap_values.reshape(*shap_values.shape[:-2], -1)
        .mean(axis=-1)
        .mean(axis=1)[:, 0]
    )
    shap_per_band_abs = (
        np.absolute(shap_values.reshape(*shap_values.shape[:-2], -1))
        .mean(axis=-1)
        .mean(axis=1)[:, 0]
    )

    plt.close("all")
    fig = plt.figure(figsize=(9, 6))
    df = pd.DataFrame(
        shap_per_band,
        columns=["UB", "B", "G", "R", "NIR", "SW-IR-1", "SW-IR-2"],
    )
    df.transpose().plot(kind="bar", stacked=True)
    plt.hlines(y=0, xmin=-1, xmax=7, colors="black")
    plt.title("Mean SHAP values per band", fontsize=13)
    plt.xlabel("Landsat-8 Band", fontsize=12)
    plt.ylabel("SHAP", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(
        title="Pred. IPC Score", fontsize=12, bbox_to_anchor=(1.04, 1), loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(f"imgs/shap50_mean_{seed}_{num_bg_samples}.pdf")
    plt.show()

    plt.close("all")
    # fig, ax = plt.subplots()
    df_abs = pd.DataFrame(
        shap_per_band_abs,
        columns=["UB", "B", "G", "R", "NIR", "SW-IR-1", "SW-IR-2"],
    )
    df_abs.transpose().plot(kind="bar", stacked=True)
    # plt.hlines(y=0, xmin=-1, xmax=7, colors="black")
    plt.title("Mean absolute SHAP values per band")
    plt.xlabel("Landsat-8 Band", fontsize=12)
    plt.ylabel("Absolute SHAP", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Pred. IPC Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"imgs/shap50_mean_abs_{seed}_{num_bg_samples}.pdf")
    plt.show()

    plt.close("all")
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(7, 9))
    shap_per_band_per_ipc = (
        shap_values.reshape(4, 4, num_samples, 1, 7, -1)
        .mean(axis=-1)
        .mean(axis=2)[:, :, 0]
    )
    for ipc, ax in zip([0, 1, 2, 3], axs):
        df = pd.DataFrame(
            shap_per_band_per_ipc[:, ipc],
            columns=["UB", "B", "G", "R", "NIR", "SW-IR-1", "SW-IR-2"],
        )
        df.transpose().plot(kind="bar", stacked=True, ax=ax)
        ax.hlines(y=0, xmin=-1, xmax=7, colors="black")
        ax.set_title(f"Images with IPC label {ipc}")
        if ipc == 0:
            ax.legend(
                title="Predicted IPC Score",
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
                fontsize=13,
            )
        else:
            ax.get_legend().remove()
    fig.supylabel("Importance", fontsize=13)
    plt.xticks(
        [0, 1, 2, 3, 4, 5, 6],
        ["UB", "B", "G", "R", "NIR", "SW-IR-1", "SW-IR-2"],
        rotation=90,
        fontsize=13,
    )
    plt.yticks(fontsize=13)
    plt.xlabel("Landsat-8 band", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"imgs/shap50_mean_ipc_{seed}_{num_bg_samples}.pdf")
    plt.show()

    plt.close("all")
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(7, 9))
    shap_per_band_per_ipc_abs = (
        np.absolute(shap_values)
        .reshape(4, 4, num_samples, 1, 7, -1)
        .mean(axis=-1)
        .mean(axis=2)[:, :, 0]
    )
    for ipc, ax in zip([0, 1, 2, 3], axs):
        df = pd.DataFrame(
            shap_per_band_per_ipc_abs[:, ipc],
            columns=["UB", "B", "G", "R", "NIR", "SW-IR-1", "SW-IR-2"],
        )
        df.transpose().plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"Images with IPC label {ipc}")
        # ax.hlines(y=0, xmin=-1, xmax=7, colors="black")
        if ipc == 0:
            ax.legend(
                title="Predicted IPC Score",
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
                fontsize=13,
            )
        else:
            ax.get_legend().remove()
    fig.supylabel("Absolute importance", fontsize=13)
    plt.xticks(
        [0, 1, 2, 3, 4, 5, 6],
        ["UB", "B", "G", "R", "NIR", "SW-IR-1", "SW-IR-2"],
        rotation=90,
        fontsize=13,
    )
    plt.yticks(fontsize=13)
    plt.xlabel("Landsat-8 band", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"imgs/shap50_mean_abs_ipc_{seed}_{num_bg_samples}.pdf")
    plt.show()

    first_per_class = np.array([0, 1, 2, 3]) * num_samples
    bands = utils.Constants.RGB_BANDS
    for i in tqdm(range(num_samples)):
        for (band_names, shap_bands) in [
            ("RGB", utils.Constants.RGB_BANDS),
            ("R-NIR-SWIR", [3, 4, 5]),
            ("B-SWIR12", [1, 5, 6]),
        ]:
            shap_to_plot = [
                np.swapaxes(s[first_per_class + i, 0], 1, 3)[:, :, :, shap_bands]
                for s in shap_values
            ]
            # take the first 3 images, skip the singleton dimension, swap channel dim to last, take RGB
            img_to_plot = np.swapaxes(tiles.cpu()[first_per_class + i, 0], 1, 3)[
                :, :, :, bands
            ]
            # un-normalize:
            un_img = (
                np.nan_to_num(img_to_plot, nan=0.0)
                * utils.Constants.CHANNEL_STDS[bands].numpy()
                + utils.Constants.CHANNEL_MEANS[bands].numpy()
            )
            plt.close("all")
            labels = [[0, 1, 2, 3]] + 3 * [4 * [""]]
            image_plot(
                shap_to_plot,
                1.3 * un_img ** (1 / 1.6),
                # true_labels=labels[0],
                labels=np.array(labels),
                hspace=0.15,
                # labelpad=-5.0,
                show=False,
            )
            plt.suptitle(f"SHAP values for example images", fontsize=22)
            plt.tight_layout()
            plt.savefig(
                f"imgs/shap50_examples_{i}_{seed}_{num_bg_samples}_{band_names}.pdf"
            )
            plt.show()

    print("ok")
