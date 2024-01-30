import logging

import torch
import yaml
from sssl.config import Config
from sssl.data.landsat8 import SSSLDataset, train_dataloader
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

    cfg.pretrain.K = 1
    cfg.deterministic = True
    cfg.debug = False
    cfg.num_workers = 20
    cfg.pretrain.batch_size = 50
    cfg.cuda = True
    cfg.landsat8_replace_nan = False
    cfg.landsat8_bands = "ALL"
    cfg.landsat8_normalize = False

    print("building ds")
    ds = SSSLDataset(cfg, split="train")
    print("building dl")
    dl = train_dataloader(ds, cfg)

    ms, cs = [0 for _ in range(7)], 0
    hcs = 0
    for j, batch in tqdm(enumerate(iter(dl))):
        # mean over every dimension except channel dimension
        # batch.tiles = batch.tiles.cuda(0)

        cs += batch.bs
        for i in range(7):
            val = batch.tiles[:, :, i].nanmean(dim=[1, 2, 3]).sum(0).detach()
            ms[i] += val

    means = torch.tensor([m / cs for m in ms])
    sqms = [0 for _ in range(7)]

    for j, batch in tqdm(enumerate(iter(dl))):
        batch.tiles = batch.tiles.cuda(0)

        for i in range(7):
            var = (
                (batch.tiles[:, :, i] - means[i]).pow(2).nansum(dim=[0, 1, 2, 3])
                / (~torch.isnan(batch.tiles[:, :, i].reshape(-1)) / batch.bs).sum()
            ).detach()
            sqms[i] += var

    stds_norm = torch.tensor([s / (cs - 1) for s in sqms]).sqrt()

    logger.info("Means: %s" % means)
    logger.info("Stds norm: %s" % stds_norm)
    logger.info("Done")
