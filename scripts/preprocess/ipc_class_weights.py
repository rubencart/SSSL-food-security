import logging

import numpy as np
import yaml
from sssl.config import Config
from sssl.data.ipc import IPCLandsat8Files, IPCScoreDataset

logger = logging.getLogger("pytorch_lightning")


if __name__ == "__main__":

    with open("config/ipc/debug.yaml", "r") as f:
        dict_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Config()
    cfg.from_dict(dict_cfg)
    cfg.process_args()

    cfg.do_pretrain = False
    cfg.do_downstream = True
    cfg.debug = False
    cfg.num_workers = 10
    cfg.finetune.percentage_of_training_data = 100

    # run once for True and once for False
    for binarize in (True, False):
        logger.info("Running with binarize IPC: %s" % binarize)
        cfg.finetune.binarize_ipc = binarize

        files = IPCLandsat8Files(cfg, "train", load_zone2box2p=True)
        ds = IPCScoreDataset(cfg, "train")

        # use this if you want to bigger regions to have more weight, otherwise just count z,d and normalize by Z,D
        reg_sizes = {}
        tot_size = len(ds.boxes)
        for region in ds.regions:
            reg_sizes[region] = len(files.zone2box2p[region])

        print(sum(reg_sizes.values()))
        print(tot_size)

        zone_time_combos = [
            (z, d, ds.get_ipc_for_zone_date(z, d))
            for z in ds.zone_ids
            for d in range(len(files.all_end_dates))
        ]

        ipc_weights = (
            np.array([0.0, 0.0, 0.0, 0.0])
            if not cfg.finetune.binarize_ipc
            else np.array([0.0, 0.0])
        )
        for zi, ti, ipc in zone_time_combos:
            ipc_weights[ipc] += reg_sizes[ds.f.all_admins[zi]]

        # wrong: dividing by number of admin zones not needed, already normalizing by boxes!
        # ipc_weights / (tot_size * len(ds.zone_ids) * len(files.all_end_dates))
        print(ipc_weights)
        print(1 - (ipc_weights / tot_size / len(files.all_end_dates)))
        print((1 / ipc_weights) / sum(1 / ipc_weights))
