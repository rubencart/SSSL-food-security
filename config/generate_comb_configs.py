cfgstr = """
pretrain_cfg: 'config/pretrain/%s'
do_pretrain: True
pretrained_on: 'own'
checkpoints: 'all'    # all, best, last

downstream_cfg:
  - 'config/ipc/frz_mlp_100.yaml'

best_cfg:
  - 'config/ipc/frz_mlp_100_bin.yaml'
  - 'config/ipc/frz_mlp_70.yaml'
  - 'config/ipc/frz_mlp_50.yaml'
  - 'config/ipc/frz_mlp_20.yaml'
  - 'config/ipc/frz_mlp_5.yaml'
  - 'config/ipc/frz_mlp_1.yaml'
  - 'config/ipc/ft_mlp_100.yaml'
  - 'config/ipc/ft_mlp_100_bin.yaml'
  - 'config/ipc/ft_mlp_70.yaml'
  - 'config/ipc/ft_mlp_50.yaml'
  - 'config/ipc/ft_mlp_20.yaml'
  - 'config/ipc/ft_mlp_5.yaml'
  - 'config/ipc/ft_mlp_1.yaml'
  - 'config/ipc/ft_mlp_100_0fut_after202003.yaml'
  - 'config/ipc/ft_mlp_100_1fut_after202003.yaml'
  - 'config/ipc/ft_mlp_100_2fut_after202003.yaml'
  - 'config/ipc/ft_mlp_100_3fut_after202003.yaml'
  - 'config/ipc/frz_mlp_100_0fut_after202003.yaml'
  - 'config/ipc/frz_mlp_100_1fut_after202003.yaml'
  - 'config/ipc/frz_mlp_100_2fut_after202003.yaml'
  - 'config/ipc/frz_mlp_100_3fut_after202003.yaml'
# """

for cnn in ("resnet18",):
    for loss in ("tile2vec", "sssl"):
        for lbands in ("ALL", "RGB"):
            for sltype in ("degrees", "admin"):
                slimits = ("admin",) if sltype == "admin" else (0.15, 0.4)
                for slimit in slimits:
                    tlimits = (1, 4, 12, 36, 84)
                    for tlimit in tlimits:
                        print(cnn, loss, lbands, sltype, slimit, tlimit)
                        cname = (
                            "%s_%s_%s_%s_%s"
                            % (
                                loss,
                                cnn,
                                f"t{tlimit}",
                                f"s{slimit}" if sltype == "degrees" else "admin",
                                lbands,
                            )
                        ).replace("0.", "0")
                        cname += "" if lbands == "ALL" else "_RGB"
                        cname += ".yaml"
                        fstring = cfgstr % cname
                        with open(f"config/pretrain_ipc/{cname}", "w") as f:
                            f.write(fstring)
