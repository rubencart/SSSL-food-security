cfgstr = """
cfg_name: '%s'
cnn_type: '%s'
landsat8_bands: '%s'
train:
  max_epochs: %s
pretrain:
  loss_type: '%s'
  time_pair_limit: %s
  space_pair_limit: %s
  space_limit_type: '%s'
  lr: %s
"""

# for cnn in ('resnet18', 'conv4', 'resnet34'):
for cnn in ("resnet18",):
    for loss in ("tile2vec", "sssl"):
        for lbands in ("RGB", "ALL"):
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
                        ).replace(".", "")
                        nb_epochs = 20 if loss == "tile2vec" else 10
                        lr = 0.0001
                        fstring = cfgstr % (
                            cname,
                            cnn,
                            lbands,
                            nb_epochs,
                            loss,
                            tlimit,
                            slimit,
                            sltype,
                            lr,
                        )
                        with open(f"config/pretrain/{cname}.yaml", "w") as f:
                            f.write(fstring)
