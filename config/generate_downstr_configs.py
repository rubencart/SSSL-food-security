cfgstr = """
cfg_name: '%s'

do_pretrain: False
do_downstream: True

landsat8_bands: '%s'

finetune:
  freeze_backbone: %s
  pretrained_ckpt_path: ''
  pretrained: %s
  pretrained_on: '%s'
  clf_head: '%s'
  percentage_of_training_data: %s
  binarize_ipc: %s
  n_steps_in_future: %s
  temporally_separated: %s
"""

for freeze in (True, False):
    for head in ("mlp", "linear"):
        for perc in (100, 70, 50, 20, 5, 1):
            for pretr in (True, False):
                pretr_ons = ("own", "ImageNet") if pretr else ("random",)
                for pretr_on in pretr_ons:
                    for bands in ("ALL", "RGB"):
                        for bin in (True, False) if perc == 100 else (False,):
                            for temp_sep in (True, False) if perc == 100 else (False,):
                                step_aheads = (0, 1, 2, 3) if temp_sep else (0,)
                                for step in step_aheads:
                                    print(
                                        freeze, head, perc, pretr, pretr_on, bin, step
                                    )
                                    cname = "%s_%s_%s" % (
                                        "frz" if freeze else "ft",
                                        head,
                                        perc,
                                    )
                                    if pretr_on in ("ImageNet", "random"):
                                        cname = pretr_on + "_" + cname
                                    if bands == "RGB":
                                        cname += "_RGB"
                                    if temp_sep:
                                        cname += f"_{step}fut_after202003"
                                    if bin:
                                        cname += "_bin"
                                    fstring = cfgstr % (
                                        cname,
                                        bands,
                                        freeze,
                                        pretr,
                                        pretr_on,
                                        head,
                                        perc,
                                        bin,
                                        step,
                                        temp_sep,
                                    )
                                    with open(f"config/ipc/{cname}.yaml", "w") as f:
                                        f.write(fstring)
