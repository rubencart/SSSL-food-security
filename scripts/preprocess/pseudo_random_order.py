import json
import os
import random

indices_dir = "data/indices/"
random.seed(42)

with open(os.path.join(indices_dir, "ood_splits.json"), "r") as f:
    ood_split_dict = json.load(f)
with open(os.path.join(indices_dir, "val_splits.json"), "r") as f:
    val_split_dict = json.load(f)
with open(os.path.join(indices_dir, "test_splits.json"), "r") as f:
    test_split_dict = json.load(f)
with open(os.path.join(indices_dir, "train_splits.json"), "r") as f:
    train_split_dict = json.load(f)
with open(os.path.join(indices_dir, "downstr_splits.json"), "r") as f:
    downstr_split_dict = json.load(f)
with open(os.path.join(indices_dir, "to_exclude.json"), "r") as f:
    to_exclude = json.load(f)
with open(os.path.join(indices_dir, "region_counts.json"), "r") as f:
    region_counts = json.load(f)
with open(os.path.join(indices_dir, "dicts.json"), "r") as f:
    dicts = json.load(f)

result = {}
for (dct, split) in (
    (ood_split_dict, "ood"),
    (val_split_dict, "val"),
    (test_split_dict, "test"),
    (train_split_dict, "train"),
    (downstr_split_dict, "all"),
):
    result[split] = random.sample(range(len(dct["paths"])), k=len(dct["paths"]))

excl_num = len(val_split_dict["paths"]) - len(to_exclude["paths"])
result["val_excl"] = random.sample(range(excl_num), k=excl_num)
result["boxes"] = random.sample(
    range(max(region_counts.values())), k=max(region_counts.values())
)
result["dates"] = random.sample(
    range(len(dicts["all_end_dates"])), k=len(dicts["all_end_dates"])
)

with open(os.path.join("data/indices/", "fixed_random_order.json"), "w") as f:
    json.dump(result, f)
