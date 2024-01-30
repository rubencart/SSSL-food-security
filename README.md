# SSSL-food-security
Code and data export scripts accompanying our article 
"[Spatiotemporal self-supervised pre-training on satellite imagery improves food insecurity prediction](https://www.cambridge.org/core/journals/environmental-data-science/article/spatiotemporal-selfsupervised-pretraining-on-satellite-imagery-improves-food-insecurity-prediction/47FDCFF96FF9A99D31548C1539D506A5)".
## Set up

Dependencies
- pytorch
- pytorch-lightning
- typed-argument-parser
- tqdm
- rasterio 1.3
- geopandas 0.8.1
- pyyaml
- wandb
- h5py
- scikit-learn
- matplotlib 

```
conda create -n ssslenv python=3.10 ipython
conda activate ssslenv
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge rasterio=1.3.0.post1 gdal=3.5.1 poppler=22.04.0 pytorch-lightning=1.8.6
# if error `ImportError: libLerc.so.4: cannot open shared object file: No such file or directory` when `import rasterio`:
# conda install -c conda-forge lerc=4.0.0
pip install geopandas tqdm wandb scikit-learn typed-argument-parser matplotlib seaborn
conda install h5py -c conda-forge
# https://github.com/ContinuumIO/anaconda-issues/issues/10351
# conda install "poppler<0.62"
# conda install -c conda-forge poppler=21.09 gdal=3.3.3
# conda install -c conda-forge poppler=22.04 gdal=3.5.1 rasterio=1.3 --force-reinstall
```
```
rasterio                  1.3.0.post1     py310h1bedc6d_0    conda-forge
geopandas                 0.8.1              pyhd3eb1b0_0  
geopandas-base            0.11.1             pyha770c72_0    conda-forge
gdal                      3.5.1           py310hb7951cf_2    conda-forge
poppler                   22.04.0              h1434ded_1    conda-forge
poppler-data              0.4.11               hd8ed1ab_0    conda-forge
pytorch                   1.13.0          py3.10_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cuda              11.7                 h67b0de4_0    pytorch
```

Finally (!) install the sssl package as module so the scripts can import it: 
run from the repository root directory:
```
pip install -e .
```

## Data

- Start date: '2013-05-01'
- End date 3-month intervals: '2015-11-01'
- End date 4-month intervals: '2020-03-01'
- Total months: 7y * 12m - 2m = 82m


We took 2013-05-01 as start date for GEE images, which means the first composite is from 2013-05-01 until 2013-08-01,
from which we predict the IPC score corresponding to 2013-08-01.
This means that we can't predict a score for 2013-05-01, even though in the .csv file, there is a score for this date.
If we wanted to include this date, we should have used 2013-02-01 as start date in GEE.
We exclude this date from data (no change needed in neural net code, since no exported tile corresponds to this date,
but not in e.g. random forest code since that uses the .csv).

## Download & Preprocess

Make sure you ran `pip install -e .` first!

IPC scores:
- Extract `data/predicting_food_crises_data.zip`

If running on different country/region/timeframe than Somalia 2013-2020:
- Adjust and run `scripts/preprocess/preprocess_ipc_csv.py`
- Do the set-up under 'Tiles' below
- Run `scripts/preprocess/ipc_class_weights.py` to recompute relative weights of IPC classes 
if using different data than Somalia 2013-2020 and update in `utils.Constants`

Tiles:
- Download data from GEE with code in `scripts/earth_engine/js/export_somalia.js`
- Download from google cloud to local server
- Run `scripts/preprocess/build_indices.py`. The indices (and output of the next 2 steps)
used for results in the publication are included in `data/indices.zip`. You can
hence skip this and the subsequent 2 items by extracting that zip file.
- Check for tiles that don't have enough positives with `scripts/preprocess/search_tiles_not_enough_neighbors.py`
- Run `scripts/preprocess/pseudo_random_order.py` so tiles in val set for pretraining always have
the same positives.
- Run `scripts/preprocess/make_h5.py` (optional but much faster), use `cfg.use_h5 = True` in subsequent runs
- Run `scripts/preprocess/channel_mean_std.py` if using different data than Somalia Landsat8 2013-2020, update means/stds in `utils.Constants`

Config files:
- Run `config/generate_pretrain_configs.py` to generate pretrain run config files, to run with `python code/pretrain.py --cfg config/pretrain/<config>.yaml`
- Run `config/generate_downstr_configs.py` to generate downstream IPC run config files, to run with `python code/finetune.py --cfg config/ipc/<config>.yaml`
- Run `config/generate_comb_configs.py` to generate pretrain run config files, to run with `python code/pretrain_then_finetune.py --cfg config/pretrain_ipc/<config>.yaml`


## Run

Pretrain:
```
CUDA_VISIBLE_DEVICES=0 python code/pretrain.py --cfg config/pretrain/debug.yaml
```

Finetune:
```
CUDA_VISIBLE_DEVICES=0 python code/finetune.py --cfg config/ipc/debug.yaml
```

Pretrain then finetune:
```
CUDA_VISIBLE_DEVICES=0 python code/pretrain_then_finetune.py --cfg config/pretrain_ipc/debug.yaml --seed 41
```

## Plots and results

- Run the `limit_plot` function in `scripts/results/loss_and_threshold_plots.py` to generate plots like in Figure 5.
- Run `scripts/results/tile_ipc_plots.py` to generate plots like in Figure 2, 3, 4.
- Run `scripts/results/seasonality_plots.py` to generate a plot like in Figure 8.
- Run `scripts/results/less_data_future_plots.py` to generate plots like in Figure 6 and 7.
- Run `scripts/results/deeplift_shap_plots.py` to generate SHAP value plots like in Figure 9.
