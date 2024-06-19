 IACFormer: A Transformer Framework with Instantaneous Average Convolution for Temporal Action Detection


## Overview


To quickly get start with the model architecture, you can focus mainly on the following files:

- `libs/modeling/blocks.py`
- `libs/modeling/backbones.py`
- `libs/modeling/meta_archs.py`

## Installation

1. Please ensure that you have installed PyTorch and CUDA. **(This code requires PyTorch version >= 1.11. We use
   version=1.11.0 in our experiments)**

2. Install the required packages by running the following command:

```shell
pip install  -r requirements.txt
```

3. Install NMS

```shell
cd ./libs/utils
python setup.py install --user
cd ../..
```

4. Done! We are ready to get start!

## Data Preparation

- We adpot the feature for **THUMOS14**, **ActivityNet** and **Epic-Kitchen** datasets
  from ActionFormer repository ([see here](https://github.com/happyharrycn/actionformer_release)).
  To use these features, please download them from their link and unpack them into the `./data` folder.

- For the **HACS** dataset, we use the [official I3D feature](http://hacs.csail.mit.edu/hacs_segments_features.zip) of
  the RGB stream and the [SlowFast feautre](https://github.com/qinzhi-0110/Temporal-Context-Aggregation-Network-Pytorch)
  from TCANet in our experiments.
  Please unpack the I3D feature into `./data/hacs/i3d_feature` and the SlowFast feature
  into `./data/hacs/slowfast_feature`. We provide processed
  annotation json files for the I3D feature and the SlowFast feature in the `./data/hacs/annotations` folder.
- The folder structure for `./data/hacs/i3d_feature` should be as follows:
  ```
  ./data/hacs/i3d_feature
  |
  |───training/
  │    └───xxx.npy
  │    └───...
  └───validation/
  │    └───xxx.npy
  │    └───...
  └───testing/
  │    └───xxx.npy
  │    └───...
  ```
- The folder structure for `./data/hacs/slowfast_feature` should be as follows:
  ```
  ./data/hacs/slowfast_feature
  |
  |───training/
  │    └───xxx.pkl
  │    └───...
  └───validation/
  │    └───xxx.pkl
  │    └───...

  ```

## Quick Start

We provide a list of scripts that allow you to reproduce our results with just one click. These scripts are located in
the `./tools` folder and include:

- thumos_i3d_script.sh
- epic_noun_slowfast_script.sh
- epic_verb_slowfast_script.sh
- hacs_slowfast_script.sh
- ant_tsp_script.sh

To easily reproduce our results, simply run the following command:

```shell
bash SCRIPT_PATH GPU_NUM
```

For example, if you want to train and eval our model on THUMOS14 dataset using the first GPU on you machine, you can
run:

```shell
bash tools/thumos_i3d_script.sh 0
```

The mean average precision (mAP) results for each dataset are:

| Dataset  | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | Avg   |
|----------|-------|-------|-------|-------|-------|-------|
| THUMOS14 | 82.73 | 79.47 | 72.27 | 61.91 | 46.00 | 68.48 |

| Dataset           | 0.1   | 0.2   | 0.3   | 0.4   | 0.5  | Avg   |
|-------------------|-------|-------|-------|-------|------|-------|
| EPIC-KITCHEN-noun | 26.69 | 25.53 | 23.43 | 20.99 | 17.4 | 22.81 |

| Dataset           | 0.1   | 0.2  | 0.3   | 0.4   | 0.5   | Avg |
|-------------------|-------|------|-------|-------|-------|-----|
| EPIC-KITCHEN-verb | 27.68 | 26.6 | 25.35 | 23.18 | 20.09 |24.58|

| Dataset     | 0.5  | 0.75 | 0.95 | Avg  |
|-------------|------|------|------|------|
| ActivityNet | 54.9 | 38.0 | 8.0  | 36.7 |



*There has been a slight improvement in the results of some datasets compared to those reported in the paper.
*Note: We conduct all our experiments on a single A100-40G GPU and the training results may vary depending on the type of GPU used.

## Test

```shell
python eval.py ./configs/CONFIG_FILE PATH_TO_CHECKPOINT
```

 
