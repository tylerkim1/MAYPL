# Structure Is All You Need: Structural Representation Learning on Hyper-Relational Knowledge Graphs

This is the official implementation of the paper:

**Structure Is All You Need: Structural Representation Learning on Hyper-Relational Knowledge Graphs**

Accepted to the 42nd International Conference on Machine Learning (ICML 2025).

All codes are written by Jaejun Lee (jjlee98@kaist.ac.kr).\
If you use this code, please cite our paper.


## Requirements

We used python 3.9.18 and PyTorch 2.0.1 with cudatoolkit 11.7.

You can install all requirements (except python) with:

```setup
pip install -r requirements.txt
```

## Datasets

Preprocessed versions of all datasets used in the paper, along with the preprocessing scripts, are available in the `./data` directory.

## Reproducing Reported Results using Checkpoints

We provide the checkpoints used to produce all reported results.

To use the checkpoints:
1. Download and unzip `ckpt_MAYPL.zip` file.
2. Place the unzipped `ckpt` folder under the `./code` directory.

You can download the checkpoints from [here](https://drive.google.com/file/d/1USr78S0jiw-uBo_SxknOx2axpoJ0oYeV/view?usp=sharing).

To reproduce results using the checkpoints, run:

```
cd ./ckpt_bash
bash [dataset_name].sh
```

Replace `[dataset_name]` with the appropriate dataset name (e.g., `WD50K`, `WikiPeople-`, `WD20K100v1`, `WK-50`, etc).

## Reproducing Reported Results from Scratch

### Training

To train MAYPL from scratch and reproduce the results reported in the paper:

```
cd ./train_bash
bash [dataset_name].sh
```

### Evaluation

During training, only validation performance is monitored. Final test performance is measured separately using `test.py`.

To evaluate the trained model on the test set:

```
cd ./test_bash
bash [dataset_name].sh
```

## Running on Your Own Dataset
To train MAYPL on a custom dataset:
1. Use `train.py` for training and hyperparameter tuning on the valid set.
2. Use `test.py` for evaluation on the test set.

> **Note**: You may need to tune hyperparameters for optimal performance on new datasets.

## License
Our codes are released under the CC BY-NC-SA 4.0 license.

