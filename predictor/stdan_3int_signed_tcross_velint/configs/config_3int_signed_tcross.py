from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "stdan_3int_signed_tcross_velint"
SOURCE_DATA_ROOT = REPO_ROOT / "experiments" / "stdan_3int_signed_tcross" / "data" / "processed_3int_signed_tcross"
DATA_ROOT = EXP_ROOT / "data" / "processed_3int_signed_tcross_velint"
CHECKPOINT_ROOT = EXP_ROOT / "checkpoints_out30"
WANDB_DIR = EXP_ROOT / "wandb"
FIGURE_ROOT = EXP_ROOT / "eval" / "figures"

TRAIN_MAT = DATA_ROOT / "TrainSet.mat"
VAL_MAT = DATA_ROOT / "ValSet.mat"
TEST_MAT = DATA_ROOT / "TestSet.mat"

seed = 72
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 5e-4
dataset = "ngsim_3int_signed_tcross_velint"

args = {
    "seed": seed,
    "device": device,
    "path": str(CHECKPOINT_ROOT),
    "train_mat": str(TRAIN_MAT),
    "val_mat": str(VAL_MAT),
    "test_mat": str(TEST_MAT),
    "source_data_root": str(SOURCE_DATA_ROOT),
    "wandb_dir": str(WANDB_DIR),
    "wandb_project": "stdan_v2_velint",
    "figure_root": str(FIGURE_ROOT),
    "num_worker": 8,
    "lstm_encoder_size": 64,
    "n_head": 4,
    "att_out": 48,
    "in_length": 31,
    "out_length": 30,
    "dt": 0.1,
    "prediction_target": "velocity_integrated_to_position",
    "f_length": 5,
    "traj_linear_hidden": 32,
    "batch_size": 128,
    "use_elu": True,
    "dropout": 0.5,
    "relu": 0.1,
    "num_intentions": 3,
    "lat_length": 3,
    "lon_length": 0,
    "use_true_man": False,
    "epoch": 50,
    "use_spatial": False,
    "intention_weight": 1.0,
    "lambda_endpoint":0.1,
    "lambda_tcross": 10.0,
    "t_cross_label_type": "signed_normalized_relative_step",
    "t_cross_norm_denominator": 28.0,
    "use_maneuvers": True,
    "cat_pred": True,
    "use_mse": False,
    "pre_epoch": 10,
    "val_use_mse": True,
    "train_flag": True,
    "grad_clip": 10.0,
    "early_stop_patience": 20,
}
