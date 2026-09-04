args = {}
import os
import random
import numpy as np
import torch as t
args['path'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckpt')
args['train_mat'] = './data/stdan/NGsim/0113_ratio211/TrainSet.mat'

# args['path'] = 'artifacts/checkpoints/ngsim/1227_data_ratio_augmented/'

# -------------------------------------------------------------------------
# 参数设置
seed = 72
random.seed(seed)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

learning_rate = 0.0005 # 0.0005 1.088e-5
dataset = "ngsim"  # highd ngsim
args['num_worker'] = 16
args['device'] = device

args['f_length'] = 5
args['in_length'] = 31 # 3 seconds
args['out_length'] = 50 # 5 seconds

args['lstm_encoder_size'] = 128
args['traj_linear_hidden'] = 64
args['batch_size'] = 128 # 128

args['relu'] = 0.1 # relu의 기울기
args['epoch'] = 200

args['use_elu'] = True

args['d_s'] = 1
args['dropout'] = 0.2
args['lat_length'] = 3
args['lon_length'] = 3  