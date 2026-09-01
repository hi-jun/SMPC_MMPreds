args = {}
import random
import numpy as np
import torch as t
import model5f_mult_vel as model
args['path'] = 'checkponint/vel_prediction/0113_ratio211/'
# args['path'] = 'checkponint/ngsim/1227_data_ratio_augmented/'

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
args['lstm_encoder_size'] = 64
args['n_head'] = 4 # multi-head attention 헤드 갯수
args['att_out'] = 48 # attention output 차원?
args['in_length'] = 31 # 3 seconds
args['out_length'] = 50 # 5 seconds
args['f_length'] = 5 # ?
args['traj_linear_hidden'] = 32
args['batch_size'] = 128 # 128
args['use_elu'] = True
args['dropout'] = 0.5
args['relu'] = 0.1 # relu의 기울기
args['lat_length'] = 3
args['lon_length'] = 3  # 2 2로 하니 에러 떠서 우선 3으로 0904
args['use_true_man'] = False # training 시 예측 의도를 사용할지 실제 의도를 사용할지
args['epoch'] = 200
args['use_spatial'] = False # spatial feature를 residual로 사용할지?

# 多模态 multi-modal
args['intention_weight'] = 1e0

args['use_maneuvers'] = True

# 单模态 是否 拼接 预测意图 : single-modal일 경우, 주행 의도를 특징 벡터에 이어 붙일 것인지
args['cat_pred'] = True # 이어붙일 경우, multi-modal 의도를 고려한 single-modal output 

args['use_mse'] = False # MSE만 가지고 학습할 것인지
args['pre_epoch'] = 10 # pre_epoch 만큼은 MSE 학습
args['val_use_mse'] = True # val시 mse 사용할지


# -------------------------------------------------------------------------
