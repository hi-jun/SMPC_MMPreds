import torch as t

def get_stdan_args(dt=int, history=int, future=int):
    """
    stdan 모델을 위한 하이퍼파라미터 딕셔너리를 생성하고 반환합니다.
    """
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    args_stdan = {}
    args_stdan['dt']= dt
    args_stdan['num_worker'] = 0
    args_stdan['device'] = device
    args_stdan['lstm_encoder_size'] = 64
    args_stdan['n_head'] = 4 # multi-head attention 헤드 갯수
    args_stdan['att_out'] = 48 # attention output 차원?

    args_stdan['f_length'] = 5 # ?
    args_stdan['traj_linear_hidden'] = 32
    args_stdan['use_elu'] = True
    args_stdan['dropout'] = 0.2
    args_stdan['relu'] = 0.1 # relu의 기울기
    args_stdan['lat_length'] = 3
    args_stdan['lon_length'] = 3  # 2 2로 하니 에러 떠서 우선 3으로 0904
    args_stdan['use_true_man'] = False # training시 예측 의도를 사용할지 실제 의도를 사용할지
    args_stdan['epoch'] = 20
    args_stdan['use_spatial'] = False  # Residual connection 시 spa_values도 connect 할 것인지? 

    # HJ
    args_stdan['grid_size'] = (13,3)
    args_stdan['in_length'] = int(history/dt+1) # 3 seconds
    args_stdan['out_length'] = int(future/dt) # 5 seconds

    args_stdan['train_flag'] = False

    args_stdan['use_maneuvers'] = True
    args_stdan['cat_pred'] = True
    args_stdan['use_mse'] = False
    args_stdan['pre_epoch'] = 6
    args_stdan['val_use_mse'] = False

    
    return args_stdan