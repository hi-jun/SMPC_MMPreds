from typing import Dict, List, NewType, Any, Tuple
from collections import deque
from enum import Enum
import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import pandas as pd

import math
from scipy.spatial.transform import Rotation as R
import json
import os

from prediction.trajectron.trajectron.model.model_registrar import ModelRegistrar
from prediction.trajectron.trajectron.environment import Environment, Scene, Node, derivative_of

import glob

from prediction.trajectron.trajectron.model.trajectron import Trajectron

ObjectID = NewType('ObjectID,', int)
TargetID = NewType('TargetID,', int)

class utils():
    def __init__(self):
        pass

    def quaternion_to_euler(self, x, y, z, w):
        r = R.from_quat([x, y, z, w])
        return r.as_euler('xyz', degrees=False)

    def transformation(self, points, origin_x, origin_y, yaw, inverse=False):
        # 정방향 변환(-yaw)에 사용할 회전 행렬
        # cos(-yaw) = cos(yaw), sin(-yaw) = -sin(yaw)
        rotation_matrix = np.array([ # 표준 회전 행렬의 공식 (반시계 방향 yaw만큼 좌표 변환)
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])

        # R의 Transpose = -yaw만큼 좌표계 회전
        if inverse: # local to global
            rotated_points = np.dot(points, rotation_matrix.T) # 순서 변경 및 행렬 수정
            translated_points = rotated_points + np.array([origin_x, origin_y])
        else: # global to local
            translated_points = points - np.array([origin_x, origin_y]) # 원점 기준 이동
            translated_points = translated_points @ rotation_matrix   # global 좌표계를 local 좌표 모양으로 돌린다 ( = 시계방향으로 yaw만큼 돌려야 함 )

        return translated_points

    def AngleToRad(self,theta):
        if theta < 0:
            theta = theta + 360
        theta = theta*np.pi/180
        return theta
    
    def get_current_idx(self,state,GT):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(GT,np.ndarray):
            GT = np.array(GT)
        
        dist = np.sqrt(np.sum((GT - state[:2])**2,axis=2)) # (3,129)
        idx_dim0 = np.argmin(np.min(dist,axis=1)) # (3,)
        idx_dim1 = np.argmin(np.min(dist,axis=0)) # (129,)
        current_idx = (idx_dim0,idx_dim1)
        
        return current_idx
    
    #TODO : Linear에서도 동작하도록 수정 

class dataloader(utils):
    def __init__(self,dt):
        super().__init__()
        self.dt = dt
    
    def load_csv(self,csv_dir): # loader 따로 만들기
        # 데이터 로드, 속도 계산, 좌표 회전 TODO : To function 
        all_agent_data = {} # {id:data}
        csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
        if not csv_files:
            print(f"'{csv_dir}' 폴더에 CSV 파일이 없습니다. 프로그램을 종료합니다.")
            return
        all_dfs_for_range = []
        for file_path in csv_files:
            object_id = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_csv(file_path)
            df.index = pd.to_timedelta(df['time'], unit='s')
            
            df_resampled = df.resample(f"{int(self.dt * 1000)}ms").first().dropna() # 0.1 hz
            agent_data_no_speed = df_resampled[['x', 'y', 'yaw']].to_numpy()
            xy_data = agent_data_no_speed[:, :2]
            distances = np.sqrt(np.sum(np.diff(xy_data, axis=0)**2, axis=1))
            speeds_ms = distances / self.dt
            speeds_kmh = speeds_ms * 3.6
            final_speeds_kmh = np.insert(speeds_kmh, 0, 0.0)
            agent_data = np.hstack((agent_data_no_speed, final_speeds_kmh.reshape(-1, 1)))
            all_agent_data[object_id] = agent_data
            all_dfs_for_range.append(df_resampled)

        combined_df = pd.concat(all_dfs_for_range)
        object_ids = list(all_agent_data.keys())
    
        return all_agent_data, object_ids

    def init_rotate(self, all_agent_data:dict):
        # 기준 차량 찾기
        try:
            reference_id = [oid for oid in all_agent_data.keys() if 'BlockedNbr_2' in oid][0]
        except IndexError:
            print("경고: 'Ego' 차량을 찾을 수 없어 회전을 수행하지 않습니다.")
            return all_agent_data # Ego가 없으면 원본 데이터 반환

        reference_yaw = all_agent_data[reference_id][0, 2]
        print(f"기준 차량 '{reference_id}'의 초기 yaw: {reference_yaw:.4f} (rad)")
        print("이 값을 기준으로 모든 좌표를 회전해 시각화를 개선합니다.")

        rotated_agent_data = {}
        for object_id, agent_data in all_agent_data.items():
            # 복사본을 만들어 원본 데이터를 수정하지 않도록 합니다.
            rotated_data = agent_data.copy()

            # 2. transformation 함수를 사용하여 x, y 좌표를 회전합니다.
            # 원점(0,0)을 기준으로 회전하므로 origin_x, origin_y는 0입니다.
            # global to local 변환을 사용하여 기준 yaw만큼 좌표를 회전시킵니다.
            points_to_rotate = agent_data[:, :2] # [x, y] 좌표만 추출
            rotated_points = self.transformation(
                points=points_to_rotate,
                origin_x=0,
                origin_y=0,
                yaw=reference_yaw,
                inverse=False
            )

            # 3. 회전된 좌표와 정규화된 yaw로 데이터를 업데이트합니다.
            rotated_data[:, :2] = rotated_points  # 회전된 [x, y] 좌표로 교체
            rotated_data[:, 2] -= reference_yaw   # 각 agent의 yaw도 기준 yaw만큼 빼서 정규화, -> 직진만 하는 차량들의 heading은 0
            rotated_agent_data[object_id] = rotated_data
        return rotated_agent_data

class utils_linear():
    def __init__(self):
        pass
    
    def predict(self, model, target_id, target_state, history_data): # 다른 predictor와 input 맞추기 위해
        prediction = model(target_id, history_data) # (1,50,2)
        best_traj = prediction[0]
        return best_traj, prediction, None, None, None # STDAN과 길이 맞추기 위해

class utils_CSlstm(utils):
    def __init__(self,args,is_carla_test):
        super().__init__()
        self.device = args['device']
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size'] 
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size'] #(13,3)
        self.input_embedding_size = args['input_embedding_size']
        self.use_maneuvers = args['use_maneuvers']
        
        self.is_carla_test = is_carla_test

    def make_tensor(self, TV_id, TV_state, trackings, no_nbsr=False):

        # 1. tsr initialize
        hist_tsr = torch.zeros(self.in_length, 1, 2, device=self.device)
        mask_tsr = torch.zeros(1, self.grid_size[1], self.grid_size[0], self.encoder_size, device=self.device)

        if len(trackings[TV_id]) < self.in_length: # Not enough history 
            print(f'Tracking length is not enough (needs {self.in_length}), returning Zero tensors')
            n_objs = 0
            nbrs_tsr = torch.zeros(self.in_length, n_objs, 2, device=self.device)
            return hist_tsr, nbrs_tsr, mask_tsr.bool() # return zero tsr 
        
        
        # 2. Target vehicle
        
        if self.is_carla_test:
            hist = self.transformation(np.array(trackings[TV_id])[-self.in_length:, :2], TV_state[0], TV_state[1], TV_state[2]) # 1) global to local -> 2) LHS to RHS 순으로 해야 함. yaw의 회전 방향이 달라지니까
            hist = hist[:,[1,0]]
        else:
            hist = self.transformation(np.array(trackings[TV_id])[-self.in_length:, :2], TV_state[0], TV_state[1], TV_state[2] - math.pi / 2.0)
        hist_tsr[:, 0, :] = torch.from_numpy(hist * 3.281).float() # meter to feet 

        if no_nbsr: # for Only-TV-data test 
            n_objs = 0
            nbrs_tsr = torch.zeros(self.in_length, n_objs, 2, device=self.device)
            return hist_tsr, nbrs_tsr, mask_tsr

        # 3. Nbrs
        nbrs_ids = [nbr_id for nbr_id in trackings.keys() if nbr_id != TV_id]
        
        # (grid idx, history_tsr)
        valid_neighbors_data = [] # 
        
        num_lat_cells, num_lon_cells = self.grid_size[1], self.grid_size[0]
        # ... (그리드 설정 코드는 동일) ...
        cell_lat_size, cell_lon_size = 3.5, 4.6
        lat_min = -(num_lat_cells // 2) * cell_lat_size - (cell_lat_size / 2)
        lat_max = (num_lat_cells // 2) * cell_lat_size + (cell_lat_size / 2)
        lon_min = -(num_lon_cells // 2) * cell_lon_size - (cell_lon_size / 2)
        lon_max = (num_lon_cells // 2) * cell_lon_size + (cell_lon_size / 2)
        lat_grid_edges = np.linspace(lat_min, lat_max, num_lat_cells + 1)
        lon_grid_edges = np.linspace(lon_min, lon_max, num_lon_cells + 1)

        for nbr_id in nbrs_ids:
            if len(trackings[nbr_id]) < self.in_length: # Not enough history 
                continue
            
            if self.is_carla_test:
                nbr = self.transformation(np.array(trackings[nbr_id])[-self.in_length:, :2], TV_state[0], TV_state[1], TV_state[2])
                nbr = nbr[:,[1,0]]
            else:
                nbr = self.transformation(np.array(trackings[nbr_id])[-self.in_length:, :2], TV_state[0], TV_state[1], TV_state[2] - math.pi / 2.0)
            nbr_curpose = nbr[-1]
            
            grid_lat = np.digitize(nbr_curpose[0], lat_grid_edges) - 1
            grid_lon = np.digitize(nbr_curpose[1], lon_grid_edges) - 1
            
            if 0 <= grid_lat < num_lat_cells and 0 <= grid_lon < num_lon_cells:
                # grid idx for sort
                idx = grid_lat * num_lon_cells + grid_lon
                
                # nbr_tsr
                nbr_hist_tensor = torch.from_numpy(nbr * 3.281).float()
                valid_neighbors_data.append((idx, nbr_hist_tensor))
                
                # set mask tsr
                mask_tsr[0, grid_lat, grid_lon, :] = 1.0

        # Set nbrs_tsr 
        if valid_neighbors_data:
            valid_neighbors_data.sort(key=lambda x: x[0])
            sorted_hist_tensors = [item[1] for item in valid_neighbors_data]             
            nbrs_tsr = torch.stack(sorted_hist_tensors, dim=1)

        else:
            # 유효한 주변 차량이 없는 경우
            n_objs = 0
            nbrs_tsr = torch.zeros(self.in_length, n_objs, 2)
        nbrs_tsr = nbrs_tsr.to(self.device)    
        
        # ### <<< 검증 코드 >>>
        # print("\n" + "---" * 15)
        # print(f"CS-LSTM, tv_id: {TV_id}")
        
        # # 1. 마스크에서 'True'인 그리드 좌표 찾기
        # #    (1, 3, 13, 64) -> (3, 13, 64) -> (3, 13) -> (N, 2)
        # occupancy_grid = mask_tsr.squeeze(0).any(axis=-1) # 특징 차원 축소
        # occupied_indices = occupancy_grid.nonzero(as_tuple=False) # True인 (lat, lon) 좌표 추출
        
        # # 1차원 인덱스 순서와 동일하게 정렬
        # occupied_indices_list = sorted(occupied_indices.tolist(), key=lambda pos: pos[0] * self.grid_size[0] + pos[1])
        
        # print(f"Occupied Grid Cells (lat, lon): {occupied_indices_list}")
        
        # # 2. 최종 이웃 텐서의 모양 확인
        # print(f"Final nbrs_tsr Shape: {nbrs_tsr.shape}")
        
        # # 3. 순서 일치 확인
        # if nbrs_tsr.shape[1] == len(occupied_indices_list):
        #     print("Verification: Number of neighbors matches mask. Checking order...")
        #     for i in range(nbrs_tsr.shape[1]):
        #         # 정렬된 마스크 좌표
        #         mask_pos = occupied_indices_list[i] # 
                
        #         # nbrs_tsr의 i번째 이웃 데이터 (마지막 시점의 좌표만 확인)
        #         nbr_last_pos = nbrs_tsr[-1, i, :].cpu().numpy() / 3.281 # 미터 단위로 변환
                
        #         print(f"  -> Neighbor #{i+1} in tensor (last pos: [{nbr_last_pos[0]:.2f}, {nbr_last_pos[1]:.2f}]) corresponds to Mask at (lat:{mask_pos[0]}, lon:{mask_pos[1]})")
        #     print("✅ Order seems correct.")
        # else:
        #     print("❌ WARNING: Mismatch between number of neighbors in tensor and mask!")
        # print("---" * 15)
        # ### <<< 검증 코드 종료 >>>

        return hist_tsr, nbrs_tsr, mask_tsr.bool()
    
    def predict(self, model, target_id, target_state, history_data):
        """
        Prediction using CS-LSTM

        Args:
            model: CS-LSTM PyTorch model
            target_id (str): TV
            target_state (list): TV 현재 상태 [x, y, yaw].
            history_data (dict): 모든 활성 객체의 과거 경로 데이터.
        Returns:
            tuple: (예측 궤적 텐서, 측면 예측 텐서, 종방향 예측 텐서)
        """
        # If tracking is not enough : predict using zero tensor 
        
        # Make model input, return : torch.tsr, torch.tsr, torch.tsr
        hist_cs, nbrs_cs, mask_cs = self.make_tensor(target_id, target_state, history_data, no_nbsr=False)
        if self.use_maneuvers: lat_enc, lon_enc = torch.zeros(1,3).to(self.device), torch.zeros(1,2).to(self.device)
        else: lat_enc, lon_enc = torch.empty(1,1).to(self.device), torch.empty(1,1).to(self.device)

        # Prediction
        predictions, lat_predictions, lon_predictions = model(
            hist_cs, 
            nbrs_cs, 
            mask_cs, 
            lat_enc, 
            lon_enc
        )
        best_traj, all_trajs, probabilities = self.postprocessing(predictions,lat_predictions,lon_predictions, target_state)
        
        return best_traj, all_trajs, probabilities, None, None # STDAN과 output 길이 맟주기 위해
    
    def postprocessing(self, predictions, lat_probs, lon_probs, target_state):
        '''
        Post-processing e.g. feter-meter, coordinate transformation..
        '''
        
        # Multi-Modal
        if lat_probs is not None and lon_probs is not None: 
            # ----- trajectories -----  
            if not isinstance(predictions, np.ndarray): # (n,FH,2)
                predictions = np.array(predictions)
            if predictions.shape[2] == 1: # CS-LSTM (K, future, 1, 5), 모델별 squeeze axis가 다를 수 있으므로 확인 TODO : 어떤 shape의 input이던 하나의 shape으로 통일
                predictions = np.squeeze(predictions, axis=2)
            predictions = predictions * 0.3048 # feet to meter

            # ----- Probabilities -----  
            probabilities = np.outer(lon_probs, lat_probs).flatten() # outer product != cross product (outer : matrix, cross : vector)
            
            predictions = predictions[:,:,:2] # only means TODO : 분산값 쓸거면 바꿀 것
            if self.is_carla_test:
                predictions = predictions[:,:,[1,0]] # RHS to LHS
                all_trajs = self.transformation(predictions, target_state[0], target_state[1], target_state[2], inverse=True)
            else:
                all_trajs = self.transformation(predictions, target_state[0], target_state[1], target_state[2] - np.pi/2.0, inverse=True)
            
            # 가장 확률 높은 경로를 best_trajectory로 선택
            top_1_index = np.argmax(probabilities)
            best_traj = all_trajs[top_1_index]
            
            return best_traj, all_trajs, probabilities
            
        # Uni-Modal
        else:
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            predictions = np.squeeze(predictions, axis=1) * 0.3048
        
            predictions = predictions[:,:2] # only means TODO : 분산값 쓸거면 바꿀 것
            if self.is_carla_test:
                predictions = predictions[:,[1,0]] # RHS to LHS
                best_traj = self.transformation(predictions, target_state[0], target_state[1], target_state[2], inverse=True)
            else : 
                best_traj = self.transformation(predictions[:, :2], target_state[0], target_state[1], target_state[2] - np.pi/2.0, inverse=True)
            all_trajs = [best_traj] # 자료형 통일을 위해 리스트에 담음
            probabilities = None # 확률 정보 없음
            
            return best_traj, all_trajs, probabilities 

    
class utils_STDAN(utils):
    def __init__(self,args, is_carla_test):
        self.dt = args['dt']
        self.device = args['device']
        self.encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.traj_linear_hidden = args['traj_linear_hidden']
        self.use_maneuvers = args['use_maneuvers']
        self.use_elu = args['use_elu']
        self.use_spatial = args['use_spatial']
        self.dropout = args['dropout']
        self.grid_size = args['grid_size'] #(13,3)
        
        self.is_carla_test = is_carla_test

    def make_tensor(self, target_id, target_state, trackings, no_nbsr=False):
        """
        device를 인자로 받아 텐서를 해당 장치에 직접 생성합니다.
        """
        n_objs = len(trackings.keys()) - 1 if not no_nbsr else 0
        full_length = self.in_length + 2

        # --- 1. Target 차량 및 기본 텐서 초기화 ---
        hist_tsr = torch.zeros(self.in_length, 1, 2, device=self.device)
        va_tsr = torch.zeros(self.in_length, 1, 2, device=self.device)
        mask_tsr = torch.zeros(1, self.grid_size[1], self.grid_size[0], self.encoder_size, device=self.device)
        lane_tsr = torch.zeros(self.in_length, 1, 1, device=self.device)
        cls_tsr = torch.full((self.in_length, 1, 1), 2.0, device=self.device)

        # --- 조기 반환 시 사용할 비어있는 nbrs 텐서 생성 로직 ---
        def create_empty_nbrs_tensors(n_objs=0):
            nbrs_tsr = torch.zeros(self.in_length, n_objs, 2, device=self.device)
            nbrsva_tsr = torch.zeros(self.in_length, n_objs, 2, device=self.device)
            nbrslane_tsr = torch.zeros(self.in_length, n_objs, 1, device=self.device)
            nbrscls_tsr = torch.full((self.in_length, n_objs, 1), 2.0, device=self.device)
            return nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr

        if len(trackings[target_id]) < full_length:
            print(f'Tracking length is not enough (needs {full_length}), returning Zero tensors')
            nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr = create_empty_nbrs_tensors()
            return hist_tsr, nbrs_tsr, mask_tsr.bool(), va_tsr, nbrsva_tsr, lane_tsr, nbrslane_tsr, cls_tsr, nbrscls_tsr

        # --- 2. Target(Ego) 차량 데이터 처리 ---
        if self.is_carla_test:
            full_hist_np = self.transformation(np.array(trackings[target_id])[-full_length:, :2], target_state[0], target_state[1], target_state[2])
            full_hist_np = full_hist_np[:,[1,0]]

        else:
            full_hist_np = self.transformation(np.array(trackings[target_id])[-full_length:, :2], target_state[0], target_state[1], target_state[2]-math.pi/2.0)
        
        full_hist_tsr = torch.from_numpy(full_hist_np * 3.281).float().to(self.device)
        
        # 1. full_length-1 길이의 속도 벡터 계산
        full_vel_tsr = (full_hist_tsr[1:] - full_hist_tsr[:-1]) / self.dt

        # 2. full_length-2 (즉, in_length) 길이의 가속도 벡터 계산
        full_acc_tsr = (full_vel_tsr[1:] - full_vel_tsr[:-1]) / self.dt

        # 3. 모델에 입력할 최종 in_length 만큼의 데이터만 슬라이싱
        hist_tsr[:, 0, :] = full_hist_tsr[-self.in_length:]
        final_vel_tsr = full_vel_tsr[-self.in_length:] # only 양수
        final_acc_tsr = full_acc_tsr # 이미 길이가 in_length 이므로 그대로 사용

        # 4. 내적 값을 속력(speed, ||v||)으로 나누어 스칼라 가속도 계산
        speed = torch.norm(final_vel_tsr, p=2, dim=-1, keepdim=True)

        dot_product = (final_acc_tsr * final_vel_tsr).sum(dim=-1, keepdim=True) # dot(a,v)
        epsilon = 1e-8 # speed 0일 때 error 발생 막기 위해
        signed_acceleration = dot_product / (speed + epsilon) # dot(a,v)/abs(speed) = v 방향으로의 accel값
        
        va_tsr[:,0,:] = torch.cat((speed, signed_acceleration), dim=-1)
        
        if no_nbsr:
            # <<<--- 수정된 부분: no_nbsr=True일 때 반환할 비어있는 텐서들을 여기서 생성합니다.
            nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr = create_empty_nbrs_tensors()
            return hist_tsr, nbrs_tsr, mask_tsr.bool(), va_tsr, nbrsva_tsr, lane_tsr, nbrslane_tsr, cls_tsr, nbrscls_tsr
            
        ### <<< 변경됨: Single-Pass with Sorting 방식으로 주변 차량 처리 >>>
        
        valid_neighbors_data = [] # (그리드 인덱스, hist_tsr, va_tsr) 튜플을 저장할 리스트
        nbrs_ids = [nbr_id for nbr_id in trackings.keys() if nbr_id != target_id]
        
        # 그리드 경계 계산
        cell_lat_size, cell_lon_size = 3.5, 4.6
        lat_min = -(self.grid_size[1] // 2) * cell_lat_size - (cell_lat_size / 2)
        lat_max = (self.grid_size[1] // 2) * cell_lat_size + (cell_lat_size / 2)
        lon_min = -(self.grid_size[0] // 2) * cell_lon_size - (cell_lon_size / 2)
        lon_max = (self.grid_size[0] // 2) * cell_lon_size + (cell_lon_size / 2)
        lat_grid_edges = np.linspace(lat_min, lat_max, self.grid_size[1] + 1)
        lon_grid_edges = np.linspace(lon_min, lon_max, self.grid_size[0] + 1)

        # --- Pass 1: 모든 주변 차량의 데이터를 *한 번만* 처리하고, 유효한 차량만 저장 ---
        for nbr_id in nbrs_ids:
            if len(trackings[nbr_id]) < full_length:
                continue
            
            # 1. 모든 계산 수행 (좌표 변환, 속도/가속도, 텐서 생성)
            if self.is_carla_test:
                nbr_full_hist_np = self.transformation(np.array(trackings[nbr_id])[-full_length:, :2], target_state[0], target_state[1], target_state[2])
                nbr_full_hist_np = nbr_full_hist_np[:,[1,0]]
            else:
                nbr_full_hist_np = self.transformation(np.array(trackings[nbr_id])[-full_length:, :2], target_state[0], target_state[1], target_state[2]-math.pi/2.0)
            nbr_full_hist_tsr = torch.from_numpy(nbr_full_hist_np * 3.281).float().to(self.device) # to feet
            nbr_full_vel_tsr = (nbr_full_hist_tsr[1:] - nbr_full_hist_tsr[:-1]) / self.dt
            nbr_full_acc_tsr = (nbr_full_vel_tsr[1:] - nbr_full_vel_tsr[:-1]) / self.dt
            
            nbr_hist_tsr = nbr_full_hist_tsr[-self.in_length:]
            nbr_final_vel_tsr = nbr_full_vel_tsr[-self.in_length:]
            nbr_final_acc_tsr = nbr_full_acc_tsr
            
            nbr_speed = torch.norm(nbr_final_vel_tsr, p=2, dim=-1, keepdim=True)
            
            dot_product = (nbr_final_acc_tsr * nbr_final_vel_tsr).sum(dim=-1, keepdim=True) # dot(a,v)
            epsilon = 1e-8 # speed 0일 때 error 발생 막기 위해
            signed_nbr_accel = dot_product / (nbr_speed + epsilon) # dot(a,v)/abs(speed) = v 방향으로의 accel값
        
            nbr_va_tsr = torch.cat((nbr_speed, signed_nbr_accel), dim=-1)
            
            # 2. 그리드 위치 계산
            nbr_curpose = nbr_full_hist_np[-1]
            grid_lat = np.digitize(nbr_curpose[0], lat_grid_edges) - 1
            grid_lon = np.digitize(nbr_curpose[1], lon_grid_edges) - 1
            
            # 3. 그리드 내에 있는 경우, (그리드 인덱스, 처리된 텐서들)을 리스트에 추가
            if 0 <= grid_lat < self.grid_size[1] and 0 <= grid_lon < self.grid_size[0]:
                # 1차원 그리드 인덱스 계산 (C-style, row-major)
                flat_grid_index = grid_lat * self.grid_size[0] + grid_lon
                valid_neighbors_data.append((flat_grid_index, nbr_hist_tsr, nbr_va_tsr))
                
                # 마스크는 바로 설정
                mask_tsr[0, grid_lat, grid_lon, :] = 1.0
        
        # --- 4. 유효한 이웃들로 최종 텐서 구성 ---
        if valid_neighbors_data:
            # 4a. 그리드 인덱스를 기준으로 리스트를 정렬
            valid_neighbors_data.sort(key=lambda x: x[0])
            
            # 4b. 정렬된 튜플에서 각 텐서들을 추출
            sorted_hist_tensors = [item[1] for item in valid_neighbors_data]
            sorted_va_tensors = [item[2] for item in valid_neighbors_data]
            
            # 4c. 정렬된 텐서 리스트를 쌓아 최종 텐서 생성
            n_objs = len(valid_neighbors_data)
            nbrs_tsr = torch.stack(sorted_hist_tensors, dim=1)
            nbrsva_tsr = torch.stack(sorted_va_tensors, dim=1)
            
            nbrslane_tsr = torch.zeros(self.in_length, n_objs, 1, device=self.device)
            nbrscls_tsr = torch.full((self.in_length, n_objs, 1), 2.0, device=self.device)
        else: # 유효한 주변 차량이 없는 경우
            nbrs_tsr, nbrsva_tsr, nbrslane_tsr, nbrscls_tsr = create_empty_nbrs_tensors()

        ### <<< 검증 코드 시작 (수정됨) >>>
        print("\n" + "---" * 15)
        print(f"STDAN, TV_id: {target_id}")
        
        # 1. 마스크에서 'True'인 그리드 좌표 찾기
        #    (1, 3, 13, 64) -> (3, 13, 64) -> (3, 13) -> (N, 2)
        occupancy_grid = mask_tsr.squeeze(0).any(axis=-1) # 특징 차원 축소
        occupied_indices = occupancy_grid.nonzero(as_tuple=False) # True인 (lat, lon) 좌표 추출
        
        # 1차원 인덱스 순서와 동일하게 정렬
        occupied_indices_list = sorted(occupied_indices.tolist(), key=lambda pos: pos[0] * self.grid_size[0] + pos[1])
        
        print(f"Occupied Grid Cells (lat, lon): {occupied_indices_list}")
        
        # 2. 최종 이웃 텐서의 모양 확인
        print(f"Final nbrs_tsr Shape: {nbrs_tsr.shape}")
        
        # 3. 순서 일치 확인
        if nbrs_tsr.shape[1] == len(occupied_indices_list):
            print("Verification: Number of neighbors matches mask. Checking order...")
            for i in range(nbrs_tsr.shape[1]):
                # 정렬된 마스크 좌표
                mask_pos = occupied_indices_list[i]
                
                # nbrs_tsr의 i번째 이웃 데이터 (마지막 시점의 좌표만 확인)
                nbr_last_pos = nbrs_tsr[-1, i, :].cpu().numpy() / 3.281 # 미터 단위로 변환
                
                print(f"  -> Neighbor #{i+1} in tensor (last pos: [{nbr_last_pos[0]:.2f}, {nbr_last_pos[1]:.2f}]) corresponds to Mask at (lat:{mask_pos[0]}, lon:{mask_pos[1]})")
            print("✅ Order seems correct.")
        else:
            print("❌ WARNING: Mismatch between number of neighbors in tensor and mask!")
        print("---" * 15)
        ### <<< 검증 코드 종료 >>>

        return hist_tsr, nbrs_tsr, mask_tsr.bool(), va_tsr, nbrsva_tsr, lane_tsr, nbrslane_tsr, cls_tsr, nbrscls_tsr

    def predict(self, model, target_id, target_state, history_data):
        """
        STDAN 모델(Encoder + Generator)을 사용하여 특정 객체의 미래 경로를 예측합니다.

        Args:
            model : tuple, (gdEncoder, generator) 
            target_id (str): 예측할 대상 객체의 ID.
            target_state (list): 대상 객체의 현재 상태 [x, y, yaw].
            history_data (dict): 모든 활성 객체의 과거 경로 데이터.
        Returns:
            tuple: (예측 궤적 텐서, 측면 예측 텐서, 종방향 예측 텐서)
        """
        # 1. Make input tensor
        gdEncoder = model[0]
        generator = model[1]
        tensors = self.make_tensor(
            target_id, target_state, history_data, no_nbsr=False
        )
        hist_stdan, nbrs_stdan, mask_stdan, va_stdan, nbrsva_stdan, lane_stdan, nbrslane_stdan, cls_stdan, nbrscls_stdan = tensors

        if self.use_maneuvers: lat_enc_stdan, lon_enc_stdan  = torch.zeros(1, 3, device=self.device), torch.zeros(1, 3, device=self.device)
        else: lat_enc_stdan, lon_enc_stdan = torch.empty(1, 1, device=self.device), torch.empty(1, 1, device=self.device)

        # 2. Predict trajectories
        values, spatial_weight = gdEncoder(hist_stdan, nbrs_stdan, mask_stdan, va_stdan, nbrsva_stdan, lane_stdan, nbrslane_stdan, cls_stdan, nbrscls_stdan)
        predictions, lat_predictions, lon_predictions = generator(values, lat_enc_stdan, lon_enc_stdan)
        # lon : {keep, decel, accel}, lat ; {keep, left, right}

        # spatial_weight : (1,n_head*history_len,query,grid_cell))
        # HJ 
        # 1. 텐서 모양 변경 및 데이터 정제 #TODO : parameterize

        spatial_weight = spatial_weight.cpu().detach()
        spatial_weight = spatial_weight.squeeze() 
        spatial_weight = spatial_weight.view(self.n_head,self.in_length,self.grid_size[0]*self.grid_size[1]) #(4,31,39)
        mean = spatial_weight.mean(dim=0) # [31, 39], 모든 head의 weigth 평균
        spatial_weight = mean.T # shape: [39, 31]
        
        best_traj, all_trajs, probabilities = self.postprocessing(predictions, lat_predictions, lon_predictions, target_state)

        return best_traj, all_trajs, probabilities, spatial_weight, mask_stdan

    def postprocessing(self, predictions, lat_probs, lon_probs, target_state):
        '''
        Post-processing e.g. feter-meter, coordinate transformation..
        '''
        # Multi-Modal
        if lat_probs is not None and lon_probs is not None: 
            # ----- trajectories -----  
            if not isinstance(predictions, np.ndarray): # (n,FH,2)
                predictions = np.array(predictions)
            if predictions.shape[2] == 1: # CS-LSTM (K, future, 1, 5), 모델별 squeeze axis가 다를 수 있으므로 확인 TODO : 어떤 shape의 input이던 하나의 shape으로 통일
                predictions = np.squeeze(predictions, axis=2)
            predictions = predictions * 0.3048 # feet to meter
            # ----- Probabilities -----  
            probabilities = np.outer(lon_probs, lat_probs).flatten() # outer product != cross product (outer : matrix, cross : vector)
            
            predictions = predictions[:,:,:2] # only means TODO : 분산값 쓸거면 바꿀 것
            
            if self.is_carla_test:
                idx = [2,5,8]
                predictions[idx,:,0] += 0.5 # RLC에만 0.5씩 더함
                predictions = predictions[:,:,[1,0]] # RHS to LHS
                all_trajs = self.transformation(predictions, target_state[0], target_state[1], target_state[2], inverse=True)
            else:
                all_trajs = self.transformation(predictions, target_state[0], target_state[1], target_state[2] - np.pi/2.0, inverse=True)
            

            # 가장 확률 높은 경로를 best_trajectory로 선택
            top_1_index = np.argmax(probabilities)
            best_traj = all_trajs[top_1_index]
            
            return best_traj, all_trajs, probabilities
            
        # Uni-Modal
        else:
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            predictions = np.squeeze(predictions, axis=1) * 0.3048
        
            predictions = predictions[:,:2] # only means TODO : 분산값 쓸거면 바꿀 것

            if self.is_carla_test:
                predictions = predictions[:,[1,0]] # RHS to LHS
                best_traj = self.transformation(predictions, target_state[0], target_state[1], target_state[2], inverse=True)
            else : 
                best_traj = self.transformation(predictions[:, :2], target_state[0], target_state[1], target_state[2] - np.pi/2.0, inverse=True)
            all_trajs = [best_traj] # 자료형 통일을 위해 리스트에 담음
            probabilities = None # 확률 정보 없음
            
        return best_traj, all_trajs, probabilities

class utils_trajectron(utils):
    def __init__(self, dt, is_carla_test):
        super().__init__()
        self.dt = dt
        
        self.standardization = {
        'VEHICLE': {
            'position': {
                'x': {'mean': 0, 'std': 80},
                'y': {'mean': 0, 'std': 80}
            },
            'velocity': {
                'x': {'mean': 0, 'std': 15},
                'y': {'mean': 0, 'std': 15},
                'norm': {'mean': 0, 'std': 15}
            },
            'acceleration': {
                'x': {'mean': 0, 'std': 4},
                'y': {'mean': 0, 'std': 4},
                'norm': {'mean': 0, 'std': 4}
            },
            'heading': {
                'x': {'mean': 0, 'std': 1},
                'y': {'mean': 0, 'std': 1},
                '°': {'mean': 0, 'std': np.pi},
                'd°': {'mean': 0, 'std': 1}
            }
          }
        }     

        self.env = Environment(node_type_list=['VEHICLE'], standardization=self.standardization)
        attention_radius = {(self.env.NodeType.VEHICLE, self.env.NodeType.VEHICLE): 30.0}
        self.env.attention_radius = attention_radius
        self.env.robot_type = self.env.NodeType.VEHICLE
        
        self.vehicle_columns = self._create_vehicle_multi_index()
        
        self.in_length = 16 #TODO : use config file
        self.out_length = 25
        
        self.is_carla_test = is_carla_test

    def load_model(self, model_dir, ts=100):
        '''
          model_dir : model's directory path
          ts : model_epoch (Maybe?)
        '''
        model_registrar = ModelRegistrar(model_dir, 'cpu')
        model_registrar.load_models(ts)
        with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
            hyperparams = json.load(config_json)

        trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

        return trajectron, hyperparams

    def predict(self, model, hyperparams, trackings):

        # Make scenes
        import time
        start = time.time()
        scene = self.tracking_to_scene(trackings)
        end = time.time()
        self.env.scenes = [scene]

        model.set_environment(self.env)
        model.set_annealing_params()

        # 여기서 원하는 예측 길이가 있다면 hyperparameter를 오버라이드합니다. 
        # if prediction_len is not None:
        #     hyperparams['prediction_horizon'] = prediction_len

        if 'override_attention_radius' in hyperparams:
            for attention_radius_override in hyperparams['override_attention_radius']:
                node_type1, node_type2, attention_radius_val = attention_radius_override.split(' ')
                self.env.attention_radius[(node_type1, node_type2)] = float(attention_radius_val)
        scenes = self.env.scenes

        # get_timesteps_data에 전달되는 max_ft도 prediction_len으로 설정합니다.
        import time
        start = time.time()
        ph = hyperparams['prediction_horizon']
        
        with torch.no_grad():
            for scene in scenes:
                t = scene.timesteps - 1  # 최신 시점에서 예측 수행
                timesteps = np.array([t])
                # min_future_timesteps은 그대로 0, max_ft를 ph로 전달합니다.
                # scene은 잘 만들어졌다고 가정
                # timesteps 뭐가되어야할진 미지수 

                """
                Predicts the future of a batch of nodes.

                :param inputs: Input tensor including the state for each agent over time [bs, t, state].
                :param inputs_st: Standardized input tensor.
                :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
                :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                                    [[bs, t, neighbor state]]
                :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
                :param robot: Standardized robot state over time. [bs, t, robot_state]
                :param map: Tensor of Map information. [bs, channels, x, y]
                :param prediction_horizon: Number of prediction timesteps.
                :param num_samples: Number of samples from the latent space.
                 
                :param z_mode: If True: Select the most likely latent state. (하나의 mode만 선택)
                :param gmm_mode: If True: The mode of the GMM is sampled. (결정된 mode의 분포에서 sampling할 때, 평균값을 쓸지)
                :param full_dist: Samples all latent states and merges them into a GMM as output.
                :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
                :return:
                """
                
                # Most likely :  z_mode + gmm_mode  
                predictions = model.predict(
                        scene,
                        timesteps,
                        ph,
                        num_samples=6, 
                        min_history_timesteps=hyperparams['minimum_history_length'],
                        min_future_timesteps=0, # 왜 25 넣으면 이상하지?
                        z_mode=False, # false 시 num_samples만큼의 mode를 선택  
                        gmm_mode=True, # True 시 결정된 mode의 평균값 사용
                        full_dist=False, # 논문의 full mode
                        all_z_sep=True # 25개의 mode에 대해 관찰해보고 싶을 때 
                    )
                
                if not predictions:
                    continue
            
                for ts, preds in predictions.items(): # TODO : What is ts?
                    converted_predictions = {node.id: trajectory.squeeze() for node, trajectory in preds.items()}
                    # for node, pred in preds.items():
                    #     print(f'object_id : {node.id}, preds shape : {pred.shape}')
                    
                # pos_x_mean = scene.pos_x_mean
                # pos_y_mean = scene.pos_y_mean
                # for primary_idx, vehicles in predictions.items():
                #     for key, data in vehicles.items():
                #         data[..., 0] = data[..., 0] + pos_x_mean
                #         data[..., 1] = data[..., 1] + pos_y_mean # ?
        end = time.time()
        return converted_predictions

    def postprocessing(self, predictions, lat_probs, lon_probs, obj_state): #TODO : trajectron++용 함수 정리하기
        """
        다양한 모델의 출력을 받아 표준화된 형식으로 후처리합니다.
        :param predictions: 모델의 원본 궤적 예측 (ndarray or dict)
        :param lat_probs: 측면 기동 확률 (ndarray or None)
        :param lon_probs: 종단 기동 확률 (ndarray or None)
        :param obj_state: 현재 객체 상태 [x, y, yaw]
        :return: (best_trajectory, all_trajectories, probabilities) 튜플
        """

        # --- CASE 1: Trajectron++ 예측 결과 처리 (입력이 dict) ---, TODO : trajectron output도 형식 통일
        if isinstance(predictions, dict):
            # 예측 결과가 비어있는 경우 처리
            if not predictions or not list(predictions.values()):
                return None, [], None

            # 시각화를 위해 모든 예측 경로를 담을 리스트
            all_trajs = []
            
            # predictions 딕셔너리의 첫 번째 값(하나의 에이전트에 대한 예측 결과)을 가져옴
            # 이 값은 (num_samples, prediction_horizon, state) 형태의 NumPy 배열일 수 있음
            prediction_results = np.asarray(list(predictions.values())[0])

            # --- 핵심 수정 부분 ---
            # 예측 결과의 차원을 확인하여 다중 경로인지 단일 경로인지 판단
            
            # 1. 다중 경로인 경우 (e.g., shape: [5, 12, 2] -> 5개의 샘플)
            if prediction_results.ndim == 3 and prediction_results.shape[0] > 1:
                # 각 샘플(경로)을 all_trajs 리스트에 추가
                for i in range(prediction_results.shape[0]):
                    all_trajs.append(prediction_results[i, :, :])
                    
            # 2. 단일 경로인 경우 (기존 코드와 호환)
            else:
                all_trajs.append(prediction_results)

            # 반환할 'best_traj'는 첫 번째 경로를 대표로 사용 (기존 로직과 호환성을 위해)
            # 만약 all_trajs가 비어있다면 None을 반환
            best_traj = all_trajs[0] if all_trajs else None
            
            # Trajectron++는 GMM 전체로 분포를 표현하므로 개별 경로의 확률은 반환하지 않음
            probabilities = None
            
            return best_traj, all_trajs, probabilities

    def tracking_to_scene(self, trackings: dict) -> Scene:
        """추적 데이터를 Trajectron의 Scene 객체로 변환하는 최적화된 함수."""
        if not trackings:
            return Scene(timesteps=0, dt=self.dt, name="empty_trackings")

        # 유효성 검사 및 NumPy 배열 변환
        valid_histories = {str(k): np.array(v) for k, v in trackings.items() 
                           if isinstance(v, (list, np.ndarray)) and np.array(v).ndim == 2 and np.array(v).size > 0}

        if not valid_histories:
            return Scene(timesteps=0, dt=self.dt, name="empty_valid_trackings")

        max_timesteps = max(h.shape[0] for h in valid_histories.values()) # 17 + current_step
        # max_timesteps = 40 : 이건 나중에 학습할 땐 필요할
        scene = Scene(timesteps=max_timesteps, dt=self.dt, name="trackings")

        # 단일 루프로 패딩 및 노드 생성, 없던 agent가 생길 경우 length를 맞춰주기 위해 필요한 코드이지만 현재 나에겐 필요 X
        for obj_id, history in valid_histories.items():
            if history.shape[0] < max_timesteps:
                pad_length = max_timesteps - history.shape[0]
                history = np.vstack([history, np.tile(history[-1], (pad_length, 1))])
            
            # 최적화된 헬퍼 함수를 호출하여 DataFrame 생성
            node_data, node_frequency_multiplier = self._create_vehicle_dataframe(history, max_timesteps)
            # print(f'node_data : {node_data}')
            node = Node(node_type = self.env.NodeType.VEHICLE, node_id=obj_id, data=node_data, first_timestep=0, frequency_multiplier=node_frequency_multiplier)
            scene.nodes.append(node)
            
        return scene

    def _trajectory_curvature(self, t):
        path_distance = np.linalg.norm(t[-1] - t[0])

        lengths = np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1))  # Length between points
        path_length = np.sum(lengths)
        if np.isclose(path_distance, 0.):
            return 0, 0, 0
        return (path_length / path_distance) - 1, path_length, path_distance

    def _create_vehicle_multi_index(self):
        """차량 데이터프레임을 위한 MultiIndex를 생성합니다."""
        columns = pd.MultiIndex.from_product(
            [['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']]
        )
        columns = columns.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
        columns = columns.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))
        return columns

    def _create_vehicle_dataframe(self, history: np.ndarray, max_timesteps: int) -> pd.DataFrame:
        """한 객체의 이력 데이터로부터 구조화된 DataFrame을 생성하는 헬퍼 함수."""
        node_frequency_multiplier = 1
        curv_0_2 = 0
        curv_0_1 = 0
        total = 0
        
        x, y, heading = history[:, 0], history[:, 1], history[:, 2]
        vx = derivative_of(x, self.dt)
        vy = derivative_of(y, self.dt)
        ax = derivative_of(vx, self.dt)
        ay = derivative_of(vy, self.dt)

        curvature, pl, _ = self._trajectory_curvature(np.stack((x, y), axis=-1))
        if pl < 1.0:  # vehicle is "not" moving
            x = x[0].repeat(self.max_timesteps)
            y = y[0].repeat(self.max_timesteps)
            heading = heading[0].repeat(self.max_timesteps)
        total += 1
        if pl > 1.0:
            if curvature > .2:
                curv_0_2 += 1
                node_frequency_multiplier = 3*int(np.floor(total/curv_0_2))
            elif curvature > .1:
                curv_0_1 += 1
                node_frequency_multiplier = 3*int(np.floor(total/curv_0_1))

        velocity_vectors = np.stack((vx, vy), axis=-1)
        velocity_norm = np.linalg.norm(velocity_vectors, axis=-1)
        accel_norm = np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1)
        
        heading_v = np.divide(velocity_vectors, velocity_norm[..., np.newaxis], 
                              out=np.zeros_like(velocity_vectors), where=(velocity_norm[..., np.newaxis] > 1e-6))
        
        data_dict = {
            ('position', 'x'): x, ('position', 'y'): y,
            ('velocity', 'x'): vx, ('velocity', 'y'): vy, ('velocity', 'norm'): velocity_norm,
            ('acceleration', 'x'): ax, ('acceleration', 'y'): ay, ('acceleration', 'norm'): accel_norm,
            ('heading', 'x'): heading_v[:, 0], ('heading', 'y'): heading_v[:, 1],
            ('heading', '°'): heading, ('heading', 'd°'): derivative_of(heading, self.dt, radian=True)
        }
        
        # 모든 값을 1차원으로 만들어 DataFrame 생성 오류 방지
        for key, value in data_dict.items():
            if value.ndim > 1: data_dict[key] = value.squeeze()
            
        return pd.DataFrame(data_dict, columns=self.vehicle_columns), node_frequency_multiplier


class Detector(): 
    '''
    Get ego nbrs datas from carla  
    '''
    # TODO : ego 객체만 저장하지말고, 관련 정보를 미리 data로 정리
    # TODO : ego 앞에 있는 객체만 detector에 저장
    def __init__(self):
        self.ego = None # Carla.vehicle
        self.nbrs  = {} # dict{id: Carla.vehicle}
    
    def update(self, world):
        '''
        Get ego and only nbrs that
          1. on the left/right lane of ego 
          2. in front of ego
        '''
        # TODO : 실제 센서 FOV와 같게
        
        self.ego = world.player
        ego_transform = self.ego.get_transform()
        ego_location = ego_transform.location
        ego_waypoint = world.map.get_waypoint(ego_location, project_to_road=True)
        ego_lane_id = ego_waypoint.lane_id
        
        ego_forward_vec = ego_transform.get_forward_vector()
        ego_forward_vec = np.array([ego_forward_vec.x, ego_forward_vec.y, ego_forward_vec.z]) # to np.array()
        
        # FOV filtering
        nbrs = world.nbrs    
        for nbr in nbrs:
            if nbr.id in self.nbrs.keys(): # 기존에 있던 객체면 pass  
                continue

            nbr_location = nbr.get_location()
            nbr_waypoint = world.map.get_waypoint(nbr_location, project_to_road=True)
            nbr_lane_id = nbr_waypoint.lane_id

            ego_to_nbr_vec = nbr_location - ego_location
            ego_to_nbr_vec = np.array([ego_to_nbr_vec.x, ego_to_nbr_vec.y, ego_to_nbr_vec.z]) # to np.array
                
            if nbr_waypoint and (nbr_lane_id == ego_lane_id - 1 
                                or nbr_lane_id == ego_lane_id + 1 
                                or nbr_lane_id == ego_lane_id) \
                            and (ego_to_nbr_vec.dot(ego_forward_vec)>=0):
                self.nbrs[nbr.id] = nbr
        
        #TODO : 기존에 있던 객체가 없어지면 어떻게 할지도 추가해야하나? tracker에 이미 있어서 괜찮을 듯

    def is_on_ego_lane(self, world, veh_id):
        '''
        Check is vehicle's lane same with the ego's lane
        '''
        map = world.map
        ego_location = self.ego.get_location()
        ego_waypoint = map.get_waypoint(ego_location, project_to_road=True)
        ego_lane_id = ego_waypoint.lane_id
        
        veh = world.world.get_actor(veh_id)
        location = veh.get_location()
        waypoint = map.get_waypoint(location, project_to_road=True)
        
        if waypoint and waypoint.lane_id == ego_lane_id:
            return True
        else:
            return False


class Tracker(utils): 
  def __init__(self,maxlen=17):

    self._maxlen = maxlen 
    self._trackings : Dict[ObjectID, deque] = {} #map frame, 현재 frame부터 16개
    
    # self._objects: Dict[ObjectID, List[np.array]] = {} # map frame 
    self._GT : Dict[ObjectID, List[np.array]] = {}

  def update(self, detector): 
      '''
        Stack ego and nbrs's datas
      '''    
      id = detector.ego.id 
      ego_transform = detector.ego.get_transform() 
      x, y         = ego_transform.location.x, ego_transform.location.y
      yaw       = self.AngleToRad(ego_transform.rotation.yaw)     
      ego_state = np.array([id,x,y,yaw])
  
      if not ego_state[0] in self._trackings.keys(): # initialize
        ego_dq = deque(maxlen=self._maxlen) 
        ego_dq.append(ego_state[1:])
  
        self._trackings[ego_state[0]] = ego_dq
        self._GT[ego_state[0]] = [ego_state[1:3]]
      else:
        self._trackings[ego_state[0]].append(ego_state[1:])
        self._GT[ego_state[0]].append(ego_state[1:3])
      
      # n -> 0 되는 상황 커버하는 방어코드, 현재 들어오는 objects가 0이 되버리면 장애물 정보 전부 삭제
      if not detector.nbrs.keys():
        ids = list(self._trackings.keys())
        for id in ids:
          if id == ego_state[0]: # ego는 제외
            continue
          del self._trackings[id]
        print("No nbrs. Tracking inform all deleted")  
  
      # Get nbrs
      nbrs_ids = []
      for id, nbr in detector.nbrs.items(): 
        nbrs_ids.append(id)
        
        nbr_transform = nbr.get_transform()
        x = nbr_transform.location.x
        y = nbr_transform.location.y
        yaw = self.AngleToRad(nbr_transform.rotation.yaw)
        nbr_state = np.array([x,y,yaw]) 
        
        if not id in self._trackings.keys(): # New car
          nbr_dq = deque(maxlen=self._maxlen)
          nbr_dq.append(nbr_state)
          self._trackings[id] = nbr_dq
          self._GT[id] = [nbr_state[:2]]
        else: # 기존 차량 
          self._trackings[id].append(nbr_state) 
          self._GT[id].append(nbr_state[:2])
  
      # nbrs 자체가 아예 없어버리면 for 자체가 안도니 문제 발생 즉 1->0 되는 상황 커버 불가
      lost_nbrs_ids = [id for id in self._trackings.keys() if id not in nbrs_ids and id != ego_state[0]] 
      for id in lost_nbrs_ids:
        del self._trackings[id]
        print(f'nbr {id} deleted')
      
  # getter
  def get_trackings(self, id=None):
    if id:
        return self._trackings[id]
    else:
        return self._trackings

  def get_GTs(self):
    return self._GT

  def save_GTs(self,txt_name): # 시나리오가 끝나면 주행했던 경로를 csv로 저장한다.
    #TODO : if type of GT_obs_trajectories != Dict -> error
    path = os.path.join(os.getcwd(),'GT_trajectories')        
    with open(os.path.join(path,txt_name), 'w') as f:
      for idx, states in enumerate(self._GT.values()):
        poses = np.array(states) # all of x,y
        np.savetxt(f, poses,delimiter=',')#TODO : Have to change if there are many obstacles
        f.write("\n")

class Evaluation:
    ### <<< 변경됨: __init__ 함수 추가
    def __init__(self, dt):
        """
        :param dt: 시뮬레이션의 시간 간격 (예: 0.1, 0.2)
        """
        self.dt = dt
        self.steps_per_second = int(1 / self.dt)

    ### <<< 변경됨: RMSE 계산 로직 수정
    def RMSE(self, best_pred_traj, ground_truth_future, all_rmse_dict):
        """
        최고의 예측 경로와 실제 경로 사이의 RMSE를 계산하고 저장합니다.
        """
        sq_errors = np.sum((best_pred_traj - ground_truth_future)**2, axis=1)

        total_steps = ground_truth_future.shape[0]
        # DT를 기반으로 총 예측 시간(초) 계산
        num_seconds = total_steps // self.steps_per_second
        
        calculated_rmses = {}
        # 1초부터 총 예측 시간(초)까지 반복
        for t in range(1, num_seconds + 1):
            key = f'{t}s'
            # start = (t - 1) * self.steps_per_second
            # end = t * self.steps_per_second

            # start 인덱스를 항상 0으로 설정하여 누적 계산
            start = 0 
            # end 인덱스는 t초에 해당하는 스텝 (예: t=1 -> 10, t=2 -> 20)
            end = t * self.steps_per_second
            
            if len(sq_errors[start:end]) > 0 and key in all_rmse_dict:
                rmse_val = np.sqrt(np.mean(sq_errors[start:end]))
                all_rmse_dict[key].append(rmse_val)
                calculated_rmses[key] = rmse_val
        
        return calculated_rmses
    
class visualization():
    def __init__(self):
        pass
    
    def _create_prediction_plots(self, ax, num_predictions, color, style, z_order, 
                                legend_start_x = 0.88, legend_start_y=0.98, num_legend_items=2):
        """
        예측 경로 선, 경로 끝 인덱스 텍스트, 범례 영역 확률 텍스트 객체
        """
        lines = []
        index_texts = []
        legend_prob_texts = []
        
        for k in range(num_predictions):
            # 경로 선 생성
            line, = ax.plot([], [], style, color=color, markersize=3 if style[0] != '+' else 4, alpha=0.0, visible=False, zorder=z_order)
            lines.append(line)
            
            # 경로 끝 인덱스 텍스트 생성
            index_text = ax.text(0, 0, '', fontsize=7, color=color, fontweight='bold', ha='left', va='center', zorder=z_order+1, alpha=0.0, visible=False)
            index_texts.append(index_text)
            
        # 범례 영역 확률 텍스트 생성 (Top-K 개수만큼)
        for j in range(num_legend_items):
            prob_text = ax.text(legend_start_x, legend_start_y - 0.1 * j, "", transform=ax.transAxes, color=color, fontsize=12, ha='right', va='top', visible=False)
            legend_prob_texts.append(prob_text)
            
        return lines, index_texts, legend_prob_texts

    ### <<< 변경됨: 범례 생성 함수도 동적으로 수정
    def _create_legend(self, ax, viz_linear, viz_cs, viz_stdan, viz_tj):
        from matplotlib.lines import Line2D
        
        legend_elements = [ 
            Line2D([0], [0], marker='o', color='black', label='History', linestyle='-'), 
            Line2D([0], [0], marker='o', color='gray', label='Ground Truth Future', linestyle='-', alpha=0.7), 
            Rectangle((0,0), 1, 1, color='purple', label='Current Vehicle (Agent Color)')
        ]
        if viz_linear:
            legend_elements.insert(2, Line2D([0], [0], marker='v', color='cyan', label='Linear Pred', linestyle=':'))
        if viz_cs:
            legend_elements.insert(3, Line2D([0], [0], marker='o', color='red', label='CS-LSTM Pred', linestyle=':'))
        if viz_stdan:
            legend_elements.insert(4, Line2D([0], [0], marker='s', color='magenta', label='STDAN Pred', linestyle=':'))
        if viz_tj:
            legend_elements.insert(5, Line2D([0], [0], marker='x', color='blue', label='Trajectron Pred', linestyle=':'))

        ax.legend(handles=legend_elements, loc='upper left', fontsize='small')


    ### <<< 변경됨: 함수 시그니처에 플래그 추가
    def setup_plot_elements(self, ax: plt.Axes, object_ids: List[str], 
                            VISUALIZE_LINEAR: bool,
                            VISUALIZE_CSLSTM: bool, NUM_PREDICTIONS_CS: int,
                            VISUALIZE_STDAN: bool, NUM_PREDICTIONS_STDAN: int,
                            VISUALIZE_TRAJECTRON: bool, NUM_PREDICTIONS_TJ: int) -> Tuple[Dict, plt.Text]:
        """
        Matplotlib 플롯의 모든 시각적 요소를 초기화하고 구조화된 딕셔너리로 반환합니다.
        """
        VEHICLE_WIDTH = 2.0
        VEHICLE_LENGTH = 4.5
        
        colors = plt.cm.get_cmap('jet', len(object_ids))
        plot_elements = {}

        for i, object_id in enumerate(object_ids):
            color = colors(i)
            
            agent_elements = {}

            # --- CS-LSTM ---
            if VISUALIZE_CSLSTM:
                agent_elements['cslstm_preds'], agent_elements['cslstm_index_texts'], agent_elements['cslstm_legend_probs'] = \
                    self._create_prediction_plots(ax, NUM_PREDICTIONS_CS, 'red', 'o:', z_order=7, legend_start_x = 0.88 , legend_start_y=0.98, num_legend_items=3)
            
            # --- STDAN --- 
            if VISUALIZE_STDAN:
                agent_elements['stdan_preds'], agent_elements['stdan_index_texts'], agent_elements['stdan_legend_probs'] = \
                    self._create_prediction_plots(ax, NUM_PREDICTIONS_STDAN, 'magenta', 's:', z_order=8, legend_start_x = 0.98,  legend_start_y=0.98, num_legend_items=4)

            # --- Trajectron 객체 생성 ---
            if VISUALIZE_TRAJECTRON:
                 # Trajectron은 확률 텍스트 없음 (num_legend_items=0)
                 agent_elements['trajectron_preds'], agent_elements['trajectron_index_texts'], _ = \
                     self._create_prediction_plots(ax, NUM_PREDICTIONS_TJ, 'blue', 'x:', z_order=9, num_legend_items=0)
                 agent_elements['trajectron_legend_probs'] = [] # 빈 리스트 명시

            # --- Linear 객체 생성 ---
            if VISUALIZE_LINEAR:
                 # Linear는 확률 텍스트 없음 (num_legend_items=0)
                 agent_elements['linear_pred'], agent_elements['linear_index_texts'], _ = \
                     self._create_prediction_plots(ax, 1, 'green', '+-', z_order=6, num_legend_items=0)
                 agent_elements['linear_legend_probs'] = [] # 빈 리스트 명시

            # 차량 사각형 및 기본 경로 설정
            vehicle_rect = Rectangle((0, 0), VEHICLE_LENGTH, VEHICLE_WIDTH, edgecolor=color, facecolor=color, alpha=0.7, zorder=5)
            ax.add_patch(vehicle_rect)
            
            # 모든 기본 요소를 딕셔너리에 추가
            agent_elements.update({
                'history': ax.plot([], [], 'o-', color='black', markersize=2)[0],
                'future': ax.plot([], [], 'o-', color='blue', alpha=0.7, markersize=6)[0],
                'vehicle_rect': vehicle_rect,
                'current_dot': ax.plot([], [], 'o', color='white', markersize=4, zorder=6, markeredgecolor='black', markeredgewidth=0.5)[0],
                'speed_text': ax.text(0, 0, '', fontsize=8, color=color, fontweight='bold', ha='center', va='top', zorder=7, linespacing=1.5)
            })
            
            plot_elements[object_id] = agent_elements

        # 범례 생성 시에도 플래그 전달
        self._create_legend(ax, VISUALIZE_LINEAR, VISUALIZE_CSLSTM, VISUALIZE_STDAN, VISUALIZE_TRAJECTRON)
        time_text = ax.text(0.05, 0.05, '', transform=ax.transAxes, fontsize=12, verticalalignment='bottom')
        
        return plot_elements, time_text

    def update_prediction_plots(self, all_pred_trajs, probabilities, 
                                plot_lines, index_texts, legend_prob_texts,
                                num_predictions_to_show=2):
        """
        계산된 예측 경로와 확률을 사용하여 Matplotlib 객체를 업데이트합니다.
        :param all_pred_trajs: 시각화할 모든 예측 경로 (shape: (K, Timesteps, 2) or list)
        :param probabilities: 각 경로에 대한 확률 (shape: (K,)) (Uni-modal의 경우 None)
        :param plot_lines: 업데이트할 Line 객체 리스트
        :param plot_texts: 업데이트할 Text 객체 리스트
        :param num_predictions_to_show: 상위 몇 개의 예측을 보여줄지 결정
        """
        # Multi-modal case
        if probabilities is not None:
            desc_indices = np.argsort(-probabilities)
            top_indices = desc_indices[:num_predictions_to_show]
            top_probs = probabilities[top_indices]
            
            fixed_alphas = np.linspace(1.0, 0.4, num_predictions_to_show) # Top-1=1.0, Top-2=0.4 등
            for k in range(len(plot_lines)):
                line = plot_lines[k]
                index_text = index_texts[k] # <<< 경로 끝 인덱스 텍스트

                if k in top_indices:
                    rank = np.where(top_indices == k)[0][0]
                    alpha = fixed_alphas[rank]
                    traj = all_pred_trajs[k]
                    line.set_data(traj[:, 0], traj[:, 1]); line.set_alpha(alpha); line.set_visible(True)
                    
                    # <<< 변경: 경로 끝에는 인덱스만 표시 >>>
                    index_text.set_position(traj[-1]); index_text.set_text(f"#{k}"); index_text.set_alpha(1.0); index_text.set_visible(True)
                else:
                    line.set_visible(False); index_text.set_visible(False)

            # --- 범례 텍스트 업데이트 (확률만) ---
            for j in range(len(legend_prob_texts)):
                text_obj = legend_prob_texts[j]
                if j < len(top_indices): # Top-K 범위 내
                    idx = top_indices[j]
                    prob = top_probs[j]
                    # <<< 변경: 범례에는 인덱스와 확률 모두 표시 >>>
                    text_obj.set_text(f"#{idx}: {(prob*100):.1f}%")
                    text_obj.set_visible(True)
                else:
                    text_obj.set_visible(False)
                    
        else: # uni-modal
            best_traj = all_pred_trajs[0]
            for k in range(len(plot_lines)):
                line = plot_lines[k]
                index_text = index_texts[k] # <<< 경로 끝 인덱스 텍스트
                if k == 0:
                    line.set_data(best_traj[:, 0], best_traj[:, 1]); line.set_alpha(1.0); line.set_visible(True)
                    # <<< 변경: 경로 끝 인덱스 표시 >>>
                    index_text.set_position(best_traj[-1]); index_text.set_text(f"#0"); index_text.set_alpha(1.0); index_text.set_visible(True)
                else:
                    line.set_visible(False); index_text.set_visible(False)
            # 확률 텍스트는 모두 숨김
            for text_obj in legend_prob_texts:
                text_obj.set_visible(False)
                
