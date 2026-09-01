from __future__ import print_function, division
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
import h5py
from config import args
import time

from tqdm import tqdm

# hdf5
# 1. matlab은 column 우선, hdf5는 row 우선. 기존 loadmat 은 이를 알아서 맞춰서 output하기 때문에 문제 발생 X
# 2. 주소값이 저장되어있기 때문에 값 조회를 다시 해주어야함

class NgsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
            
        try:
            # 1. 기존 방식 (v7.0 이하) 시도
            mat_data = scp.loadmat(mat_file)
            self.D = mat_data['traj']
            self.T = mat_data['tracks']
            print("Loaded using scipy.io (v7.0)")
            
        except NotImplementedError:
            # 2. 실패 시 HDF5 방식 (v7.3) 시도
            print("v7.3 mat file detected. Loading using h5py...")
            with h5py.File(mat_file, 'r') as f:
                # (1) traj 로드
                # HDF5는 (Col, Row)로 저장되므로 .T(전치)를 해야 기존과 동일해짐
                self.D = np.array(f['traj']).T
                
                # (2) tracks 로드 (Cell Array -> Object Reference)
                # tracks 참조 배열도 전치 필요 (N_ds, N_veh)
                tracks_refs = np.array(f['tracks']).T
                
                # self.T를 기존 구조(Numpy Object Array)와 똑같이 만듦
                self.T = np.empty(tracks_refs.shape, dtype=object)
                
                # 참조를 따라가서 실제 데이터를 꺼내옴
                # (시간이 좀 걸릴 수 있으나, 메모리에 다 올려야 기존 코드와 호환됨)
                total_tracks = tracks_refs.size
                with tqdm(total=total_tracks, desc="Loading Tracks from HDF5") as pbar:     
                    for r in range(tracks_refs.shape[0]): # Dataset ID Loop
                        for c in range(tracks_refs.shape[1]): # Vehicle ID Loop
                            ref = tracks_refs[r, c]
                            try:
                                # 참조된 데이터 읽기
                                # HDF5로 읽으면 (Time, Features) 형태가 됨
                                # 하지만 기존 코드(getHistory)에서 .transpose()를 하고 있으므로,
                                # 여기서 미리 .T를 해서 (Features, Time) 형태로 뒤집어 놔야 함!
                                track_data = np.array(f[ref]).T 
                                self.T[r, c] = track_data
                            except:
                                # 빈 참조일 경우 빈 배열 할당
                                raise Exception("invalid ref")
                            pbar.update(1)
                                
        print(f'self.D : {self.D.shape}')
        print(f'self.T : {self.T.shape}')
        
        self.t_h = t_h  # 
        self.t_f = t_f  # 
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid
        self.alltime = 0
        self.count = 0

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        ttt = time.time()
        dsId = self.D[idx, 0].astype(int)  # dataset id
        vehId = self.D[idx, 1].astype(int)  # agent id
        t = self.D[idx, 2]  # frame
        grid = self.D[idx, 11:]  #  grid id
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)  
        refdistance = np.zeros_like(hist[:, 0]) # infer 시 필요 X
        refdistance = refdistance.reshape(len(refdistance), 1) # infer 시 필요 X
        fut_vel, fut_pos = self.getFuture(vehId, t, dsId)  
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId) 
        cclass = self.getClass(vehId, t, vehId, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis)  
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([args['lon_length']])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([args['lat_length']])
        lat_enc[int(self.D[idx, 9] - 1)] = 1

        # hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
        
        # in evaluate.py : hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data -> collate_fn의 return 값과 동일 
        # hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls : inference를 위해 꼭 필요한 정보
        # f_length = 5로 학습시킬 경우 lane정보는 필요하지 않다. -> 그냥 다 1로 넣어주자 
        # cls : 1 - motorcycle, 2 - auto, 3 - truck -> 모두 다 2로 설정해주면 됨
                
        # return -> collate_fn -> evaluate.py
        # dis, nbrdis는 사용 X
        # print(f'get_item : {time.time() - ttt}')

        return hist, fut_vel, fut_pos, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]  

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose() 
            vehTrack = self.T[dsId - 1][vehId - 1].transpose() 
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]  

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5] 

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose() 
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]  

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h) 
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos  
            if len(hist) < self.t_h // self.d_s + 1:  
                return np.empty([0, 2])
            return hist



    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()  

        # 현재 시점(t)의 인덱스와 위치(refPos) 추출 [cite: 348, 352]
        curr_idx = np.argwhere(vehTrack[:, 0] == t).item()
        refPos = vehTrack[curr_idx, 1:3]  
        
        # 미래 구간 추출 (차분을 위해 현재 시점 t부터 t_f 이후까지)
        enpt = np.minimum(len(vehTrack), curr_idx + self.t_f + 1)
        future_segment = vehTrack[curr_idx : enpt, 1:3] # [Steps+1, 2]
        
        # 1. 속도 GT 계산 (단위: feet/second) [cite: 318, 375]
        fut_vel = (future_segment[1:] - future_segment[:-1]) / 0.1
        fut_vel = fut_vel[::self.d_s, :]
        
        # 2. 위치 GT 계산 (현재 시점 refPos 대비 상대 좌표) [cite: 348, 352]
        fut_pos = (future_segment[1:] - refPos)
        fut_pos = fut_pos[::self.d_s, :]
        
        return fut_vel, fut_pos

    ## Collate function for dataloader
    def collate_fn(self, samples):
        ttt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, _, nbrs, _, _,  _, _, _, _, _, _, _, _ in samples:
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)  
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_vel_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        fut_pos_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), args['lat_length'])  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), args['lon_length'])  # (batch,3)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (hist, fut_vel, fut_pos, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                       neighborsdistance, cclass, neighborsclass) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
            fut_vel_batch[0:len(fut_vel), sampleId, 0] = torch.from_numpy(fut_vel[:, 0])
            fut_vel_batch[0:len(fut_vel), sampleId, 1] = torch.from_numpy(fut_vel[:, 1])
            fut_pos_batch[0:len(fut_pos), sampleId, 0] = torch.from_numpy(fut_pos[:, 0])
            fut_pos_batch[0:len(fut_pos), sampleId, 1] = torch.from_numpy(fut_pos[:, 1])
            op_mask_batch[0:len(fut_pos), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
                    count4 += 1
        #  mask_batch 
        self.alltime += (time.time() - ttt)
        self.count += args['num_worker']
        #if (self.count > args['time']):
        #    print(self.alltime / self.count, "data load time")

        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_vel_batch, fut_pos_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position


class HighdDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        self.D = np.transpose(h5py.File(mat_file, 'r')['traj'].value)
        self.T = h5py.File(mat_file, 'r')
        ref = self.T['tracks'][0][0]
        res = self.T[ref]
        self.t_h = t_h  # 
        self.t_f = t_f  # 
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of encoder LSTM
        self.grid_size = grid_size  # size of social context grid

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[0, idx].astype(int)  
        vehId = self.D[1, idx].astype(int)  
        t = self.D[2, idx] 
        grid = self.D[14:, idx]  
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)  
        refdistance = np.zeros_like(hist[:, 0])
        refdistance = refdistance.reshape(len(refdistance), 1)
        fut = self.getFuture(vehId, t, dsId)  
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis) 
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        lon_enc = np.zeros([2])
        lon_enc[int(self.D[idx, 13] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 12] - 1)] = 1

        # hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

        return hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance, neighborsdistance, cclass, neighborsclass

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 8] 

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 8]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]  

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6:8] 

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6:8] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]  

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)  
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1 
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos 
            if len(hist) < self.t_h // self.d_s + 1:  
                return np.empty([0, 2])
            return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()  
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3] 

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()  
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3] 
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):
        nowt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _ in samples:
            temp = sum([len(nbrs[i]) != 0 for i in range(len(nbrs))])
            nbr_batch_size += temp
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(maxlen, nbr_batch_size, 2)  # (len,batch*车数，2)
        nbrsva_batch = torch.zeros(maxlen, nbr_batch_size, 2)
        nbrslane_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsclass_batch = torch.zeros(maxlen, nbr_batch_size, 1)
        nbrsdis_batch = torch.zeros(maxlen, nbr_batch_size, 1)

        # Initialize social mask batch:
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        map_position = torch.zeros(0, 2)
        mask_batch = mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)  # (len1,batch,2)
        distance_batch = torch.zeros(maxlen, len(samples), 1)
        fut_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(self.t_f // self.d_s, len(samples), 2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), 2)  # (batch,2)
        va_batch = torch.zeros(maxlen, len(samples), 2)
        lane_batch = torch.zeros(maxlen, len(samples), 1)
        class_batch = torch.zeros(maxlen, len(samples), 1)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                       neighborsdistance, cclass, neighborsclass) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            distance_batch[0:len(hist), sampleId, :] = torch.from_numpy(refdistance)
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[0:len(fut), sampleId, :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[0:len(va), sampleId, 0] = torch.from_numpy(va[:, 0])
            va_batch[0:len(va), sampleId, 1] = torch.from_numpy(va[:, 1])
            lane_batch[0:len(lane), sampleId, 0] = torch.from_numpy(lane)
            class_batch[0:len(cclass), sampleId, 0] = torch.from_numpy(cclass)
            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[0:len(nbr), count, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[0:len(nbr), count, 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[0:len(nbrva), count1, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[0:len(nbrva), count1, 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1
            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[0:len(nbrlane), count2, :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[0:len(nbrdis), count3, :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[0:len(nbrclass), count4, :] = torch.from_numpy(nbrclass)
                    count4 += 1
        return hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, fut_batch, op_mask_batch, va_batch, nbrsva_batch, lane_batch, nbrslane_batch, distance_batch, nbrsdis_batch, class_batch, nbrsclass_batch, map_position
