from __future__ import print_function

import loader2_hdf5_vel as lo
import loader2_highD_1107_final as lo_highD

from torch.utils.data import DataLoader
import pandas as pd
from config_vel import *
import matplotlib
matplotlib.use('Agg') # GUI 창을 띄우지 않는 백엔드로 고정
import matplotlib.pyplot as plt
from pathlib import Path 
import os
import time
import wandb

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

writer = pd.ExcelWriter('A.xlsx')

    
def plot_trajectory_to_wandb(hist, fut_pos_gt, best_fut_vel, epoch, idx, dt=0.1):
    """
    최적 경로(Best Trajectory)를 wandb에 시각화합니다.
    hist: 과거 위치 [Steps, 2]
    fut_pos_gt: 실제 위치 GT [Steps, 2]
    best_fut_vel: 가장 확률이 높은 예측 속도 [Steps, 2]
    """
    plt.figure(figsize=(10, 4))
    
    # 1. 과거 경로 및 GT (상대 좌표)
    plt.plot(hist[:, 1], hist[:, 0], 'r:', label='History')
    plt.plot(fut_pos_gt[:, 1], fut_pos_gt[:, 0], 'k-', label='Ground Truth')
    
    # 2. 최적 예측 경로 복원 (속도 -> 위치 적분)
    # 현재 위치 [0,0]에서 시작하여 속도 * dt를 누적 합산
    best_fut_pos = t.cumsum(best_fut_vel * dt, dim=0).detach().cpu().numpy()
    plt.plot(best_fut_pos[:, 1], best_fut_pos[:, 0], 'b--', label='Best Prediction (Highest Prob)')

    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.title(f"Trajectory Visualization - epoch {epoch}, idx {idx}")
    plt.xlabel("Longitudinal (feet)")
    plt.ylabel("Lateral (feet)")
    
    # 3. WandB에 이미지 로그 업로드
    wandb.log({f"Visual/Best_Trajectory_{idx}": wandb.Image(plt)}, step= epoch)
    plt.close('all')


def integrate_distribution(v_dist, dt=0.1):
    """
    v_dist: [Steps, Batch, 5] -> (mu_vx, mu_vy, inv_sig_vx, inv_sig_vy, rho_v)
    Trajectron++의 공식을 기반으로 하되, 입력이 표준편차의 역수(1/sigma)임을 고려합니다.
    """
    mu_v = v_dist[:, :, 0:2]
    inv_sig_v = v_dist[:, :, 2:4] # 1/sigma_v
    rho_v = v_dist[:, :, 4]
    
    # 1. 평균 위치 통합 (기존과 동일)
    # mu_p = sum(v * dt)
    mu_p = t.cumsum(mu_v * dt, dim=0)
    
    # 2. 위치 분산 통합 (분산으로 변환 후 적분)
    # v_variance = (1 / inv_sig_v)^2 = sigma_v^2
    v_variance = 1.0 / (t.pow(inv_sig_v, 2) + 1e-6)
    
    # p_variance = sum(v_variance * dt^2) 
    p_variance = t.cumsum(v_variance * (dt**2), dim=0)
    
    # 최종 반환을 위해 다시 역수 표준편차로 변환: 1 / sqrt(p_variance)
    inv_sig_p = 1.0 / (t.sqrt(p_variance) + 1e-6)
    
    # 3. 상관계수(rho) 통합
    # 속도 도메인의 공분산: cov_v = rho_v * sigma_vx * sigma_vy
    # sigma_v = 1 / inv_sig_v 이므로 아래와 같이 계산
    sigma_v_x = 1.0 / (inv_sig_v[:, :, 0] + 1e-6)
    sigma_v_y = 1.0 / (inv_sig_v[:, :, 1] + 1e-6)
    cov_v_xy = rho_v * sigma_v_x * sigma_v_y
    
    # 위치 도메인의 공분산: cov_p = sum(cov_v * dt^2)
    cov_p_xy = t.cumsum(cov_v_xy * (dt**2), dim=0)
    
    # 위치 도메인의 상관계수: rho_p = cov_p / (sigma_px * sigma_py)
    # sigma_p = 1 / inv_sig_p 이므로: rho_p = cov_p * inv_sig_p_x * inv_sig_p_y
    rho_p = cov_p_xy * inv_sig_p[:, :, 0] * inv_sig_p[:, :, 1]
    rho_p = t.clamp(rho_p, -0.99, 0.99)
    
    return t.cat([mu_p, inv_sig_p, rho_p.unsqueeze(-1)], dim=-1)



class Evaluate():

    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1
        self.dt = 0.1

    def maskedMSETest(self, y_pred, y_gt, mask, dt):

        acc = t.zeros_like(mask)
        muX = y_pred[:, :, 0]
        muY = y_pred[:, :, 1]
        x = y_gt[:, :, 0]
        y = y_gt[:, :, 1]
        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
        acc[:, :, 0] = out
        acc[:, :, 1] = out
        acc = acc * mask
        lossVal = t.sum(acc[:, :, 0], dim=1)
        counts = t.sum(mask[:, :, 0], dim=1)
        loss = t.sum(acc) / t.sum(mask)
        return lossVal, counts, loss

    ## Helper function for log sum exp calculation: 一个计算公式
    def logsumexp(self, inputs, dim=None, keepdim=False):
        if dim is None:
            inputs = inputs.view(-1)
            dim = 0
        s, _ = t.max(inputs, dim=dim, keepdim=True)
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
        if not keepdim:
            outputs = outputs.squeeze(dim)
        return outputs

    def maskedNLLTest(self, fut_pred, lat_pred, lon_pred, fut, op_mask, num_lat_classes=3, num_lon_classes=3,
                      use_maneuvers=True):
        if use_maneuvers:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], num_lon_classes * num_lat_classes).to(device)
            count = 0
            for k in range(num_lon_classes):
                for l in range(num_lat_classes):
                    wts = lat_pred[:, l] * lon_pred[:, k]
                    wts = wts.repeat(len(fut_pred[0]), 1)
                    y_pred = fut_pred[k * num_lat_classes + l]
                    y_gt = fut
                    
                    y_pred = integrate_distribution(y_pred)
                    muX = y_pred[:, :, 0]
                    muY = y_pred[:, :, 1]
                    sigX = y_pred[:, :, 2]
                    sigY = y_pred[:, :, 3]
                    rho = y_pred[:, :, 4]
                    ohr = t.pow(1 - t.pow(rho, 2), -0.5)
                    x = y_gt[:, :, 0]
                    y = y_gt[:, :, 1]
                    # If we represent likelihood in feet^(-1):
                    # out = -(0.5 * t.pow(ohr, 2) * (
                    #         t.pow(sigX, 2) * t.pow(x - muX, 2) + 0.5 * t.pow(sigY, 2) * t.pow(
                    #     y - muY, 2) - rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                    #                 y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    out = -(0.5 * t.pow(ohr, 2) * (
                            t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(
                        y - muY, 2) - 2 * rho * t.pow(sigX, 1) * t.pow(sigY, 1) * (x - muX) * (
                                    y - muY)) - t.log(sigX * sigY * ohr) + 1.8379)
                    acc[:, :, count] = out + t.log(wts)
                    count += 1
            acc = -self.logsumexp(acc, dim=2)
            acc = acc * op_mask[:, :, 0]
            loss = t.sum(acc) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc, dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss
        else:
            acc = t.zeros(op_mask.shape[0], op_mask.shape[1], 1).to(device)
            y_pred = fut_pred

            y_pred = integrate_distribution(y_pred)

            y_gt = fut
            muX = y_pred[:, :, 0]
            muY = y_pred[:, :, 1]
            sigX = y_pred[:, :, 2]
            sigY = y_pred[:, :, 3]
            rho = y_pred[:, :, 4]
            ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # p
            x = y_gt[:, :, 0]
            y = y_gt[:, :, 1]
            # If we represent likelihood in feet^(-1):
            out = 0.5 * t.pow(ohr, 2) * (
                    t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY,
                                                                                2) - 2 * rho * t.pow(
                sigX, 1) * t.pow(sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
            acc[:, :, 0] = out
            acc = acc * op_mask[:, :, 0:1]
            loss = t.sum(acc[:, :, 0]) / t.sum(op_mask[:, :, 0])
            lossVal = t.sum(acc[:, :, 0], dim=1)
            counts = t.sum(op_mask[:, :, 0], dim=1)
            return lossVal, counts, loss

    def main(self, name, val, dataset_path:str): # name : name of model , val : validation vs test
        model_step = 1
        args['train_flag'] = not args['use_maneuvers'] # -> True이면 
        # args['train_flag'] = True
        l_path = args['path']
        generator = model.Generator(args=args)
        gdEncoder = model.GDEncoder(args=args)
        
        if name == 'best':
            path = os.path.join(l_path, 'best_model.pt')
            checkpoint = t.load(path, weights_only=True)
            gdEncoder.load_state_dict(checkpoint['gdEncoder'])
            generator.load_state_dict(checkpoint['generator'])
        else:
            generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.tar', map_location='cuda:0')) # epoch9_g
            gdEncoder.load_state_dict(t.load(l_path + '/epoch' + name + '_gd.tar', map_location='cuda:0')) # epoch9_gd

        generator = generator.to(device)
        gdEncoder = gdEncoder.to(device)
        generator.eval()
        gdEncoder.eval()
        
        # Load dataset
        if val: # Validation
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.NgsimDataset(dataset_path,t_f=50,d_s=1)#,t_f=30,d_s=1
                else:
                    t2 = lo.NgsimDataset(dataset_path,t_f=50, d_s=1)#,t_f=30,d_s=1
            else:
                t2 = lo_highD.HighdDataset(dataset_path,d_s=1)
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn)  # 6716batch
        else: # Test
            # ------------------------------------------------------------
            # a = generator.mapping
            # xx = t.tensor([[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1],
            #                [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1],
            #                [0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1]], dtype=t.float).permute(1, 0).to(
            #     device)
            # softa = t.cat(t.softmax(t.matmul(a, xx), dim=0).chunk(9, -1), dim=1).squeeze().cpu().detach().numpy()
            # a = t.cat(a.chunk(6, -1), dim=1).squeeze().cpu().detach().numpy()
            # result = np.concatenate((a, softa), axis=-1).transpose()
            # data = pd.DataFrame(result)
            # data.to_excel(writer, name, float_format='%.5f')
            # writer.save()
            if dataset == "ngsim":
                if args['lon_length'] == 3:
                    t2 = lo.NgsimDataset(dataset_path,t_f=50, d_s=1)#,t_f=30,d_s=1
                else:
                    t2 = lo.NgsimDataset(dataset_path,t_f=50, d_s=1)#,t_f=30,d_s=1
            else:
                t2 = lo_highD.HighdDataset(dataset_path,d_s=1)
            valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                       collate_fn=t2.collate_fn)

        lossVals = t.zeros(args['out_length']).to(device)
        counts = t.zeros(args['out_length']).to(device)
        avg_val_loss = 0
        all_time = 0
        nbrsss = 0

        # HJ
        # 모든 예측과 정답을 저장할 리스트 초기화
        # <<<--- 2. 전체 예측과 정답을 저장할 리스트 초기화 ---###
        all_lat_preds = []
        all_lat_gts = []
        all_lon_preds = []
        all_lon_gts = []
    
        val_batch_count = len(valDataloader) # 배치 총 12068개

        print("begin.................................", name)
        with(t.no_grad()): # 미분 비활성화 (value 중이니까 gradient update 필요 X)
            for idx, data in enumerate(valDataloader):
                print(f"{idx}/{len(valDataloader)}", end=' ')
                hist, nbrs, mask, lat_enc, lon_enc, fut_vel, fut_pos, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data

                # if idx == 10:
                #     return
                import numpy as np
                print(np.array(map_positions).shape)

                # 128 : batch_size
                # hist : history X of ego (16, 128, 2) (T=16 of x,y) -> 31, 0.1hz
                # nbrs : history X of nbrs (16, 990 ~ 1100, 2)(T=16 of x,y) -> 31, 0.1hz
                # mask : LSTM의 hidden state masking용 (128, 3, 13, 64) (3 x 13 grid, 64dim lstm hidden state)
                # lat_enc : lat_maneuver (128, 3)
                # lon_enc : lat_maneuver (128, 3)
                # fut : future of ego (GT) (25, 128, 2) (T=25 of x,y) -> 30, 0.1hz 
                # op_mask : data들간의 length가 다르면 padding이 들어가기 때문에, 이를 걸러주기 위한 변수
                # va : velocity & acceleration
                # nbrsva : velocity & acceleration of nbrs 
                # lane : ego lane
                # nbrslane : nbrs lane 
                # Cls : Car class
                # nbrscls : Nbrs class
                               
                # print(f'lane : {lane.shape}')
                # print(f'nbrslane : {nbrslane.shape}')
                # print(f'cls : {cls.shape}')
                # print(f'nbrscls : {nbrscls.shape}')
                # print(f'va : {va.shape}')
                # print(f'nbrsva : {nbrsva.shape}')
                
                hist = hist.to(device)
                nbrs = nbrs.to(device)
                mask = mask.to(device)
                lat_enc = lat_enc.to(device)
                lon_enc = lon_enc.to(device)
                fut_vel = fut_vel[:args['out_length'], :, :]
                fut_vel = fut_vel.to(device)
                
                fut_pos = fut_pos[:args['out_length'], :, :]
                fut_pos = fut_pos.to(device)
                op_mask = op_mask[:args['out_length'], :, :]
                op_mask = op_mask.to(device)
                va = va.to(device)
                nbrsva = nbrsva.to(device)
                lane = lane.to(device)
                nbrslane = nbrslane.to(device)
                cls = cls.to(device)
                nbrscls = nbrscls.to(device)
                map_positions = map_positions.to(device)
                
                te = time.time()
                values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls) # values 계산, values = interaction 반영된 것인지? 
                fut_pred, lat_pred, lon_pred = generator(values, lat_enc, lon_enc) # Predictoin 수행
            
                # <<< HJ >>>
                # 횡방향(Lateral) 의도 결과 저장
                pred_lat_indices = t.argmax(lat_pred, dim=-1).cpu().tolist()
                gt_lat_indices = t.argmax(lat_enc, dim=-1).cpu().tolist()
                all_lat_preds.extend(pred_lat_indices)
                all_lat_gts.extend(gt_lat_indices)

                # 종방향(Longitudinal) 의도 결과 저장
                pred_lon_indices = t.argmax(lon_pred, dim=-1).cpu().tolist()
                gt_lon_indices = t.argmax(lon_enc, dim=-1).cpu().tolist()
                all_lon_preds.extend(pred_lon_indices)
                all_lon_gts.extend(gt_lon_indices)
                
                pred_lat_indices_tensor = t.argmax(lat_pred, dim=-1)
                pred_lon_indices_tensor = t.argmax(lon_pred, dim=-1)
                gt_lat_indices_tensor = t.argmax(lat_enc, dim=-1)

                # [수정] 횡방향 예측이 틀린 샘플의 배치 내 인덱스 찾기
                wrong_lat_indices = (pred_lat_indices_tensor != gt_lat_indices_tensor).nonzero(as_tuple=True)[0]
                
                all_time += time.time() - te
                #nbrsss += 1
                #if nbrsss > args['time']*args['num_worker']:
                #    print(all_time / nbrsss,"ref time")
                if not args['train_flag']: # multi-modal -> eval 중엔 false
                    indices = []
                    if args['val_use_mse']:
                        fut_pred_max = t.zeros_like(fut_pred[0])
                        for k in range(lat_pred.shape[0]):  # 128
                            lat_man = t.argmax(lat_pred[k, :]).detach()
                            lon_man = t.argmax(lon_pred[k, :]).detach()
                            index = lon_man * 3 + lat_man # sample마다 가장 확률 높은 maneuver 선택
                            indices.append(index)
                            fut_pred_max[:, k, :] = fut_pred[index][:, k, :]
                        
                        fut_pred_max = integrate_distribution(fut_pred_max)
                        l, c, loss = self.maskedMSETest(fut_pred_max, fut_pos, op_mask, self.dt) # 함수 내부에서 integrate_distribution 진행
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut_pos, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                    if not val and self.drawImg : # and len(wrong_lat_indices) > 0 :
                        lat_man = t.argmax(lat_enc, dim=-1).detach()
                        lon_man = t.argmax(lon_enc, dim=-1).detach()
                        # self.draw_wrong_samples(hist, fut_pos, nbrs, mask, fut_pred, args['train_flag'], lon_man, lat_man, op_mask,
                        #           indices, wrong_lat_indices, pred_lat_indices_tensor, pred_lon_indices_tensor)
                        self.draw(hist, fut_pos, nbrs, mask, fut_pred, args['train_flag'], lon_man, lat_man, op_mask,
                                  indices)
                else: # single-modal
                    if args['val_use_mse']:
                        fut_pred = integrate_distribution(fut_pred)
                        l, c, loss = self.maskedMSETest(fut_pred, fut_pos, op_mask, self.dt)
                    else:
                        l, c, loss = self.maskedNLLTest(fut_pred, lat_pred, lon_pred, fut_pos, op_mask,
                                                        use_maneuvers=args['use_maneuvers'])
                    if not val and self.drawImg:
                        lat_man = t.argmax(lat_enc, dim=-1).detach()
                        lon_man = t.argmax(lon_enc, dim=-1).detach()
                        self.draw(hist, fut_pos, nbrs, mask, fut_pred, args['train_flag'], lon_man, lat_man, op_mask, None) # draw 코드 좌표축 안맞아서 예측 그래프 반대로 그려짐

                if val:
                    if idx % 100 == 0: # 100개 배치마다 하나씩 시각화
                        probs = t.outer(lon_pred[0], lat_pred[0]).flatten() # 첫 번째 샘플(index 0) 기준
                        best_mode_idx = t.argmax(probs).item()
                        best_fut_vel = fut_pred[best_mode_idx][:, 0, :2] # [Steps, 2]
                        # 로더에서 가져온 위치 GT(fut_pos)와 과거(hist) 전달
                        # hist[:, 0, :]는 첫 번째 샘플의 과거 경로
                        epoch = int(name.split('_')[0])
                        plot_trajectory_to_wandb(
                            hist=hist[:, 0, :].cpu(), 
                            fut_pos_gt=fut_pos[:, 0, :].cpu(), # 로더에서 위치 GT를 추가로 반환한다고 가정
                            best_fut_vel=best_fut_vel, 
                            epoch=epoch,
                            idx= idx, 
                            dt=self.dt
                        )

                lossVals += l.detach()
                counts += c.detach()
                avg_val_loss += loss.item()
                if idx == int(val_batch_count / 4) * model_step:
                    print('process:', model_step / 4)
                    model_step += 1
            # tqdm.write('valmse:', avg_val_loss / val_batch_count)
            
            # <<< HJ : 평가 루프 종료 후, 최종 정확도 계산 및 혼동 행렬 생성 >>>
            print("\n" + "---" * 20)
            print("      Maneuver Classification Performance Summary")
            print("---" * 20)

            # --- 전체 정확도 계산 ---
            lat_accuracy = accuracy_score(all_lat_gts, all_lat_preds)
            lon_accuracy = accuracy_score(all_lon_gts, all_lon_preds)
            print(f"Overall Lateral Maneuver Accuracy: {lat_accuracy * 100:.2f}%")
            print(f"Overall Longitudinal Maneuver Accuracy: {lon_accuracy * 100:.2f}%")
            print("-" * 60)

            # --- 횡방향(Lateral) 혼동 행렬 생성 및 시각화 ---
            lat_class_names = ['Lane Keep', 'Lane Change Left', 'Lane Change Right'] 
            lat_cm = confusion_matrix(all_lat_gts, all_lat_preds)
            df_lat_cm = pd.DataFrame(lat_cm, index=lat_class_names, columns=lat_class_names)
            
            plt.figure(figsize=(10, 7))
            sns.heatmap(df_lat_cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Lateral Maneuver Confusion Matrix', fontsize=16)
            plt.ylabel('Actual Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.savefig('./figures/lateral_confusion_matrix.png') # 이미지 파일로 저장
            print("Lateral confusion matrix saved as lateral_confusion_matrix.png")

            # --- 종방향(Longitudinal) 혼동 행렬 생성 및 시각화 ---
            lon_class_names = ['Constant', 'Deceleration', 'Acceleration']
            all_lon_labels = [0, 1, 2]
            lon_cm = confusion_matrix(all_lon_gts, all_lon_preds, labels=all_lon_labels)
            df_lon_cm = pd.DataFrame(lon_cm, index=lon_class_names, columns=lon_class_names)

            plt.figure(figsize=(10, 7))
            sns.heatmap(df_lon_cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Longitudinal Maneuver Confusion Matrix', fontsize=16)
            plt.ylabel('Actual Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.savefig('./figures/longitudinal_confusion_matrix.png') # 이미지 파일로 저장
            print("Longitudinal confusion matrix saved as longitudinal_confusion_matrix.png")
            print("---" * 20)

            val_loss = avg_val_loss / val_batch_count
            if args['val_use_mse']:
                print('valmse:', val_loss)
                print(t.pow(lossVals / counts, 0.5) * 0.3048)  # Calculate RMSE and convert from feet to meters
            else:
                print('valnll:', val_loss)
                print(lossVals / counts)
            # print(lossVals/counts*0.3048)
            return val_loss
            

    def add_car(self, plt, x, y, alp):
        plt.gca().add_patch(plt.Rectangle(
            (x - 5*self.scale, y - 2.5*self.scale),  
            10*self.scale,  
            5*self.scale, 
            color='maroon',
            alpha=alp
        ))


    def draw_wrong_samples(self, hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices, wrong_lat_indices, pred_lat_labels, pred_lon_labels):

        from matplotlib.ticker import MultipleLocator
        lat_map = {0: 'Lane Keep', 1: 'Lane Change Left', 2: 'Lane Change Right'}
        lon_map = {0: 'Constant', 1: 'Decel', 2: 'Accel'} # 인덱스가 0,1,2일 경우 (데이터셋에 따라 확인 필요)
        
        hist = hist.cpu()
        fut = fut.cpu()
        nbrs = nbrs.cpu()
        mask = mask.cpu()
        op_mask = op_mask.cpu()
        IPL = 0
        for i in wrong_lat_indices:
            i = i.item()
 
            lon_man_i = lon_man[i].item() # (batch,)
            lat_man_i = lat_man[i].item()
            
           # GT 정보
            gt_lat_txt = lat_map.get(lat_man_i, "Unknown")
            gt_lon_txt = lon_map.get(lon_man_i, "Unknown")
            
            # Pred 정보 (main에서 넘겨받은 예측 인덱스 사용)
            pred_lat_txt = lat_map.get(pred_lat_labels[i].item(), "Unknown")
            pred_lon_txt = lon_map.get(pred_lon_labels[i].item(), "Unknown")

            plt.axis('on')
            plt.figure(dpi=300)
            plt.ylim(-18 * self.scale, 18 * self.scale)
            
            # <<<--- 이 부분에 눈금 설정 코드 추가 ---###
            ax = plt.gca() # 현재 축(Axes) 객체 가져오기
            # [추가] 텍스트 표시 로직
            # 하얀색 박스 배경을 넣어 가독성 확보
            info_text = (f"[GT] Lat: {gt_lat_txt}, Lon: {gt_lon_txt}\n"
                        f"[Pred] Lat: {pred_lat_txt}, Lon: {pred_lon_txt}")
            
            # 좌측 상단 좌표에 텍스트 배치
            plt.text(-170 * self.scale * self.prop, 15 * self.scale, info_text, 
                    fontsize=7, color='black', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            # x축의 주 눈금 간격을 10 단위로 설정
            # 기존 코드의 scale과 prop을 함께 고려하여 실제 단위에 맞게 설정
            ax.xaxis.set_major_locator(MultipleLocator(10))

            # y축의 주 눈금 간격을 5 단위로 설정
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # <<<--- 이 부분에 눈금 숫자 크기 조절 코드 추가 ---###
            # which='major'는 주 눈금에만 적용하겠다는 의미입니다.
            # labelsize 숫자를 조절해 원하는 크기로 변경할 수 있습니다.
            ax.tick_params(axis='both', which='major', labelsize=4)

            # (선택) 눈금에 맞춰 격자(grid)를 다시 그려 가독성을 높임
            ax.grid(which='major', linestyle='--', linewidth='0.5', color='gray')
            # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
            # <<<------------------------------------###

            plt.xlim(-180 * self.scale * self.prop, 180 * self.scale * self.prop)
            # plt.figure(dpi=300, figsize=(100 * self.scale * self.prop,40 * self.scale))
            # plt.hlines([-18, -6, 6, 18], -180, 180, colors="c", linestyles="dashed")
            IPL_i = mask[i, :, :, :].sum().sum()
            IPL_i = int((IPL_i / 64).item())
            for ii in range(IPL_i):
                plt.plot(nbrs[:, IPL + ii, 1] * self.scale * self.prop, nbrs[:, IPL + ii, 0] * self.scale, ':',
                         color='blue',
                         linewidth=0.5)
                self.add_car(plt, nbrs[-1, IPL + ii, 1]* self.scale * self.prop, nbrs[-1, IPL + ii, 0]* self.scale , alp=0.2)
            IPL = IPL + IPL_i
            plt.plot(hist[:, i, 1] * self.scale * self.prop, hist[:, i, 0] * self.scale, ':', color='red',
                     linewidth=0.5)
            self.add_car(plt, hist[-1, i, 1], hist[-1, i, 0], alp=1)
            plt.plot(fut[:, i, 1] * self.scale * self.prop, fut[:, i, 0] * self.scale, '-', color='black',
                     linewidth=0.5)
            if train_flag:
                fut_pred = fut_pred.detach().cpu()
                
                # plt.plot(fut_pred[:, i, 1], fut_pred[:, i, 0], 'p', color='green', markersize=0.6)
                plt.plot(fut_pred[:, i, 1] * self.scale * self.prop, fut_pred[:, i, 0] * self.scale, color='green',
                         linewidth=0.2)
                muX = fut_pred[:, i, 0]
                muY = fut_pred[:, i, 1]
                x = fut[:, i, 0]
                y = fut[:, i, 1]
                max_y = y[-1] - y[0]
                out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
                acc = out * op_mask[:, i, 0]
                loss = t.sum(acc) / t.sum(op_mask[:, i, 0])
            else:
                for j in range(len(fut_pred)):
                    fut_pred_i = fut_pred[j].detach().cpu()
                    fut_pred_i = integrate_distribution(fut_pred_i)
                    if j == indices[i].item():
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                 color='red', linewidth=0.2)
                        muX = fut_pred_i[:, i, 0]
                        muY = fut_pred_i[:, i, 1]
                        x = fut[:, i, 0]
                        y = fut[:, i, 1]
                        max_y = y[-1] - y[0]
                        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
                        acc = out * op_mask[:, i, 0]
                        loss = t.sum(acc) / t.sum(op_mask[:, i, 0])
                    else:
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale, # feet to meter
                                 color='green', linewidth=0.2)
            plt.gca().set_aspect('equal', adjustable='box')
            
            # 
            dir_path = Path(f'./Wrong_pic/{lon_man_i + 1}_{lat_man_i + 1}')
            dir_path.mkdir(parents=True, exist_ok=True)
            
            plt.savefig('./Wrong_pic/' + str(lon_man_i + 1) + '_' + str(lat_man_i + 1) + '/' + str(self.op) + '.png')
            self.op += 1
            # fig.clf()
            plt.close()

    def draw(self, hist, fut, nbrs, mask, fut_pred, train_flag, lon_man, lat_man, op_mask, indices):

        from matplotlib.ticker import MultipleLocator
        hist = hist.cpu()
        fut = fut.cpu()
        nbrs = nbrs.cpu()
        mask = mask.cpu()
        op_mask = op_mask.cpu()
        IPL = 0
        for i in range(hist.size(1)):
            lon_man_i = lon_man[i].item() # (batch,)
            lat_man_i = lat_man[i].item()
            plt.axis('on')
            plt.figure(dpi=300)
            plt.ylim(-18 * self.scale, 18 * self.scale)
            
            # <<<--- 이 부분에 눈금 설정 코드 추가 ---###
            ax = plt.gca() # 현재 축(Axes) 객체 가져오기

            # x축의 주 눈금 간격을 10 단위로 설정
            # 기존 코드의 scale과 prop을 함께 고려하여 실제 단위에 맞게 설정
            ax.xaxis.set_major_locator(MultipleLocator(10))

            # y축의 주 눈금 간격을 5 단위로 설정
            ax.yaxis.set_major_locator(MultipleLocator(1))

            # <<<--- 이 부분에 눈금 숫자 크기 조절 코드 추가 ---###
            # which='major'는 주 눈금에만 적용하겠다는 의미입니다.
            # labelsize 숫자를 조절해 원하는 크기로 변경할 수 있습니다.
            ax.tick_params(axis='both', which='major', labelsize=4)

            # (선택) 눈금에 맞춰 격자(grid)를 다시 그려 가독성을 높임
            ax.grid(which='major', linestyle='--', linewidth='0.5', color='gray')
            # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.5)
            # <<<------------------------------------###

            plt.xlim(-180 * self.scale * self.prop, 180 * self.scale * self.prop)
            # plt.figure(dpi=300, figsize=(100 * self.scale * self.prop,40 * self.scale))
            # plt.hlines([-18, -6, 6, 18], -180, 180, colors="c", linestyles="dashed")
            IPL_i = mask[i, :, :, :].sum().sum()
            IPL_i = int((IPL_i / 64).item())
            for ii in range(IPL_i):
                plt.plot(nbrs[:, IPL + ii, 1] * self.scale * self.prop, nbrs[:, IPL + ii, 0] * self.scale, ':',
                         color='blue',
                         linewidth=0.5)
                self.add_car(plt, nbrs[-1, IPL + ii, 1]* self.scale * self.prop, nbrs[-1, IPL + ii, 0]* self.scale , alp=0.2)
            IPL = IPL + IPL_i
            plt.plot(hist[:, i, 1] * self.scale * self.prop, hist[:, i, 0] * self.scale, ':', color='red',
                     linewidth=0.5)
            self.add_car(plt, hist[-1, i, 1], hist[-1, i, 0], alp=1)
            plt.plot(fut[:, i, 1] * self.scale * self.prop, fut[:, i, 0] * self.scale, '-', color='black',
                     linewidth=0.5)
            if train_flag:
                fut_pred = fut_pred.detach().cpu()
                
                # plt.plot(fut_pred[:, i, 1], fut_pred[:, i, 0], 'p', color='green', markersize=0.6)
                plt.plot(fut_pred[:, i, 1] * self.scale * self.prop, fut_pred[:, i, 0] * self.scale, color='green',
                         linewidth=0.2)
                muX = fut_pred[:, i, 0]
                muY = fut_pred[:, i, 1]
                x = fut[:, i, 0]
                y = fut[:, i, 1]
                max_y = y[-1] - y[0]
                out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
                acc = out * op_mask[:, i, 0]
                loss = t.sum(acc) / t.sum(op_mask[:, i, 0])
            else:
                for j in range(len(fut_pred)):
                    fut_pred_i = fut_pred[j].detach().cpu()
                    fut_pred_i = integrate_distribution(fut_pred_i)
                    if j == indices[i].item():
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale,
                                 color='red', linewidth=0.2)
                        muX = fut_pred_i[:, i, 0]
                        muY = fut_pred_i[:, i, 1]
                        x = fut[:, i, 0]
                        y = fut[:, i, 1]
                        max_y = y[-1] - y[0]
                        out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
                        acc = out * op_mask[:, i, 0]
                        loss = t.sum(acc) / t.sum(op_mask[:, i, 0])
                    else:
                        plt.plot(fut_pred_i[:, i, 1] * self.scale * self.prop, fut_pred_i[:, i, 0] * self.scale, # feet to meter
                                 color='green', linewidth=0.2)
            plt.gca().set_aspect('equal', adjustable='box')
            
            # 
            dir_path = Path(f'./pic/{lon_man_i + 1}_{lat_man_i + 1}')
            dir_path.mkdir(parents=True, exist_ok=True)
            
            plt.savefig('./pic/' + str(lon_man_i + 1) + '_' + str(lat_man_i + 1) + '/' + str(self.op) + '.png')
            self.op += 1
            # fig.clf()
            plt.close()

            # from xlrd import open_workbook
            # from xlutils.copy import copy
            # r_xls = open_workbook("test.xls")  
            # print(lon_man_i * 3 + lat_man_i)
            # row = r_xls.sheets()[lon_man_i * 3 + lat_man_i].nrows  
            # excel = copy(r_xls) 
            # table = excel.get_sheet(lon_man_i * 3 + lat_man_i)  
            # table.write(row, 0, str(self.op - 1))
            # table.write(row, 1, max_y.item())  
            # table.write(row, 2, loss.item())  
            # table.write(row, 3, str(acc))
            # excel.save("./test.xls")  


if __name__ == '__main__':

# generator.load_state_dict(t.load(l_path + '/epoch' + name + '_g.tar', map_location='cuda:0')) 
    test_dataset_path = './stdan/NGsim/0113_ratio211/TestSet.mat' 
    # test_dataset_path = './stdan/data/dataset_t_v_t/TestSet.mat' 

    # names = ['10hz']
    evaluate = Evaluate()
    evaluate.main(name='best', val=False, dataset_path=test_dataset_path)
