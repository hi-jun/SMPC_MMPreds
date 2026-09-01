from torch.utils.data import DataLoader
import loader2_hdf5_vel as lo
import loader2_highD_1107_final as lo_highD
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from early_stopping import EarlyStopping

import os
from evaluate5f_vel import Evaluate
from config_vel import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch as t
import wandb
import argparse

import torch

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
    mu_p = torch.cumsum(mu_v * dt, dim=0)
    
    # 2. 위치 분산 통합 (분산으로 변환 후 적분)
    # v_variance = (1 / inv_sig_v)^2 = sigma_v^2
    v_variance = 1.0 / (torch.pow(inv_sig_v, 2) + 1e-6)
    
    # p_variance = sum(v_variance * dt^2) 
    p_variance = torch.cumsum(v_variance * (dt**2), dim=0)
    
    # 최종 반환을 위해 다시 역수 표준편차로 변환: 1 / sqrt(p_variance)
    inv_sig_p = 1.0 / (torch.sqrt(p_variance) + 1e-6)
    
    # 3. 상관계수(rho) 통합
    # 속도 도메인의 공분산: cov_v = rho_v * sigma_vx * sigma_vy
    # sigma_v = 1 / inv_sig_v 이므로 아래와 같이 계산
    sigma_v_x = 1.0 / (inv_sig_v[:, :, 0] + 1e-6)
    sigma_v_y = 1.0 / (inv_sig_v[:, :, 1] + 1e-6)
    cov_v_xy = rho_v * sigma_v_x * sigma_v_y
    
    # 위치 도메인의 공분산: cov_p = sum(cov_v * dt^2)
    cov_p_xy = torch.cumsum(cov_v_xy * (dt**2), dim=0)
    
    # 위치 도메인의 상관계수: rho_p = cov_p / (sigma_px * sigma_py)
    # sigma_p = 1 / inv_sig_p 이므로: rho_p = cov_p * inv_sig_p_x * inv_sig_p_y
    rho_p = cov_p_xy * inv_sig_p[:, :, 0] * inv_sig_p[:, :, 1]
    rho_p = torch.clamp(rho_p, -0.99, 0.99)
    
    return torch.cat([mu_p, inv_sig_p, rho_p.unsqueeze(-1)], dim=-1)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=None, help='WandB run name')
wandb_args, _ = parser.parse_known_args()

# Start a new wandb run to track this script.
run = wandb.init(
    project="stdan_v1",
    name = wandb_args.name,
    
    # Track hyperparameters and run metadata.
    config={
        "run_name" : wandb_args.name,
        "learning_rate": 0.0005,
        "decay_rate" : 0.8,
        "dataset": "0113",
        "epochs": 200,
        "pre_epoch": 10
    },
)

def maskedNLL(y_pred, y_gt, mask):
    # mask = t.cat((mask[:15, :, :], t.zeros_like(mask[15:, :, :])), dim=0)
    acc = t.zeros_like(mask)
    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]
    sigY = y_pred[:, :, 3]
    rho = y_pred[:, :, 4]
    ohr = t.pow(1 - t.pow(rho, 2), -0.5)  # (1-rhp^2)^0.5
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    # If we represent likelihood in feet^(-1)
    out = 0.5 * t.pow(ohr, 2) * (
            t.pow(sigX, 2) * t.pow(x - muX, 2) + t.pow(sigY, 2) * t.pow(y - muY, 2) - 2 * rho * t.pow(sigX,
                                                                                                      1) * t.pow(
        sigY, 1) * (x - muX) * (y - muY)) - t.log(sigX * sigY * ohr) + 1.8379
    # If we represent likelihood in m^(-1):meter out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX,
    # 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2) - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX)
    # * (y - muY)) - torch.log(sigX * sigY * ohr) + 1.8379 - 0.5160
    acc[:, :, 0] = out
    acc[:, :, 1] = out

    acc = acc * mask
    lossVal = t.sum(acc) / t.sum(mask)
    return lossVal

def MSELoss2(g_out, fut, mask):
    acc = t.zeros_like(mask)
    muX = g_out[:, :, 0]
    muY = g_out[:, :, 1]
    x = fut[:, :, 0]
    y = fut[:, :, 1]
    out = t.pow(x - muX, 2) + t.pow(y - muY, 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    
    acc = acc * mask
    lossVal = t.sum(acc) / t.sum(mask)
    return lossVal


def CELoss(pred, target):
    epsilon = 1e-9
    value = t.log(t.sum(pred * target, dim=-1)+epsilon)
    return -t.sum(value) / value.shape[0]


def main():
    if t.cuda.is_available():
        print('CUDA working')
    else:
        raise Exception('CUDA not working')
    print(f'Now device is : {device}')
    
    args['train_flag'] = True
    evaluate = Evaluate()
        #   test 
    # t1 = lo.NgsimDataset('data/5feature/TrainSet.mat')
    # t1.collate_fn([t1.__getitem__(4587)])
    gdEncoder = model.GDEncoder(args)
    generator = model.Generator(args)
    
    wandb.watch(gdEncoder)
    wandb.watch(generator)
    
    # generator.load_state_dict(t.load('checkpoints/4fx/epoch0_g.tar'))
    # discriminator.load_state_dict(t.load('checkpoints/4fx/epoch0_d.tar'))
    gdEncoder = gdEncoder.to(device)
    generator = generator.to(device)
    gdEncoder.train()
    generator.train()
    if dataset == "ngsim":
        if args['lon_length'] == 3:
            t1 = lo.NgsimDataset('./stdan/NGsim/0113_ratio211/TrainSet.mat' , t_f = 50, d_s=1) # ../
        else:
            t1 = lo.NgsimDataset('../data/5feature/TrainSet.mat') # ?
        trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                     collate_fn=t1.collate_fn) 
    else:
        t1 = lo_highD.HighdDataset('./stdan/highD/TrainSet.mat',d_s=1)
        trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                     collate_fn=t1.collate_fn)  
    optimizer_gd = optim.Adam(gdEncoder.parameters(), lr=learning_rate)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    # scheduler_gd = ExponentialLR(optimizer_gd, gamma=0.8)
    # scheduler_g = ExponentialLR(optimizer_g, gamma=0.8)
    scheduler_gd = ReduceLROnPlateau(optimizer_gd, mode='min', factor=0.5, patience=5)
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5)


    epoch_losses_g = []  # TOTAL
    epoch_losses_g1 = [] # MSE
    epoch_losses_gx = [] # gx2 + gx3
    epoch_losses_gx2 = [] # gx2 
    epoch_losses_gx3 = [] # gx2 
    
    # 15번 정도 참아보고 안 줄면 끈다 (데이터 1.4만개 기준 적절)
    save_path = os.path.join(args['path'], 'best_model.pt')
    early_stopping = EarlyStopping(patience=50, verbose=True, path=save_path)
    
    for epoch in range(args['epoch']):
        print("epoch:", epoch + 1, 'lr', optimizer_g.param_groups[0]['lr'])

        loss_gi = 0
        loss_gi1 = 0
        loss_gix = 0
        loss_gx_2i = 0
        loss_gx_3i = 0
        
        loss_g_sum = 0
        loss_g1_sum = 0
        loss_gx_sum = 0
        loss_gx_2_sum = 0
        loss_gx_3_sum = 0
        for idx, data in enumerate(tqdm(trainDataloader)):
            hist, nbrs, mask, lat_enc, lon_enc, fut_vel, fut_pos, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions = data
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
            dis = dis.to(device)
            nbrsdis = nbrsdis.to(device)
            map_positions = map_positions.to(device)
            cls = cls.to(device)
            nbrscls = nbrscls.to(device)

            values = gdEncoder(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            g_out, lat_pred, lon_pred = generator(values, lat_enc, lon_enc)

            dt = 0.1
            if args['use_mse']:
                g_out = integrate_distribution(g_out)
                loss_g1 = MSELoss2(g_out, fut_pos, op_mask)
            else:
                if epoch < args['pre_epoch']:
                    g_out = integrate_distribution(g_out)
                    loss_g1 = MSELoss2(g_out, fut_pos, op_mask)
                else:
                    g_out = integrate_distribution(g_out)
                    loss_g1 = maskedNLL(g_out, fut_pos, op_mask)
            loss_gx_3 = CELoss(lat_pred, lat_enc) # Cross Entropy
            loss_gx_2 = CELoss(lon_pred, lon_enc) # Cross Entropy
            loss_gx = loss_gx_3 + loss_gx_2 # Cross entropy of lat & lon
            
            intention_weight = args['intention_weight']
            loss_g = loss_g1 + intention_weight * loss_gx # data 한 개당 loss
            optimizer_g.zero_grad()
            optimizer_gd.zero_grad()
            loss_g.backward() # calculate gradient
            
            a = t.nn.utils.clip_grad_norm_(generator.parameters(), 10)
            a = t.nn.utils.clip_grad_norm_(gdEncoder.parameters(), 10)
            optimizer_g.step() # take one step using gradient
            optimizer_gd.step()

            loss_gi += loss_g.item()
            loss_gi1 += loss_g1.item()
            loss_gix += intention_weight*loss_gx.item()
            loss_gx_2i += intention_weight*loss_gx_2.item()
            loss_gx_3i += intention_weight*loss_gx_3.item()
            # g_out_i +=  g_out.item()


            loss_g_sum += loss_g.item()
            loss_g1_sum += loss_g1.item()
            loss_gx_sum += intention_weight*loss_gx.item()
            loss_gx_2_sum += intention_weight*loss_gx_2.item()
            loss_gx_3_sum += intention_weight*loss_gx_3.item()
            

            if idx % 1000 == 999:
                print('mse/NLL:', loss_gi1 / 1000, '|loss_gx_2 (lon):', loss_gx_2i / 1000, '|loss_gx_3 (lat)', loss_gx_3i / 1000)
                # print('pred :', g_out_i / 1000, )
                
                loss_gi = 0 # data가 10000개일 때마다 loss 출력하기 위한 용도
                loss_gi1 = 0
                loss_gix = 0
                loss_gx_2i = 0
                loss_gx_3i = 0
                
        run.log({"pred_loss" : loss_g1_sum/len(trainDataloader),
                 "maneuver_loss" : loss_gx_sum/len(trainDataloader), 
                 "lat_loss" : loss_gx_3_sum/len(trainDataloader),
                 "lon_loss" : loss_gx_2_sum/len(trainDataloader),
                 "total_loss" : loss_g_sum/len(trainDataloader),
                 },step=epoch+1)
        
        epoch_losses_g.append(loss_g_sum / len(trainDataloader))  # TOTAL
        epoch_losses_g1.append(loss_g1_sum / len(trainDataloader)) # MSE
        epoch_losses_gx.append(loss_gx_sum / len(trainDataloader)) # gx2 + gx3
        epoch_losses_gx2.append(loss_gx_2_sum / len(trainDataloader)) # gx2 
        epoch_losses_gx3.append(loss_gx_3_sum / len(trainDataloader)) # gx2 
            # if idx == int(len(trainDataloader) / 4) * model_step:
            #     print('mse:', loss_gi1 / int(len(trainDataloader) / 4), '|c1:',
            #           loss_gix / int(len(trainDataloader) / 4), '|c:', loss_gi3 / int(len(trainDataloader) / 4), '|d:',
            #           loss_gi4 / int(len(trainDataloader) / 4))
            #     loss_gi1 = 0
            #     loss_gix = 0
            #     loss_gi3 = 0
            #     loss_gi4 = 0
            #     model_step += 1
            #     if model_step == 4:
            #         model_step = 1
            #     if epoch >= 4:
            #         evaluate(name=str(epoch) + str(model_step), valDataloader=valDataloader, device=device,
            #                  cdgEncoder=cgdEncoder,
            #                  generator=generator, discriminator=discriminator, classified=classified,
            #                  f_length=args['f_length'], use_maneuvers=args['use_maneuvers'])

        save_model(name=str(epoch + 1)+'_10hz', gdEncoder=gdEncoder,
                   generator=generator, path = args['path'])
        eval_dataset_path = './stdan/NGsim/0113_ratio211/ValSet.mat' 
        val_loss = evaluate.main(name=str(epoch + 1)+'_10hz', val=True, dataset_path=eval_dataset_path)
        wandb.log({
            "Evaluation Loss": val_loss
        }, step=epoch+1)

        state_dict_pack = {
            'gdEncoder': gdEncoder.state_dict(),
            'generator': generator.state_dict()
        }
        
        # EarlyStopping 객체 호출 (val_loss와 묶은 모델 전달)
        early_stopping(val_loss, state_dict_pack)

        # [4] 조기 종료 조건 확인
        if early_stopping.early_stop:
            print("Early stopping triggered! Training stopped.")
            break

        scheduler_gd.step(val_loss)
        scheduler_g.step(val_loss)

    print("Loading the best model found during training...")
    
    # 저장된 best_model.pt 불러오기
    checkpoint = t.load(save_path)
    gdEncoder.load_state_dict(checkpoint['gdEncoder'])
    generator.load_state_dict(checkpoint['generator'])

    # =====================================================================
    # Test Set에 대한 최종 평가 
    # =====================================================================
    # print("Starting evaluation on Test Set...")
    
    # # Test Dataset 로드
    # test_mat_path = './stdan/NGsim/data_ratio/TestSet.mat'
    # if os.path.exists(test_mat_path):        
    #     # Test Loss 계산 (evaluate.main 함수 활용 또는 직접 루프)
    #     # evaluate.py의 구조에 따라 다를 수 있으나, 보통 val=True 옵션으로 호출 가능
    #     test_loss = evaluate.main(name='FINAL_TEST', val=False, dataset_path=test_mat_path) 
        
    #     print(f"Final Test Set Loss: {test_loss:.4f}")
    #     wandb.log({"Final Test Loss": test_loss})
    # else:
    #     print("TestSet.mat not found. Skipping final evaluation.")

    wandb.finish()

    # --- Graph Plotting ---
    # [수정 3] range 길이를 실제 학습된 epoch 수(len)에 맞춤
    actual_epochs = len(epoch_losses_g)
    epochs_range = range(1, actual_epochs + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs_range, epoch_losses_g,   marker='o', linestyle='-', label='Total Loss')
    plt.plot(epochs_range, epoch_losses_g1,  marker='o', linestyle='-', label='Path Loss (MSE/NLL)')
    plt.plot(epochs_range, epoch_losses_gx,  marker='o', linestyle='-', label='Total Intent Loss')
    plt.plot(epochs_range, epoch_losses_gx2, marker='o', linestyle='-', label='Lon. Intent Loss')
    plt.plot(epochs_range, epoch_losses_gx3, marker='o', linestyle='-', label='Lat. Intent Loss')
    
    plt.title('Training Losses per Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(epochs_range) # x축 눈금도 맞춤
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # 파일명에 run name 포함하면 구분하기 좋음
    graph_path = os.path.join(args['path'], f'training_losses_{wandb_args.name if wandb_args.name else "final"}.png')
    plt.savefig(graph_path)
    
    print(f"\nAll 5 loss graphs have been saved as {graph_path}")

def save_model(name, gdEncoder, generator, path):
    l_path = args['path']
    if not os.path.exists(l_path):
        os.makedirs(l_path)
    t.save(gdEncoder.state_dict(), l_path + '/epoch' + name + '_gd.tar')
    t.save(generator.state_dict(), l_path + '/epoch' + name + '_g.tar')

if __name__ == '__main__':
    main()
