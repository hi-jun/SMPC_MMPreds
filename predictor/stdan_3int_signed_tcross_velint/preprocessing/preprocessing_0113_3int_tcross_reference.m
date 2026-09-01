%% Process dataset into mat files %%
%% Data curation + data augmentation : augment LC/RC by mirroring RC/LC (Applied to Train/Val/Test)

clear;
clc;

%% Inputs:
us101_1 = './US-101/trajectories-0750am-0805am.txt';
us101_2 = './US-101/trajectories-0805am-0820am.txt';
us101_3 = './US-101/trajectories-0820am-0835am.txt';
i80_1 = './I-80/trajectories-0400-0415.txt';
i80_2 = './I-80/trajectories-0500-0515.txt';
i80_3 = './I-80/trajectories-0515-0530.txt';

%% Load data and add dataset id
disp('Loading data...')
traj{1} = load(us101_1);    
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);
traj{2} = load(us101_2);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(us101_3);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
traj{4} = load(i80_1);    
traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
traj{5} = load(i80_2);
traj{5} = single([5*ones(size(traj{5},1),1),traj{5}]);
traj{6} = load(i80_3);
traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);

for k = 1:6
    traj{k} = traj{k}(:,[1,2,3,6,7,13,14,15,12]);  % ds_id, veh_id, frame_id, local_x, local,y, v_vel, v_acc, lane_id, v_class 
    if k <=3
        traj{k}(traj{k}(:,8)>=6,8) = 6;
    end
end

%% containers를 이용한 코드최적화 
vehTrajs = cell(1,6);
vehTimes = cell(1,6);
for k=1:6
    vehTrajs{k} = containers.Map; % (ds_id, veh_id) : each unique veh_id trajectories
    vehTimes{k} = containers.Map; % (ds_id, frame_id) : info of each frame_id
end


%% Parse fields
disp('Parsing fields...')

for ii = 1:6
    vehIds = unique(traj{ii}(:,2)); 
    for v = 1:length(vehIds)
        vehTrajs{ii}(int2str(vehIds(v))) = traj{ii}(traj{ii}(:,2) == vehIds(v),:); 
    end
    
    timeFrames = unique(traj{ii}(:,3)); % ds_id, traj at time v
    for v = 1:length(timeFrames)
        vehTimes{ii}(int2str(timeFrames(v))) = traj{ii}(traj{ii}(:,3) == timeFrames(v),:);
    end
    
    
    % --- [추가된 부분] 진행률 표시 설정 ---
    num_rows = size(traj{ii}, 1); % 전체 행 개수
    fprintf('Processing Dataset %d/6: ', ii); % 데이터셋 번호 출력
    reverseStr = ''; % 이전에 출력한 문자를 지우기 위한 변수
    % ---------------------------------

    for k = 1:length(traj{ii}(:,1)) % every samples     
        time = traj{ii}(k,3);
        dsId = traj{ii}(k,1);
        vehId = traj{ii}(k,2);
        vehtraj = vehTrajs{ii}(int2str(vehId));
        ind = find(vehtraj(:,3)==time); % current_frame
        ind = ind(1); %
        lane = traj{ii}(k,8);
        
       %% Get lateral maneuver : (논문 기준 2.8초 = 28프레임 적용됨)
        ub = min(size(vehtraj,1),ind+28);
        lb = max(1, ind-28);
        if vehtraj(ub,8)>vehtraj(ind,8) || vehtraj(ind,8)>vehtraj(lb,8)
            traj{ii}(k,10) = 3; % right
        elseif vehtraj(ub,8)<vehtraj(ind,8) || vehtraj(ind,8)<vehtraj(lb,8)
            traj{ii}(k,10) = 2; % left
        else
            traj{ii}(k,10) = 1;
        end
        
       %% Get longitudinal maneuver:
        ub = min(size(vehtraj,1),ind+50); 
        lb = max(1, ind-30); 
        if ub==ind || lb ==ind
            traj{ii}(k,11) =1;
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb); 
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);  
            if vFut/vHist <0.8 
                traj{ii}(k,11) =2;
            elseif vFut/vHist > 1.25 
                traj{ii}(k,11) = 3;
            else
                traj{ii}(k,11) = 1; 
            end
        end

        % Get grid locations:
        t = vehTimes{ii}(int2str(time)); % traj info of time t
        frameEgo = t(t(:,8) == lane,:); % (sample, trajs) of time t
        fprintf('t size: %d %d\n', size(t));
        frameL = t(t(:,8) == lane-1,:);
        frameR = t(t(:,8) == lane+1,:); % 
        if ~isempty(frameL)
            for l = 1:size(frameL,1)
                y = frameL(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 1+round((y+90)/15);
                    traj{ii}(k,11 + gridInd) = frameL(l,2); % veh_id 
                end
            end
        end
        for l = 1:size(frameEgo,1)
            y = frameEgo(l,5)-traj{ii}(k,5);
            if abs(y) <90 && y~=0
                gridInd = 14+round((y+90)/15);
                traj{ii}(k,11+gridInd) = frameEgo(l,2);
            end
        end
        if ~isempty(frameR)
            for l = 1:size(frameR,1)
                y = frameR(l,5)-traj{ii}(k,5);
                if abs(y) <90
                    gridInd = 27+round((y+90)/15);
                    traj{ii}(k,11+gridInd) = frameR(l,2);
                end
            end
        end
        % --- [추가된 부분] 진행률 업데이트 ---
        % 매번 출력하면 느려지므로 1000번마다 한 번씩 업데이트
        if mod(k, 1000) == 0
            percentDone = 100 * k / num_rows;
            msg = sprintf('%.2f%%', percentDone); % 현재 퍼센트 문자열 생성
            fprintf([reverseStr, msg]); % 이전 문자 지우고 새 문자 출력
            reverseStr = repmat('\b', 1, length(msg)); % 지울 문자 개수 계산
        end
        % ---------------------------------
    end
    % --- [추가된 부분] 루프 완료 후 정리 ---
    fprintf([reverseStr, '100.00%%\n']); % 100% 찍고 줄바꿈
    % ---------------------------------
end

save('preprocessed_DB.mat', 'traj', 'vehTrajs', 'vehTimes', '-v7.3');
save('allData_s','traj');

%% Split train, validation, test
load('./allData_s','traj');
disp('Splitting into train, validation and test sets...')
tracks = {};
trajAll_cells = {}; 
cell_idx = 1;

for k = 1:6
    fprintf('Processing Dataset %d...\n', k); 
    vehIds = unique(traj{k}(:, 2));
    
    for l = 1:length(vehIds)
        vehTrack = traj{k}(traj{k}(:, 2)==vehIds(l), :);
        tracks{k,vehIds(l)} = vehTrack(:, 3:11)'; % ds_id, veh_id, frame_id, local_x, local,y, v_vel, v_acc, lane_id, v_class 
        
        filtered = vehTrack(30+1:end-50, :);
        if ~isempty(filtered)
            trajAll_cells{cell_idx, 1} = filtered;
            cell_idx = cell_idx + 1;
        end
    end
end

disp('Concatenating all trajectories...');
trajAll = vertcat(trajAll_cells{:});
clear traj; 

% 최대 Frame id 
global_max_frame = max(trajAll(:, 3));
min_aug_frame = min(trajAll(:, 3));
frame_shift = global_max_frame - min_aug_frame + 100;

trajTr=[]; trajVal=[]; trajTs=[];
for ii = 1:6
    no = trajAll(find(trajAll(:,1)==ii),:);
    len1 = round(length(no)*0.7);
    len2 = round(length(no)*0.8);
    trajTr = [trajTr; no(1:len1,:)];
    trajVal = [trajVal; no(len1+1:len2,:)];
    trajTs = [trajTs; no(len2+1:end,:)];
end

disp('Initial Split Complete.');

disp('Saving default dataset..');

traj = trajTr;
save('TrainSet_default', 'traj', 'tracks'); 

traj = trajVal;
save('ValSet_default', 'traj', 'tracks'); 

traj = trajTs;
save('TestSet_default', 'traj', 'tracks'); 


% 모든 데이터셋을 합쳐서 최대 ID
all_ids = [trajTr(:, 2); trajVal(:, 2); trajTs(:, 2)];
max_veh_id = max(all_ids);
fprintf('max_veh_id : %d', max_veh_id)



% =========================================================================
%% Data validation
% =========================================================================

LABEL_COL = 10; % 예시: 라벨이 있는 열 번호
VAL_LK = 1;     % Lane Keeping 값 (사용자 코드 기준)
VAL_LLC = 2;    % Left Lane Cut-in 값
VAL_RLC = 3;    % Right Lane Cut-in 값

% 1. 각 클래스별 인덱스 추출
idx_LK = find(trajTr(:, LABEL_COL) == VAL_LK);
idx_LLC = find(trajTr(:, LABEL_COL) == VAL_LLC);
idx_RLC = find(trajTr(:, LABEL_COL) == VAL_RLC);

% 2. 데이터 개수 확인
num_LK = length(idx_LK);
num_LLC = length(idx_LLC);
num_RLC = length(idx_RLC);

fprintf('Before balancing: LK=%d, LLC=%d, RLC=%d\n', num_LK, num_LLC, num_RLC);


%% =========================================================================
%% Class Balancing (Undersampling for 2:1:1 Ratio)
%% =========================================================================
disp('Balancing Training Data (2:1:1 ratio for LK, LLC, RLC)...');

% 1. 클래스별 인덱스 추출 (Train Set 기준)
idx_LK  = find(trajTr(:, 10) == 1); % Lane Keeping
idx_LLC = find(trajTr(:, 10) == 2); % Left Lane Change
idx_RLC = find(trajTr(:, 10) == 3); % Right Lane Change

% 2. 현재 데이터 분포 출력
fprintf('  [Before Balance] LK: %d, LLC: %d, RLC: %d\n', length(idx_LK), length(idx_LLC), length(idx_RLC));

% 3. 가장 적은 클래스의 개수 찾기 (Target Count)
min_count = min([length(idx_LK), length(idx_LLC), length(idx_RLC)]);
fprintf('  [Train Set] Target count per class: %d\n', min_count);

% 4. 무작위 샘플링 (Undersampling)
% randperm을 사용하여 각 클래스 인덱스에서 min_count 만큼만 랜덤하게 뽑습니다.
idx_LK_bal  = idx_LK(randperm(length(idx_LK), 2*min_count));
idx_LLC_bal = idx_LLC(randperm(length(idx_LLC), min_count));
idx_RLC_bal = idx_RLC(randperm(length(idx_RLC), min_count));

% 5. 균형 잡힌 데이터셋 재구성
% 뽑은 인덱스들을 합칩니다.
balanced_indices = [idx_LK_bal; idx_LLC_bal; idx_RLC_bal];
trajTr_balanced = trajTr(balanced_indices, :);

% 6. 시간 순서대로 정렬 (선택 사항이나, 데이터 정렬을 위해 추천), 프레임 ID(3번 컬럼) 혹은 데이터셋 ID(1번) 순으로 정렬해두면 보기에 좋습니다.
trajTr_balanced = sortrows(trajTr_balanced, [1, 2, 3]);

disp('Balancing Validation Data...');
idx_LK_val  = find(trajVal(:, 10) == 1);
idx_LLC_val = find(trajVal(:, 10) == 2);
idx_RLC_val = find(trajVal(:, 10) == 3);

% 2. 최소 개수 찾기
min_count_val = min([length(idx_LK_val), length(idx_LLC_val), length(idx_RLC_val)]);
fprintf('  [ Val Set] Target count per class: %d\n', min_count_val);

% 3. 랜덤 샘플링 & 적용
bal_idx_val = [ ...
    idx_LK_val(randperm(length(idx_LK_val), 2*min_count_val)); ...
    idx_LLC_val(randperm(length(idx_LLC_val), min_count_val)); ...
    idx_RLC_val(randperm(length(idx_RLC_val), min_count_val)) ...
];
trajVal_balanced = trajVal(bal_idx_val, :);
trajVal_balanced = sortrows(trajVal_balanced, [1, 2, 3]); % 정렬

disp('Balancing Test Data...');

% 1. 클래스별 인덱스
idx_LK_ts  = find(trajTs(:, 10) == 1);
idx_LLC_ts = find(trajTs(:, 10) == 2);
idx_RLC_ts = find(trajTs(:, 10) == 3);

% 2. 최소 개수 찾기
min_count_ts = min([length(idx_LK_ts), length(idx_LLC_ts), length(idx_RLC_ts)]);
fprintf('  [Test Set] Target count per class: %d\n', min_count_ts);

% 3. 랜덤 샘플링 & 적용
bal_idx_ts = [ ...
    idx_LK_ts(randperm(length(idx_LK_ts), 2*min_count_ts)); ...
    idx_LLC_ts(randperm(length(idx_LLC_ts), min_count_ts)); ...
    idx_RLC_ts(randperm(length(idx_RLC_ts), min_count_ts)) ...
];
trajTs_balanced = trajTs(bal_idx_ts, :);
trajTs_balanced = sortrows(trajTs_balanced, [1, 2, 3]); % 정렬

% =========================================================================
%% Data validation
% =========================================================================
idx_LK = find(trajTr_balanced(:, LABEL_COL) == VAL_LK);
idx_LLC = find(trajTr_balanced(:, LABEL_COL) == VAL_LLC);
idx_RLC = find(trajTr_balanced(:, LABEL_COL) == VAL_RLC);

% 2. 데이터 개수 확인
num_LK = length(idx_LK);
num_LLC = length(idx_LLC);
num_RLC = length(idx_RLC);

fprintf('After balancing: LK=%d, LLC=%d, RLC=%d\n', num_LK, num_LLC, num_RLC);
fprintf('All Sets Balanced. Proceeding to Augmentation...\n');

% =========================================================================
%% DATA AUGMENTATION (Applied to Train, Validation, and Test Sets)
% =========================================================================
disp('Augmenting Data (Applying Mirror World to Train, Val, and Test)...');

LANE_WIDTH = 12; 

% 1. Global Parameters Calculation (Apply consistant shift across all sets)
frame_shift_val = frame_shift; 

% --- C. Tracks Generation for Augmented Vehicles ---
disp('Generating Augmented Tracks...');
[ds_idx, veh_idx] = find(~cellfun(@isempty, tracks));
unique_pairs = [ds_idx, veh_idx];
num_vehicles = size(unique_pairs, 1);

for i = 1:num_vehicles
    u_ds_id = unique_pairs(i, 1);
    u_old_id = unique_pairs(i, 2);
    u_new_id = u_old_id + max_veh_id; 
 
    if u_new_id <= size(tracks, 2) && ~isempty(tracks{u_ds_id, u_new_id})
        continue;
    end

    if isempty(tracks{u_ds_id, u_old_id})
        continue;
    end
    
    orig_track = tracks{u_ds_id, u_old_id};% frame_id, local_x, local,y, v_vel, v_acc, lane_id, v_class 
    new_track = orig_track;
    
    if u_ds_id <= 3
        max_lane = 5; road_width = 5 * LANE_WIDTH;
    else
        max_lane = 6; road_width = 6 * LANE_WIDTH;
    end
    
    % Flip Tracks Info (Time, X, Lane)
    new_track(1, :) = new_track(1, :) + frame_shift_val;
    new_track(2, :) = road_width - new_track(2, :);
    new_track(6, :) = (max_lane + 1) - new_track(6, :);
    
    % Flip Maneuver Label (2 <-> 3) within tracks
    maneuvers = new_track(8, :);
    new_track(8, maneuvers == 2) = 3;
    new_track(8, maneuvers == 3) = 2;
    
    tracks{u_ds_id, u_new_id} = new_track;
end

% 2. Process Loop
sets_to_process = {trajTr_balanced, trajVal_balanced, trajTs_balanced};
set_names = {'TrainSet', 'ValSet', 'TestSet'};
final_sets = cell(1, 3);

for s_idx = 1:3
    current_traj = sets_to_process{s_idx};
    current_name = set_names{s_idx};
    
    if isempty(current_traj)
        fprintf('%s is empty, skipping augmentation.\n', current_name);
        final_sets{s_idx} = current_traj;
        continue;
    end
    
    fprintf('Augmenting %s...\n', current_name);
    
    aug_data = current_traj; % Copy original data
    
    % --- A. Time & ID Shifting ---
    aug_data(:, 3) = aug_data(:, 3) + frame_shift_val; % Time Shift
    aug_data(:, 2) = aug_data(:, 2) + max_veh_id;      % ID Shift
    
    % Grid Neighbor ID Shift
    grid_cols = aug_data(:, 12:50);
    has_neighbor = grid_cols > 0;
    grid_cols(has_neighbor) = grid_cols(has_neighbor) + max_veh_id;
    aug_data(:, 12:50) = grid_cols;
    
    % --- B. Geometric Mirroring ---
    aug_ds_ids = aug_data(:, 1);
    
    % US-101 (Dataset 1-3)
    us101_mask = aug_ds_ids <= 3;
    if any(us101_mask)
        max_lane = 5; road_width = 5 * LANE_WIDTH;
        aug_data(us101_mask, 8) = (max_lane + 1) - aug_data(us101_mask, 8);
        aug_data(us101_mask, 4) = road_width - aug_data(us101_mask, 4);
    end
    
    % I-80 (Dataset 4-6)
    i80_mask = aug_ds_ids >= 4;
    if any(i80_mask)
        max_lane = 6; road_width = 6 * LANE_WIDTH;
        aug_data(i80_mask, 8) = (max_lane + 1) - aug_data(i80_mask, 8);
        aug_data(i80_mask, 4) = road_width - aug_data(i80_mask, 4);
    end
   
    % --- D. Maneuver & Grid Swapping ---
    left_idxs = aug_data(:, 10) == 2;
    right_idxs = aug_data(:, 10) == 3;
    aug_data(left_idxs, 10) = 3; 
    aug_data(right_idxs, 10) = 2; 
    
    left_grid = aug_data(:, 12:24);
    right_grid = aug_data(:, 38:50);
    aug_data(:, 12:24) = right_grid;
    aug_data(:, 38:50) = left_grid;
    
    % --- E. Merge --- : filtering ramp datas
    valid_mask = aug_data(:, 8) > 0 & aug_data(:, 8) <= 6; 
    aug_data = aug_data(valid_mask, :);
    
    final_sets{s_idx} = [current_traj; aug_data];
    fprintf('   -> Original: %d rows, Augmented: %d rows. Total: %d\n', ...
        size(current_traj,1), size(aug_data,1), size(final_sets{s_idx},1));
end

% Assign back to original variables
trajTr_final = final_sets{1};
trajVal_final = final_sets{2};
trajTs_final = final_sets{3};

% =========================================================================
%% Data validation
% =========================================================================

LABEL_COL = 10; % 예시: 라벨이 있는 열 번호
VAL_LK = 1;     % Lane Keeping 값 (사용자 코드 기준)
VAL_LLC = 2;    % Left Lane Cut-in 값
VAL_RLC = 3;    % Right Lane Cut-in 값

% 1. 각 클래스별 인덱스 추출
idx_LK = find(trajTr_final(:, LABEL_COL) == VAL_LK);
idx_LLC = find(trajTr_final(:, LABEL_COL) == VAL_LLC);
idx_RLC = find(trajTr_final(:, LABEL_COL) == VAL_RLC);

% 2. 데이터 개수 확인
num_LK = length(idx_LK);
num_LLC = length(idx_LLC);
num_RLC = length(idx_RLC);

fprintf('After augmentation: LK=%d, LLC=%d, RLC=%d\n', num_LK, num_LLC, num_RLC);

% =========================================================================
disp('Saving mat files...')
%%
traj = trajTr_final;
save('TrainSet', 'traj', 'tracks'); 

traj = trajVal_final;
save('ValSet', 'traj', 'tracks');

traj = trajTs_final;
save('TestSet', 'traj', 'tracks');

disp('Done! All files saved successfully.');
