#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cut-in 논문 표(overleaf/table_cutin.tex)용 런별 지표 추출 + 정책/그룹 집계.

사용: python aggregate_cutin_table.py [sweep_root] [--ego-speeds 14,17] [--groups A,B]

스윕 레이아웃: <root>/<policy>/<group>/<kind>_<NNNN>/{scenario_result.pkl,
resolved_config.json, metrics.json, summary.json}

[정의]
채점 구간(window) 과 시간 기준 t0
  t0 = ego policy_log.steps[0].time_s (= state_trajectory 첫 샘플 시각, 제어 시작),
  창 = [t0, t0 + sweep_params.max_sim_time_s (기본 25 s)] = 메인 루프 전체.
  ``--window-s`` 로 더 짧게 자를 수 있고, ``--window-start-s`` 로 창의 **앞**을
  잘라낼 수 있다 (기본: 자르지 않음). 앞을 자르는 것은 **스폰 정렬 과도**를 빼기
  위해서다 -- ego 는 요구 안전거리 안쪽에서 시작해 첫 9 틱 동안 accel_cmd 를
  -1.00 에서 -3.00 까지 틱당 0.25 씩(= 0.25/0.05 = 5.0, ``jerk_limit`` 에 정확히
  포화) 내린다. 네 정책이 완전히 같은 값이라, 자르지 않으면 **j_max 가 정책이
  아니라 시나리오 초기 조건을 잰다** (컷아웃 64 런 중 42 런에서 창 전체 j_max 의
  최대치가 첫 0.5 s 안에 나고 그중 32 건이 5.00 포화). ``--window-start-s 3``
  이면 컷아웃 j_max 가 이벤트 구간 값과 소수 둘째 자리까지 같아진다
  (SCC 1.34 / LSTM 0.83 / STDAN 0.99 / Proposed 1.00, 2026-09-10).
  **평균 열에는 쓰지 말 것** -- j_avg 는 0.23 -> 0.15, a_avg 는 0.40 -> 0.29 로
  이벤트 값(0.18~0.21 / 0.37~0.38)을 지나쳐 더 내려간다. 이벤트 창은 뒤도 자르는데
  평균은 뒤쪽의 조용한 정상주행이 지배하기 때문이다. 앞을 잘라 이벤트 값을 재현할
  수 있는 것은 **최대 열뿐**이다. 상대시각(t_trigger 등)의 기준 t0 는 잘라도
  제어 시작 그대로다. **컷인 그룹은 17.5 (절대 시각 약 32 s)
  를 쓴다**: 컷인이 끝난 뒤 ego 가 CARLA 경로를 따라 계속 달리며 추적 리드가 옆
  차선 차량으로 넘어가고, 범퍼 간격이 단조 붕괴하는데 ``accel_cmd`` 는 0 근처에
  머문다 (01/cutin_0014, 절대 21.5 s 부터 간격 19.9 -> -6.3 m, delta 135 %).
  그 구간의 충돌은 시나리오와 무관한 주행 아티팩트이지 제어 실패가 아니다
  (사용자 확인, 2026-09-09) -- 이 시나리오가 재려는 것은 컷인 근처뿐이다.
  자르면 SCC aggressive 의 충돌 런이 5 -> 3 이 되는데, 줄어든 2 건이 그것이다.
  delta_max / min_bumper_gap / dt_ant / T_rec 은 창을 잘라도 사실상 불변이고
  (컷인 응답이 32 s 안에 끝난다), j_avg / delta_avg 는 평균이라 뒤쪽의 조용한
  정상주행이 빠지면서 18~29 % 올라간다.
  충돌 판정과 모든 상대시각(t_trigger/t_cross/t_onset/t_pass)도 같은 t0 를 쓴다.
  **`_spawn_settle_log.sim_elapsed_s + _cruise_warmup_log.sim_elapsed_s` 를 t0 로
  쓰면 안 된다.** CARLA 의 time_s 는 서버 경과 시계라 시나리오 스폰 전에 이미 ~5 s
  가 지나 있다 — 저 합(2026-09-04 스윕에서 9.90 s)은 실제 제어 시작(첫 스텝 14.4~
  15.2 s)보다 4.4~5.3 s 이르다. 그걸 창의 시작으로 쓰면 모든 런에서 뒤쪽 4~5 s
  (회복 구간)가 잘리고 창 내용이 런마다 달라진다. 옛 aggregate_confchance.py 와
  RESULT.md 의 상대시각이 이 방식이라 이 스크립트 값과 4.4~5.3 s 어긋난다.

승차감 (창의 첫 샘플 1개 버림 — 첫 차분이 스폰 과도를 포함한다)
  **표에 쓰는 저크는 대역 제한 명령 저크**: j_avg_cmd_filt / j_max_cmd_filt =
  accel_cmd 를 실측과 같은 0.2 s 대역으로 누른 뒤의 Δ/Δt.  명령은 20 Hz 로
  갱신되고 예측이 틱마다 흔들려 정상 추종 중에도 ±0.5 m/s^2 의 고주파가 남는데,
  차량은 이를 따라가지 못하지만 원시 j_*_cmd 는 그대로 세어 예측 기반 정책만
  불리해진다(원시 j_max_cmd 는 4 정책 중 3 개가 한계 10 에 포화한다).
  원시값 j_avg_cmd / j_max_cmd 와 이벤트 구간값 j_*_cmd_ev 는 CSV 에 남는다.
  제어기가 실제로 낸 양이고 ``jerk_limit`` 이 직접 묶는 값이라 정책 간 비교가
  된다. a_avg 는 측정 가속도의 절대값 평균(mean|actual_accel|) — 절대값 평균이라
  잡음에 둔감하다(한 런에서 0.975 vs 명령 0.824). a_avg_cmd / a_min_cmd 도 함께 낸다.
  ※ **원시 측정 저크(j_avg/j_max)는 액추에이터 잡음이 지배해 표에 쓸 수 없다.**
  이상적 액추에이터의 1틱 deadbeat 보정 때문에 측정 가속도가 명령 주위에서 틱마다
  진동한다 — 세게 밟는 구간(02_cutin_aggressive/aggressive_cutin_0017, chance0.6,
  ego 17)에서 명령이 -2.19 → -1.21 로 매끈한데 측정값은 -3.98, -0.31, -3.89, +0.44,
  -2.20, +0.25, -12.76, +2.52 로 튄다(|actual_accel-accel_cmd| > 1.0 인 스텝이 그 런의
  17%). 그대로 쓰면 j_max 가 40~300 m/s^3, a_min 이 -12.8 m/s^2 (한계 -3.0) 로 나온다.
  참고용으로 **제어 주기(0.2 s) 대역** 값도 계산한다:
    a_filt[k] = 4틱(0.2 s) 이동평균 = Δv/0.2s,  j_filt[k] = (a_filt[k+4]-a_filt[k])/0.2
  → a_min_filt / j_avg_filt / j_max_filt (마크다운 표의 괄호 값). 원시 j_avg / j_max /
  j_p99 / a_min 도 CSV·요약에 남는다.
  ※ jerk_limit 은 2026-09-05 부터 10 m/s^3 (그 전 스윕은 1.5 라 j_max_cmd 가 전 정책
  1.50 으로 포화해 변별력이 없다).

안전 (모든 제어기에 대해 완화 없는 공칭 안전거리)
  d_safe(t) = d0 + tau*v_ego(t) = 3.0 + 1.3*v  (범퍼-투-범퍼)
  g(t) = (s_lead - s_ego) - L,  L = 4.5      (범퍼 간격)
  delta(t) = max(0, (d_safe - g)/d_safe) * 100 [%],  차선 내 선행차 없으면 0.
  delta_max = 창 내 최대,  delta_avg = **선행차가 존재한 스텝만**의 평균,
  delta_avg_window = 창 전체 평균(선행차 없는 스텝은 0으로).

차선 내 선행차 = s_actor > s_ego 인 차 중 가장 가까운 차, 단
  (a) lane_trajectory 의 lane_id 가 그 시각 ego 와 같고
  (b) 횡방향 |x_actor - x_ego| <= 1.75 m (차선 반폭)
  road_id 일치는 **요구하지 않는다**. 근거(실측): 이 맵에서 lane_id 는 road 간
  일관되지 않아 lane_id+road_id 만으로는 |dx| 13.8 m 짜리 차도 "같은 차선"으로
  잡히고(전체 lane 매칭 8936 샘플 중 |dx|>3.5 m 가 약 1300), 반대로 road_id 를
  요구하면 앞차가 다음 road 세그먼트로 넘어간 순간이 통째로 빠진다 —
  01_cutin_normal/cutin_0022(base, ego14) 에서 범퍼 간격 0.29 m 의 최근접
  순간이 정확히 그렇게 누락됐다. lane_id(=CARLA 가 차선 중심 경계 1.75 m 에서
  스냅한 값) + 횡거리 조합이 두 오류를 모두 없앤다.

세로방향 좌표 s: ego 스텝의 `ego_s`(Frenet)만 로그에 있고 다른 차는 x,y 뿐이다.
  이 Town04 구간은 도로가 -y 방향이라 s_actor(t) = ego_s(t) + (y_ego(t) - y_actor(t)).
  즉 매 시각 ego 자신의 s 를 원점으로 쓰고 -y 차이만 쓴다(전역 절편을 안 잡으므로
  ego 근방 수십 m 에서만 직선이면 된다).
  검증 1 (전역): a = mean(ego_s + y_ego) 로 s = a - y 를 맞추면 잔차 rms 0.144 m
  = 0.5/sqrt(12) — ego_s 가 0.5 m 격자에 스냅된 양자화 오차 그 자체이고, 유효
  런의 잔차 최대는 0.25~0.64 m 다. 즉 창 안에서 도로는 사실상 직선이다.
  (ego 20 m/s 무효 런처럼 500 m 를 가면 곡선에 들어가 잔차가 4~5 m 로 커진다.)
  검증 2 (국소, 실제 사용하는 가정): 3 s 간격 두 시점의 Δego_s 와 Δ(-y) 차이가
  s_map_err_window — 45~50 m 떨어진 차를 -y 로 잴 때 생기는 오차 그 자체다.
  무효 판정에 쓰는 s_map_err 는 **차선 내 선행차가 존재한 스텝**(다른 차의 s 를
  실제로 쓰는 유일한 구간)에서의 최대값이고, S_MAP_TOL(1.0 m)을 넘으면 무효다.
  오차는 종방향 이격에 비례하므로 이격 0(추월 순간)에서는 0 이다. 창 전체로 재면
  ego 17 m/s 가 400 m 를 달려 마지막 3 s 에 곡선(road 144)에 들어가는 런들이
  1.9~3.3 m 로 걸리는데, 정작 선행차가 있던 구간(t+6.7 s 이전)의 오차는 0.42 m 다
  (baselines_tvspeed_20260905 aggressive_cutin_0010~0012 실측).

cut-in 응답
  t_cross = cut-in TV(`target_cutin_*`)가 위 차선 내 판정을 처음 만족한 시각.
           road_id 를 안 따지므로 RESULT.md 의 "경계 통과"(lane_id+road_id)보다
           0.2~0.7 s 이르다 — t_cross-t_trigger 가 1.65~1.80 s 로 나오고
           RESULT.md 표의 1.85~2.50 s 와 그만큼 어긋난다. Δt_ant 에는 보수적이다.
  t_trigger = TV policy_log.trigger_time_s (차선변경 트리거; 03 그룹은 None).
  t_onset = 창 안에서 accel_cmd <= -0.3 m/s^2 가 2스텝 연속인 첫 시각.
  (2026-09-08 부터 cut-in 의 Δt_ant 도 cut-out 과 같은 **예측 선제성**이다:
   TV 의 모드 중 지평선 끝 3 스텝이 ego 차선 안인 것들의 확률합이 0.5 이상인
   상태가 1.0 s 이상 이어지는 첫 시점 t_pred 부터 실제 진입 t_cross 까지.
   아래 제어 개시 기준은 dt_ant_ctrl 열로 남는다.)
  dt_ant_ctrl = t_cross - t_onset  (t_onset < t_cross 일 때만; 아니면 None.
           차가 이미 들어온 뒤에야 반응한 제어기는 선제성이 없다).
           ※ 이 스윕에서 감속 개시가 창 시작 0.3~0.6 s 뒤에 몰린다(예측기가 조향
           개시 2.3 s 전에 cut-in 을 검출하는데 그 시점이 메인 루프 시작 근처다).
           즉 Δt_ant 는 **좌측 절단**될 수 있다 — t_onset 이 0.5 s 미만인 런은
           "더 일찍 밟았을 수도 있는데 창이 그때 시작했다"는 뜻이다.
  T_rec = t_cross 이후 delta(t) == 0 이 0.5 s 이상 연속으로 유지되는 첫 시각
          - t_cross.  t_cross 시점에 이미 delta == 0 이면 0.0.
          창 안에서 회복 못하면 None + T_rec_censored = 창끝 - t_cross.

03_no_cutin_decel (TV가 인접 차선에서 감속만 함) 추가 지표
  v_avg = 창 내 ego_v 평균, v_min, a_min, passed = 창 끝에서 s_ego > s_TV,
  t_pass = s_ego > s_TV 가 되는 첫 시각. delta 계열은 실제로 ego 차선에 차가
  있을 때만 값이 생긴다(없으면 delta 0, has_inlane_lead=False).

유효성
  invalid = pkl/json 로드 실패 | ran_successfully False
            | fallback_ratio > FALLBACK_TOL (0.02) | s_map_err > 1.0 m.
  fallback_ratio = solve_path == "fallback" 인 스텝 비율. v_max 교착(ego 20 m/s)
  은 0.17~1.0 이라 제어기 지표 전부 무의미하다. 충돌 직후 1~2 스텝(≤0.4 %)
  의 fallback 은 충격으로 튄 속도 때문이라 런을 버리지 않는다.
  slack_steps = feasible False 스텝 수(진단용, 무효 사유 아님).
  충돌은 **무효 사유가 아니다** — 제어기의 결과이므로 빼면 표가 유리해진다.
  대신 collisions_in_window / ego_collision 로 표시하고 METRICS.md 에 나열한다.
  collisions_in_window = 창(+0.25 s) 안의 서로 다른 (역할,상대역할) 쌍 수.

시나리오 건전성 (제외 사유 아님)
  tv_lead_contact = [t_trigger-0.5 s, t_cross+2.0 s] (t_cross 없으면 t_trigger+3 s)
  안에 target_cutin ↔ traffic_cutin_lane_lead 충돌이 기록된 런. T = 3.67 s 에서
  cut-in 차량이 옛 차선 앞차를 횡방향 여유 ~0 m 로 스치고 지나가서, cm 단위 차이로
  CARLA 가 접촉을 잡느냐 마느냐가 갈린다(SCC cutin_0012/0015: t_cross 0.25 s 뒤
  접촉, ego 가 TV 에 닿기 전). 셀별 개수를 요약에 같이 낸다.

[cut-out 정의]
그룹 이름에 `cutout` 이 들어간 셀에만 적용된다(표: overleaf/table_cutout.tex,
열 순서 a_avg, j_avg, j_max, Δt_ant, v_avg). 창/t0/승차감/안전거리(delta)/유효성
정의는 cut-in 과 **완전히 같다** — 아래는 달라지는 것만 적는다.

역할
  LV    = `target_cutout_*`           : ego 차선 앞을 달리다 옆 차선으로 빠지는 차.
  subLV = `target_lead_after_cutout_*`: LV 뒤에 가려져 있던 느린 앞차(ego 차선).
  kind `cutout_no_sublv` = subLV 없음(시간 트리거), `cutout_sublv` = subLV 있음
  (lead-gap 트리거). 어느 쪽인지는 **그룹/런 디렉터리 이름**으로 정한다
  ("no_sublv" 가 들어가면 subLV 없음). 실제 액터 유무는 has_sublv_actor 로 남긴다.

cut-out 시각
  t_trigger = LV policy_log.trigger_time_s (t0 기준 상대). trigger_mode /
    trigger_distance_at_start / cutout_started / cutout_completed 도 같이 옮긴다.
  t_out = t_trigger 이후 LV 의 lane_id 가 ego 의 lane_id 와 **2스텝(0.1 s) 연속**
    다른 첫 시각 = LV 중심이 ego 차선을 벗어난 순간. 2스텝 hold 는 도로 경계에서
    lane_id 가 한 샘플 튀는 것을 막는 장치다(실측: 같은 차선·종방향 이격 20~80 m
    46k 샘플에서 lane_id 불일치 0건, 이격 0~20 m 에서만 0.7~0.8 % — cut-out LV 는
    20~35 m 앞이라 안전한 범위다). t_trigger 가 없으면(트리거 미발동) t_out=None.
  t_out_minus_trigger = t_out - t_trigger (LV 가 차선을 비우는 데 걸린 시간).

트리거 직전 정상상태 (진단값, 무효 사유 아님)
  settled_before_trigger = [t_trigger-3 s, t_trigger] 안에 |accel_cmd| < 0.15 m/s^2
    가 1.0 s 이상 연속인 구간이 하나라도 있으면 1 (ACC 가 LV 뒤에서 안정 추종 중).
    셀 집계값은 그 조건을 만족한 런의 비율이다.
    ※ 이 조건은 "정속 순항 중"과 "LV 뒤 정상 추종 중"을 구분하지 못한다. LV 를 아예
    안 보고 자유주행하던 런도 1 이 되니 gap_at_trigger 와 같이 읽어야 한다.
  lv_inlane_at_trigger = t_trigger 시각에 LV 의 lane_id 가 ego 와 같은가.
    **False 면 그 런은 cut-out 이 아니다** — 트리거가 걸릴 때 LV 가 이미 ego 차선
    밖이라 t_out 이 t_trigger 와 같아지고(t_out_minus_trigger=0, Δt_ant=0.00)
    지표가 나오긴 하는데 잰 대상이 없다. METRICS.md 5.3 에 따로 나열한다.
  gap_at_trigger = t_trigger 시각의 ego-LV 범퍼 간격(중심거리 - 4.5 m),
  v_ego_at_trigger / v_lv_at_trigger = 같은 시각 두 차의 속도.

Δt_ant (**예측 선제성**; 2026-09-08 부터 제어 반응이 아니라 예측 시점이다)
  t_pred = LV 가 ego 차선을 벗어난다고 예측기가 처음 말한 순간. 틱마다 LV 의
    모드별 예측 궤적 중 지평선 안에서 차선을 벗어나는(끝 3 스텝이 밖) 모드들의
    확률을 합치고, 그 합이 0.5 이상인 상태가 [t_trigger − 3 s, 끝] 안에서 1.0 s
    이상 연속되는 첫 시점을 쓴다. LV 는 그 틱의 실제 s 에 가장 가까운
    processed_target 으로 찾는다(summary 에 actor–CARLA id 대응이 없다).
    STDAN 계열은 cutout 모드가, LSTM 은 단일 모드가 이 조건을 만든다. 예측기가
    없는 SCC 는 항상 None → 표에 '-'.
  Δt_ant = max(0, t_out − t_pred); 부호 있는 값은 dt_ant_signed.
    cutout_call_max 는 그 런에서 관측된 확률합의 최대(왜 안 잡혔는지 진단용).
  ※ 왜 제어 시점이 아닌가: subLV 가 있는 장면에서 빨리 감속하는 것이 무조건
    좋은 것이 아니다. 벌어진 간격을 다시 채워 앞차 간격을 유지하면서 subLV 까지
    절충하는 것이 목적이므로, 선제성은 예측기가 얼마나 일찍 알았는지로 재고
    그 결과의 좋고 나쁨은 gap_err/저크/최소간격이 따로 잰다.

dt_ant_ctrl (옛 Δt_ant. 제어기 반응 개시 선제성, 진단용으로만 남긴다)
  t_onset = [t_trigger − 3 s, t_out + 5 s] 창 안에서, 부호 맞는 가속이 **1.0 s 이상
    연속으로** 임계를 넘는 첫 구간의 시작:
    accel_cmd >= +0.3 m/s^2 (kind cutout_no_sublv: 비워진 차선으로 가속) 또는
    accel_cmd <= -0.3 m/s^2 (kind cutout_sublv: subLV 때문에 감속).
    ※ 정착-구간 기준(옛 정의)은 STDAN 계열의 잔류 진동(정착 후에도 ±0.3~0.6 m/s^2,
    ego 17/LV 11 은 12 s 에도 ±1) 을 개시로 오인해 Δt_ant 가 6~8 s 로 부풀었다.
    1.0 s hold 는 진동(0.2 s 안팎)과 트리거 전의 약한 예비 제동(-0.25)을 걸러내고
    본격 반응 램프만 잡는다. 쓴 부호는 onset_sign 열에 남긴다. t_settle_end /
    settled_before_trigger 는 진단용으로만 남긴다.
  dt_ant_ctrl = max(0, t_out − t_onset); 부호 있는 값은 dt_ant_ctrl_signed 열
    (음수 = t_out 이후에야 반응). 창 안에 onset 이 없으면 None.
  ※ 정상상태를 기준점으로 잡는 이유: cut-out 은 트리거 전이 정속 추종 구간이라
    창 시작 직후의 스폰 과도를 onset 으로 잘못 집기 쉽다.

속도
  v_avg = **[t_trigger, 창끝] 의 ego 속도 평균** — table_cutout.tex 의 v_avg 열.
    (cut-in 쪽 v_avg 는 창 전체 평균이다. 같은 이름이지만 구간이 다르다.)
  v_avg_window = 창 전체 평균, v_avg_post = [t_out, 창끝] 평균.

LV 와의 간격이 얼마나 어긋나는가
  gap_err = |LV 범퍼간격 − d_safe| / d_safe * 100 [%]. **부족과 초과를 모두**
    벌한다 — 선제 감속이 과해 뒤처져도, 앞차에 바싹 붙어도 커진다. delta 는
    부족한 쪽만 재므로 짝으로 읽는다.
  구간은 [t_trigger − 1 s, t_out] 중 LV 가 아직 ego 차선에서 앞서는 스텝.
    LV 가 빠진 뒤의 subLV 간격은 시나리오 기하(앞차를 89 m 앞에 둔다)가 지배해
    제어 품질을 재지 못하므로 제외한다.
  gap_err_max / gap_err_avg = 그 구간의 최대/평균. table_cutout.tex 에는 max.
  gap_err_signed_avg / gap_err_short_max / gap_err_excess_max = 절대값을 씌우기
    전의 분해. signed = (범퍼간격 - d_safe)/d_safe*100 이고 음수가 안전거리 안,
    양수가 뒤처진 쪽이다. short_max = max(-signed, 0) 의 최대(안으로 얼마나
    들어갔나 = 안전), excess_max = max(signed, 0) 의 최대(얼마나 뒤처졌나 =
    선제 감속의 대가). gap_err_max = max(short_max, excess_max) 라 정확히
    쪼개진다. **선제 감속은 excess 만 키우므로, 두 방향을 합쳐 재는
    gap_err_max 는 일찍 감속한 정책과 늦게 감속한 정책을 같은 방향으로
    벌한다** — 정책을 가릴 때는 분해한 쪽을 보는 편이 낫다.

이벤트 구간 표 (table_event_rows.tex / 마크다운 "이벤트 구간:")
  창은 [t_trigger − 5 s, t_trigger + 9 s] (선제 반응 + 기동 + 회복).
  트리거가 없는 그룹(03/04)은 창 전체 값이 그대로 들어간다.
  **2026-09-10 에 앞을 −1 s 에서 −5 s 로 넓혔다.** −1 s 는 예측 정책이 실제로
  먼저 움직인 구간을 잘라냈다 — 컷인 Δt_ant 가 2.1 s, 컷아웃 제동 개시가 t_out
  대비 −4.2 s 라 재려던 기동의 앞부분이 창 밖이었다. 대신 정속주행이 4 s 더
  섞이므로 **평균 열(a_avg_ev / j_avg_cmd_ev / delta_avg_ev)은 낮아진다** —
  −1 s 로 잰 이전 표와 직접 비교하지 말 것.
  cut-out 의 gap_err_* 도 같은 상수를 쓴다. 넓혀도 **표에 쓰는 gap_err_max /
  gap_err_excess_max 는 사실상 안 변한다** — 최댓값은 언제나 트리거 이후 기동에서
  나오고, 4 정책 3 그룹 모두 순위가 같다(06/07 은 소수 둘째 자리까지 동일, SCC 만
  +0.2~0.5). 대신 **평균 gap_err_signed_avg 는 절반으로 줄고 순위가 뒤집힌다** —
  오차 0 인 정속주행이 평균에 섞이는데, 선제 감속으로 트리거 뒤에 오차를 많이 낸
  정책일수록 더 많이 깎인다(STDAN 06: 14.69 → 6.13, SCC 는 6.06 → 6.10 로 거의
  그대로). −1 s 에서는 SCC < Proposed < STDAN, −5 s 에서는 Proposed < SCC ≈ STDAN
  이다. 평균 열을 쓸 거면 창을 의식하고 고를 것.
  a_avg_ev  = 그 창의 |accel_cmd| 평균,  a_min_ev = 최솟값(최대 제동).
  j_avg_cmd_ev_filt / j_max_cmd_ev_filt = 그 창의 0.2 s 대역 명령 저크
    평균/최대. 표의 j_*_cmd_filt 은 같은 식이되 **런 전체** 평균이라 기동이
    끝난 뒤 정상 추종 구간이 지배한다. 기동만 보려면 이 열을 쓴다.
  delta_max_ev / delta_avg_ev / min_bumper_gap_ev / v_avg_ev / v_min_ev =
    같은 창의 안전·속도 지표. delta_max 는 컷인 응답 안에서 나므로 창 전체판과
    거의 같지만, delta_avg 는 창 전체판이 회복 후 0 인 스텝에 눌려 있어 다르다.
    collisions_in_event / ego_collision_ev 는 같은 창의 충돌만 센다 — 컷인과
    무관한 뒤쪽 경로주행 충돌이 빠진다.
  T_rec 과 gap_err_* 는 이미 t_cross / t_out 기준이라 이벤트판이 따로 없다.

subLV (kind cutout_sublv 만)
  min_bumper_gap_sublv = subLV 가 ego 차선에 있고 앞설 때의 최소 범퍼 간격.
  gap_recover_avg = t_out 이후 subLV 를 상대로 남은 **초과 여유의 시간평균** [m]
    = mean(max(범퍼간격 - d_safe, 0)).  gap_err_excess_max 는 [t_trigger-5 s,
    t_out] 만 보고 최댓값 하나를 내므로 "얼마나 뒤처졌나"는 재도 "얼마나 빨리
    되감았나"는 못 잰다.  이 열은 이탈 **이후**를 보고, 크기와 지속을 함께 세며,
    수렴하지 못한 런도 검열 없이 값을 낸다.  낮을수록 빨리 회수했다.
  t_gap_recover = 그 초과가 요구 안전거리의 10 % 이내로 들어와 **그 뒤로 계속
    유지되는** 첫 시각 [t_out 이후 s].  창 끝까지 못 들면 None 이고
    gap_recover_censored = True -- 정책 간 비교에 쓸 때는 검열된 런 수를 함께
    적을 것.  gap_recover_span_s 는 그 창의 길이다.
  delta_max / delta_avg 는 cut-in 과 같은 "차선 내 최근접 선행차" 로직을 그대로
    쓴다 — LV 가 빠지면 자동으로 subLV 가 잡힌다(sublv_inlane_steps 로 확인).
  lv_sublv_contact = [t_trigger-0.5 s, t_out+2 s] 안의 target_cutout ↔
    target_lead_after_cutout 충돌(시나리오 건전성, 제외 사유 아님). 요약 CSV 는
    cut-in 의 TV-앞차 접촉과 같은 열(tv_lead_contact_runs)에 이 개수를 낸다
    (런 단위 CSV 에서는 lv_sublv_contact 열로 따로 남는다).
  ego_collision / collisions_in_window 는 cut-in 과 동일.
"""
import argparse
import collections
import csv
import json
import pathlib
import pickle
import re

import numpy as np

D0 = 3.0
TAU = 1.3
VEH_LEN = 4.5
LANE_HALF_WIDTH = 1.75
CTRL_DT = 0.2
ONSET_ACCEL = -0.3
ONSET_HOLD_STEPS = 2
REC_HOLD_S = 0.5
DEFAULT_WINDOW_S = 25.0
COLLISION_PAD_S = 0.25
S_MAP_LAG_S = 3.0
S_MAP_TOL = 1.0
# 충돌 직후의 1~2 fallback 스텝(500 스텝 중)은 허용; v_max 교착은 0.17 이상.
FALLBACK_TOL = 0.02
# cut-out
CUTOUT_ONSET_ACCEL = 0.3      # 부호는 kind 에 따라 (+: 가속 개시, -: 감속 개시)
CUTOUT_ONSET_HOLD_S = 1.0     # 임계를 이만큼 연속으로 넘어야 반응 개시 (정착 진동 배제)
CUTOUT_ONSET_PRE_S = 3.0      # 개시 탐색 창: 트리거 이전
CUTOUT_ONSET_POST_S = 5.0     # 개시 탐색 창: t_out 이후
CUTOUT_OUT_HOLD_STEPS = 2     # lane_id 한 샘플 튐 방지 (0.1 s)
SETTLE_ACCEL = 0.15
SETTLE_MIN_S = 1.0
SETTLE_LOOKBACK_S = 3.0
EVENT_PRE_S = -5.0    # 이벤트 구간: 트리거 5 s 전부터 (2026-09-10, -1.0 에서 넓혔다).
                      # -1 s 는 예측 정책의 선제 감속을 창 밖으로 잘라냈다 — 컷인
                      # Δt_ant 가 2.1 s, 컷아웃 제동 개시가 t_out 대비 -4.2 s 라
                      # 정작 재려던 기동의 앞부분이 빠졌다. 대신 정속주행 4 s 가
                      # 섞여 들어와 평균 열(j_avg_ev / delta_avg_ev)은 낮아진다.
EVENT_POST_S = 9.0    # 트리거 9 s 후까지 (기동 + 회복)

CUTOUT_CONTACT_PRE_S = 0.5
CUTOUT_CONTACT_POST_S = 2.0
CUTOUT_PRED_PROB = 0.5        # 예측 개시: 차선을 벗어나는 모드들의 확률합 임계
CUTOUT_PRED_HOLD_S = 1.0      # 임계를 이만큼 연속으로 넘어야 예측 개시 (예측 떨림 배제)
CUTOUT_PRED_SUSTAINED_STEPS = 3   # 궤적이 지평선 끝에서 이만큼 나가 있어야 "이탈"
CUTOUT_PRED_MATCH_M = 3.0     # processed_target 을 LV 로 인정할 s 오차 한계

SWEEP_PARAMS = ("ego_speed", "target_speed", "lane_change_distance",
                "lane_change_time_s", "target_lead_gap", "trigger_distance",
                "ego_gap_at_trigger")
TEX_METRICS = ("a_avg", "j_avg_cmd_filt", "j_max_cmd_filt", "delta_max", "delta_avg",
               "dt_ant", "T_rec")
TEX_METRICS_03 = ("v_avg", "a_min_filt", "passed", "t_pass")
# gap_err_max 는 절댓값이라 부호 두 방향이 한 열에 섞인다: 뒤처져도 붙어도 같은
# 크기로 찍혀, 안전거리를 일부러 완화하는 정책과 그냥 못 따라가는 정책이 구분되지
# 않는다(2026-09-10, 05: SCC 7.17 은 전부 뒤처짐, Proposed 7.55 는 전부 파고듦).
# 표는 분해한 두 열을 쓴다.
TEX_METRICS_CUTOUT = ("a_avg", "j_avg_cmd_filt", "j_max_cmd_filt", "dt_ant",
                      "gap_err_short_max", "gap_err_excess_max", "v_avg")
# 이벤트 구간 [t_trigger-5 s, t_trigger+9 s] 만 자른 표
# 이벤트 구간 표는 전체구간 표와 **같은 열**을 쓴다. 여기 없는 열(dt_ant,
# t_out_minus_trigger, gap_at_trigger, settled_before_trigger, gap_err_max)은
# 이미 이벤트 시각 기준이라 창을 바꿔도 같은 값이다.
EV_OF = {"a_avg": "a_avg_ev",
         "j_avg_cmd_filt": "j_avg_cmd_ev_filt", "j_max_cmd_filt": "j_max_cmd_ev_filt",
         "j_avg_filt": "j_avg_filt_ev", "j_max_filt": "j_max_filt_ev",
         "delta_max": "delta_max_ev", "delta_avg": "delta_avg_ev",
         "T_rec": "T_rec_ev", "v_avg": "v_avg_ev", "dt_ant": "dt_ant_ev",
         "min_bumper_gap_sublv": "min_bumper_gap_sublv_ev"}
# 표 값(명령 저크) 옆 괄호에 함께 보일 실측(0.2 s 대역) 값
RAW_OF_FILT = {"j_avg_cmd_filt": "j_avg_filt", "j_max_cmd_filt": "j_max_filt"}
AGG_METRICS = ("a_avg", "a_avg_cmd", "a_min", "a_min_filt", "a_min_cmd",
               "j_avg", "j_avg_filt", "j_avg_cmd", "j_avg_cmd_ev", "j_avg_cmd_filt", "j_max",
               "j_max_filt", "j_max_cmd", "j_max_cmd_ev", "j_max_cmd_filt", "j_p99", "delta_max", "delta_avg", "delta_avg_window",
               "min_bumper_gap", "t_cross_minus_trigger", "dt_ant", "dt_ant_ev", "dt_ant_ctrl", "T_rec",
               "v_avg", "v_min", "passed", "t_pass",
               "a_avg_ev", "a_min_ev", "j_avg_cmd_ev_filt", "j_max_cmd_ev_filt",
               "delta_max_ev", "delta_avg_ev", "min_bumper_gap_ev", "v_avg_ev", "v_min_ev",
               "j_avg_filt_ev", "j_max_filt_ev", "T_rec_ev",
               "j_at_limit", "j_at_limit_ev")
AGG_METRICS_CUTOUT = AGG_METRICS + (
    "gap_err_signed_avg", "gap_err_short_max", "gap_err_excess_max",
    "t_out_minus_trigger", "v_avg_window", "v_avg_post", "gap_at_trigger",
    "v_ego_at_trigger", "v_lv_at_trigger", "min_bumper_gap_sublv",
    "settled_before_trigger", "t_pred_minus_trigger",
    "gap_err_max", "gap_err_avg", "min_bumper_gap_sublv_ev",
    "gap_recover_avg", "t_gap_recover", "gap_recover_span_s")
SCENARIO_LABEL = {"01_cutin_normal": "Normal \\\\ Cut-in",
                  "02_cutin_aggressive": "Aggressive \\\\ Cut-in",
                  "03_no_cutin_decel": "No Cut-in \\\\ (adj. decel)",
                  "04_no_cutin_decel_onset": "No Cut-in \\\\ (decel onset 22/28/34)"}


def _predicts_lane_entry(mask, sustained_steps=CUTOUT_PRED_SUSTAINED_STEPS):
    """이 예측 궤적이 지평선 끝에서 ego 차선 **안**에 있는가 (cut-in 쪽 거울상)."""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size < sustained_steps:
        return False
    return bool(mask[-int(sustained_steps):].all())


def _predicts_lane_exit(mask, sustained_steps=CUTOUT_PRED_SUSTAINED_STEPS):
    """이 예측 궤적이 지평선 끝에서 ego 차선 밖에 있는가.

    predictor 의 ``ends_outside_ego_lane`` 과 같은 3 스텝 규칙이다. 끝 3 스텝만
    보므로 "곧 벗어난다"와 "이미 벗어났다"를 모두 호출로 센다 — LV 가 이미
    완전히 빠져 점유 마스크가 전부 0 이 되는 구간에서도 예측 호출이 t_out 직전
    1.0 s 를 채운다(LSTM, 2026-09-08). 차선 가장자리에서 한두 스텝 삐져나오는
    흔들림은 여전히 걸러진다.
    """
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size < sustained_steps:
        return False
    return not mask[-int(sustained_steps):].any()


def _call_probability(target, predicate):
    """이 틱에 예측기가 이 차량에 얹은 '차선을 벗어난다/들어온다' 확률.

    모드별 차선 점유(mode_lane_membership)와 모드 확률(raw_mode_prob)을 키로
    맞춰 곱한 합. LSTM 은 점유 키가 'LK', 확률 키가 'lstm' 이라 키가 어긋나므로
    개수가 같으면 순서로 짝짓는다. 짝지을 수 없으면 None.
    """
    membership = target.get("mode_lane_membership") or {}
    probs = target.get("raw_mode_prob") or {}
    if not membership or not probs:
        return None
    if set(membership) <= set(probs):
        pairs = [(probs[k], membership[k]) for k in membership]
    elif len(membership) == len(probs):
        pairs = list(zip(list(probs.values()), list(membership.values())))
    else:
        return None
    return float(sum(p for p, mask in pairs if predicate(mask)))


def _target_call(steps, s_target, predicate):
    """틱마다 그 차량에 대한 예측 호출 확률. 예측기가 없으면(SCC) 전부 NaN.

    summary 에는 actor 와 CARLA id 의 대응이 없으므로, 그 틱의 실제 s 에 가장
    가까운 processed_target 을 그 차량으로 본다.
    """
    out = np.full(len(steps), np.nan)
    for i, step in enumerate(steps):
        targets = (step.get("stdan_debug") or {}).get("processed_targets")
        if not targets or not np.isfinite(s_target[i]):
            continue
        best, best_err = None, np.inf
        for target in targets:
            frenet = target.get("pred_traj_frenet") or {}
            if not frenet:
                continue
            s0 = float(np.asarray(next(iter(frenet.values())), dtype=float)[0][0])
            err = abs(s0 - s_target[i])
            if err < best_err:
                best, best_err = target, err
        if best is None or best_err > CUTOUT_PRED_MATCH_M:
            continue
        call = _call_probability(best, predicate)
        if call is not None:
            out[i] = call
    return out


def _prediction_onset(t, call, t_from, dt):
    """확률합이 임계를 1.0 s 이상 연속으로 넘는 첫 시각 (절대 시각). 없으면 None."""
    called = np.isfinite(call) & (call >= CUTOUT_PRED_PROB)
    if t_from is None or not called.any():
        return None
    idx = np.nonzero(t >= t_from)[0]
    if not idx.size:
        return None
    hold = max(1, int(round(CUTOUT_PRED_HOLD_S / dt)))
    i_pred = _first_run_start(called[idx[0]:], hold)
    return None if i_pred is None else float(t[idx[0] + i_pred])


def is_cutout_group(group):
    return "cutout" in group


def cutout_label(group):
    return ("Cut-out \\\\ w/o Second-LV" if "no_sublv" in group
            else "Cut-out \\\\ with Second-LV")


def policy_label(policy):
    if "const_lead" in policy:
        return "SCC"
    if "lstm" in policy:
        return "SCC + LSTM"
    if "cutin_chance" in policy:
        m = re.search(r"cutin_chance([0-9.]+)", policy)
        return "Proposed (chance $\\beta_{ref}$=%s)" % (m.group(1) if m else "?")
    if "stdan_3int" in policy:
        return "STDAN ACC (no chance)"
    return policy


def _finite(values):
    out = []
    for v in values:
        if isinstance(v, (int, float)) and np.isfinite(v):
            out.append(float(v))
    return out


def _nearest(times, query):
    """times 에서 query 각 원소에 가장 가까운 인덱스."""
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = np.abs(query - times[idx - 1]) <= np.abs(times[idx] - query)
    return np.where(left, idx - 1, idx)


def _actor_track(entry, ego_t, ego_s, ego_y):
    """다른 차의 (s, x, lane_id) 를 ego 스텝 시각에 맞춰 준다."""
    st = np.asarray(entry["state_trajectory"], dtype=float)
    lane = entry["lane_trajectory"]
    lane_t = np.array([e["time_s"] for e in lane], dtype=float)
    j = _nearest(lane_t, ego_t)
    lane_id = np.array([lane[k]["lane_id"] for k in j], dtype=float)
    return (ego_s + ego_y - np.interp(ego_t, st[:, 0], st[:, 2]),
            np.interp(ego_t, st[:, 0], st[:, 1]), lane_id)


def _moving_avg(values, width):
    """폭 width 이동평균 (길이 n-width+1). 가속도에 걸면 Δv/(width·dt) 와 같다."""
    if values.size < width or width < 2:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    return (cumsum[width:] - cumsum[:-width]) / float(width)


def _first_run_start(mask, hold):
    """mask 가 hold 스텝 이상 연속 True 인 첫 인덱스."""
    run = 0
    for i, ok in enumerate(mask):
        run = run + 1 if ok else 0
        if run >= hold:
            return i - hold + 1
    return None


def _mask_runs(mask, hold):
    """mask 가 hold 스텝 이상 연속 True 인 극대 구간 (시작, 끝) 인덱스 목록."""
    out = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= hold:
                out.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= hold:
        out.append((start, len(mask) - 1))
    return out


def cutout_response(row, data, group, run_name, sweep, tracks, t, t0, dt,
                    cmd, ego_s, ego_x, ego_v, ego_lane_id, steps, d_safe):
    """cut-out 그룹 전용 지표. cut-in 블록이 None 으로 남긴 열을 덮어쓴다."""
    name = "%s/%s" % (group, run_name)
    row["cutout_kind"] = re.sub(r"_[0-9]+$", "", run_name)
    sublv_expected = "no_sublv" not in name
    # cut-out 전용 스윕 축 (target_lead_gap/trigger_distance 는 SWEEP_PARAMS 에 이미 있다)
    row["cutout_direction"] = sweep.get("cutout_direction")
    row["target_lead_speed_delta"] = sweep.get("target_lead_speed_delta")
    row["cfg_trigger_time_s"] = sweep.get("trigger_time_s")
    lv_key = next((k for k in data if k.startswith("target_cutout")), None)
    sublv_key = next((k for k in data if k.startswith("target_lead_after_cutout")), None)
    row["has_sublv_actor"] = sublv_key is not None
    row["v_avg_window"] = row["v_avg"]  # cut-in 과 같은 창 전체 평균
    row["v_avg"] = None                 # cut-out 의 v_avg 는 [t_trigger, 창끝]
    row["v_avg_post"] = None
    for key in ("t_out", "t_out_minus_trigger", "t_settle_end", "gap_at_trigger",
                "v_ego_at_trigger", "v_lv_at_trigger", "min_bumper_gap_sublv",
                "trigger_distance_at_start", "cutout_started", "cutout_completed",
                "lv_inlane_at_trigger", "t_pred", "t_pred_minus_trigger",
                "dt_ant_ctrl", "dt_ant_ctrl_signed", "gap_err_max", "gap_err_avg",
                "gap_err_signed_avg", "gap_err_short_max", "gap_err_excess_max"):
        row[key] = None
    row["sublv_inlane_steps"] = 0
    row["lv_sublv_contact"] = False
    row["onset_sign"] = "-" if sublv_expected else "+"

    trig_abs = None
    if lv_key is not None:
        lv_log = data[lv_key].get("policy_log", {})
        trig = lv_log.get("trigger_time_s")
        trig_abs = None if trig is None else float(trig)
        row["t_trigger"] = None if trig_abs is None else round(trig_abs - t0, 3)
        row["trigger_mode"] = lv_log.get("trigger_mode")
        row["trigger_distance_at_start"] = lv_log.get("trigger_distance_at_start")
        row["cutout_started"] = lv_log.get("cutout_started")
        row["cutout_completed"] = lv_log.get("cutout_completed")

    # --- t_out: LV 중심이 ego 차선을 벗어난 순간 ---
    out_abs = None
    if lv_key in tracks:
        s_lv, _x_lv, lane_lv = tracks[lv_key]
        if trig_abs is not None:
            after = np.nonzero(t >= trig_abs - 1e-9)[0]
            if after.size:
                i0 = int(after[0])
                i_out = _first_run_start(lane_lv[i0:] != ego_lane_id[i0:],
                                         CUTOUT_OUT_HOLD_STEPS)
                if i_out is not None:
                    out_abs = float(t[i0 + i_out])
            if t[0] - 1e-9 <= trig_abs <= t[-1] + 1e-9:
                i_tr = int(_nearest(t, np.array([trig_abs]))[0])
                row["lv_inlane_at_trigger"] = bool(lane_lv[i_tr] == ego_lane_id[i_tr])
                row["gap_at_trigger"] = float(s_lv[i_tr] - ego_s[i_tr] - VEH_LEN)
                row["v_ego_at_trigger"] = float(ego_v[i_tr])
                lv_st = np.asarray(data[lv_key]["state_trajectory"], dtype=float)
                row["v_lv_at_trigger"] = float(
                    np.interp(trig_abs, lv_st[:, 0], lv_st[:, 4]))
    row["t_out"] = None if out_abs is None else round(out_abs - t0, 3)
    if out_abs is not None and row["t_trigger"] is not None:
        row["t_out_minus_trigger"] = round(row["t_out"] - row["t_trigger"], 3)

    # --- 트리거 직전 정상상태 / Δt_ant ---
    hold_settle = max(1, int(round(SETTLE_MIN_S / dt)))
    calm = np.abs(cmd) < SETTLE_ACCEL
    row["settled_before_trigger"] = None
    if trig_abs is not None:
        look = (t >= trig_abs - SETTLE_LOOKBACK_S) & (t <= trig_abs + 1e-9)
        row["settled_before_trigger"] = float(
            look.any() and _first_run_start(calm[look], hold_settle) is not None)
    settle_end = None
    if out_abs is not None:
        for i0, i1 in _mask_runs(calm, hold_settle):
            if t[i0] < out_abs:  # t_out 을 걸쳐 이어지는 구간도 그 구간의 끝을 쓴다
                settle_end = float(t[i1])
    row["t_settle_end"] = None if settle_end is None else round(settle_end - t0, 3)
    # 반응 개시 = 트리거 3 s 전부터 t_out 5 s 후 사이에서, 부호 맞는 가속이 1.0 s
    # 이상 끊기지 않고 임계를 넘는 첫 구간의 시작. STDAN 계열은 정착 후에도
    # ±0.3~0.6 m/s^2 진동이 남아 짧은 hold 로는 진동을 개시로 오인한다.
    onset = (cmd <= -CUTOUT_ONSET_ACCEL) if sublv_expected else (cmd >= CUTOUT_ONSET_ACCEL)
    t_onset = None
    if trig_abs is not None:
        hi = (out_abs + CUTOUT_ONSET_POST_S if out_abs is not None
              else trig_abs + CUTOUT_ONSET_POST_S + 3.0)
        win = (t >= trig_abs - CUTOUT_ONSET_PRE_S) & (t <= hi)
        idx = np.nonzero(win)[0]
        if idx.size:
            hold = max(1, int(round(CUTOUT_ONSET_HOLD_S / dt)))
            i_on = _first_run_start(onset[idx[0]:idx[-1] + 1], hold)
            if i_on is not None:
                t_onset = float(t[idx[0] + i_on])
    row["t_onset"] = None if t_onset is None else round(t_onset - t0, 3)
    row["dt_ant_ctrl"] = None
    row["dt_ant_ctrl_signed"] = None
    if t_onset is not None and out_abs is not None:
        row["dt_ant_ctrl_signed"] = round(out_abs - t_onset, 3)
        row["dt_ant_ctrl"] = max(0.0, row["dt_ant_ctrl_signed"])

    # --- Δt_ant: 예측 선제성 (제어 반응이 아니라 예측 시점) ---
    # LV 가 차선을 벗어난다고 예측기가 처음 말한 순간부터 실제로 벗어난 t_out
    # 까지. 제어기가 그 예측으로 무엇을 했는지와 무관한 예측기 지표다. subLV 가
    # 있는 장면에서 무조건 빨리 감속하는 것이 좋은 것은 아니어서(벌어진 간격을
    # 다시 채우는 절충이 중요하다) 제어 개시 시점은 dt_ant_ctrl 로 따로 남긴다.
    row["dt_ant"] = None
    row["dt_ant_signed"] = None
    row["cutout_call_max"] = None
    if lv_key in tracks and len(steps) == t.size:
        s_lv, _x, _lane = tracks[lv_key]
        call = _target_call(steps, s_lv, _predicts_lane_exit)
        finite = call[np.isfinite(call)]
        row["cutout_call_max"] = round(float(np.max(finite)), 4) if finite.size else None
        pred_abs = _prediction_onset(
            t, call, None if trig_abs is None else trig_abs - CUTOUT_ONSET_PRE_S, dt)
        row["t_pred"] = None if pred_abs is None else round(pred_abs - t0, 3)
        if pred_abs is not None and row["t_trigger"] is not None:
            row["t_pred_minus_trigger"] = round(row["t_pred"] - row["t_trigger"], 3)
        if pred_abs is not None and out_abs is not None:
            row["dt_ant_signed"] = round(out_abs - pred_abs, 3)
            row["dt_ant"] = max(0.0, row["dt_ant_signed"])

    # --- LV 와의 간격이 얼마나 어긋나는가 (부족·초과 모두) ---
    # |범퍼간격 - d_safe| / d_safe. 뒤처져도 붙어도 벌한다. LV 가 아직 ego
    # 차선에서 앞서는 동안만 본다: LV 가 빠진 뒤의 subLV 간격은 시나리오
    # 기하(앞차를 89 m 앞에 둔다)가 지배해 제어 품질을 재지 못한다.
    if lv_key in tracks and trig_abs is not None:
        s_lv, x_lv, lane_lv = tracks[lv_key]
        lead = ((lane_lv == ego_lane_id) & (np.abs(x_lv - ego_x) <= LANE_HALF_WIDTH)
                & (s_lv > ego_s) & (t >= trig_abs + EVENT_PRE_S))
        if out_abs is not None:
            lead &= t <= out_abs
        if lead.any():
            signed = (s_lv[lead] - ego_s[lead] - VEH_LEN - d_safe[lead]) / d_safe[lead] * 100.0
            err = np.abs(signed)
            row["gap_err_max"] = round(float(np.max(err)), 4)
            row["gap_err_avg"] = round(float(np.mean(err)), 4)
            # 부호를 살린 분해. 음수는 안전거리 안으로 들어간 쪽(부족), 양수는
            # 뒤처진 쪽(초과)이고 gap_err_max = max(short_max, excess_max) 이다.
            # 선제 감속은 초과만 키우므로, 둘을 합쳐 재는 gap_err_max 는 일찍
            # 감속한 정책을 늦게 감속한 정책과 같은 방향으로 벌한다.
            row["gap_err_signed_avg"] = round(float(np.mean(signed)), 4)
            row["gap_err_short_max"] = round(float(np.max(np.maximum(-signed, 0.0))), 4)
            row["gap_err_excess_max"] = round(float(np.max(np.maximum(signed, 0.0))), 4)

    # --- 속도 ---
    if trig_abs is not None:
        sel = t >= trig_abs - 1e-9
        row["v_avg"] = float(np.mean(ego_v[sel])) if sel.any() else None
    if out_abs is not None:
        sel = t >= out_abs - 1e-9
        row["v_avg_post"] = float(np.mean(ego_v[sel])) if sel.any() else None

    # --- subLV ---
    if sublv_key in tracks:
        s_sub, x_sub, lane_sub = tracks[sublv_key]
        inlane = ((lane_sub == ego_lane_id) & (np.abs(x_sub - ego_x) <= LANE_HALF_WIDTH)
                  & (s_sub > ego_s))
        row["sublv_inlane_steps"] = int(inlane.sum())
        if inlane.any():
            row["min_bumper_gap_sublv"] = float(
                np.min(s_sub[inlane] - ego_s[inlane]) - VEH_LEN)
        # 갭 회수 속도. gap_err_excess_max 는 [t_trigger-5 s, t_out] 만 보고 최댓값
        # 하나를 내므로 "얼마나 뒤처졌나"는 재도 "얼마나 빨리 되감았나"는 못 잰다.
        # 이탈 이후 subLV 를 상대로 남은 초과 여유를 두 가지로 잰다.
        if out_abs is not None:
            after = inlane & (t >= out_abs - 1e-9)
            if after.any():
                excess = np.maximum(
                    (s_sub[after] - ego_s[after] - VEH_LEN) - d_safe[after], 0.0)
                span = float(t[after][-1] - t[after][0])
                # 시간평균 초과 [m]: 낮을수록 빨리 되감았다. 검열이 없다.
                row["gap_recover_avg"] = round(float(np.mean(excess)), 4)
                # 요구 안전거리의 10 % 이내로 들어와 그 뒤로 유지되는 첫 시각
                # [t_out 이후 s]. 창 끝까지 못 들면 None (검열) -- 그 런은
                # gap_recover_censored 로 표시한다.
                near = excess <= 0.10 * d_safe[after]
                t_after = t[after]
                settle = None
                for k in range(near.size):
                    if near[k] and bool(np.all(near[k:])):
                        settle = float(t_after[k] - out_abs); break
                row["t_gap_recover"] = None if settle is None else round(settle, 3)
                row["gap_recover_censored"] = settle is None
                row["gap_recover_span_s"] = round(span, 3)
    if trig_abs is not None:
        hi = (out_abs + CUTOUT_CONTACT_POST_S if out_abs is not None
              else trig_abs + 3.0)
        for e in data.get("_collision_log", []):
            if not trig_abs - CUTOUT_CONTACT_PRE_S <= float(e.get("time_s", 0.0)) <= hi:
                continue
            roles = "%s|%s" % (e.get("actor_role"), e.get("other_actor_role"))
            if "target_cutout" in roles and "target_lead_after_cutout" in roles:
                row["lv_sublv_contact"] = True
                break
    for key in ("gap_at_trigger", "v_ego_at_trigger", "v_lv_at_trigger",
                "v_avg", "v_avg_window", "v_avg_post", "min_bumper_gap_sublv"):
        if row.get(key) is not None:
            row[key] = round(row[key], 4)


#: 이벤트 구간 열. ``--window-start-s`` 로 창 앞을 잘라도 이 열들은 절단 전
#: 창에서 잰 값을 쓴다 -- 이벤트 창은 t_trigger 기준으로 이미 제 구간을 자르고
#: 있어서, 창을 또 자르면 같은 지표를 두 번 자르는 셈이 된다.
EVENT_KEYS = ("j_avg_cmd_ev", "j_max_cmd_ev", "j_at_limit_ev", "j_avg_cmd_ev_filt",
              "j_max_cmd_ev_filt", "a_avg_ev", "a_min_ev", "j_avg_filt_ev",
              "j_max_filt_ev", "T_rec_ev", "min_bumper_gap_sublv_ev", "delta_max_ev",
              "delta_avg_ev", "min_bumper_gap_ev", "v_avg_ev", "v_min_ev", "dt_ant_ev",
              "collisions_in_event", "ego_collision_ev")


def collect_run(policy, group, run_dir, window_s=None, window_start_s=None):
    row = collections.OrderedDict()
    row["policy"] = policy
    row["policy_label"] = policy_label(policy)
    row["group"] = group
    row["run"] = run_dir.name
    row["run_dir"] = str(run_dir)
    for key in SWEEP_PARAMS:
        row[key] = None
    row["valid"] = False
    row["invalid_reason"] = ""
    try:
        cfg = json.load(open(str(run_dir / "resolved_config.json")))
        met = json.load(open(str(run_dir / "metrics.json")))
        data = pickle.load(open(str(run_dir / "scenario_result.pkl"), "rb"))
    except Exception as exc:  # 깨진 런도 데이터다
        row["invalid_reason"] = "load: %r" % (exc,)
        return row
    sweep = cfg.get("sweep_params", {})
    for key in SWEEP_PARAMS:
        row[key] = sweep.get(key)
    if row["lane_change_distance"] is None:  # 소요시간으로 스윕하면 파생값만 남는다
        for params in cfg.get("scenario", {}).get("vehicle_params", []):
            if params.get("role") in ("target_cutin", "target_cutout"):
                row["lane_change_distance"] = params.get("lane_change_distance")
                break
    row["ran_successfully"] = bool(met.get("ran_successfully"))

    ego_key = next(k for k in data if k.startswith("ego"))
    ego = data[ego_key]
    steps = ego["policy_log"]["steps"]
    all_t = np.array([s["time_s"] for s in steps], dtype=float)
    w0 = float(all_t[0])
    # 창 상한. 기본은 스윕이 돈 시간 전체이고, ``window_s`` 로 더 짧게 자를 수
    # 있다.  컷인 런의 뒤쪽에서 추적 리드가 옆 차선 차량으로 넘어가 범퍼 간격이
    # 단조 붕괴하는데 ``accel_cmd`` 는 0 근처에 머무는 구간이 있다(2026-09-09,
    # 01/cutin_0014 에서 t 절대 21.5 s 부터 간격 19.9 -> -6.3 m, delta 135 %).
    # 제어 결과가 아니라 리드 판정 아티팩트라 지표를 오염시킨다.
    w1 = w0 + float(sweep.get("max_sim_time_s") or DEFAULT_WINDOW_S)
    if window_s is not None:
        w1 = min(w1, w0 + float(window_s))
    # 창 하한. ``window_start_s`` 는 제어 시작 직후를 잘라낸다. ego 는 요구
    # 안전거리 안쪽에서 스폰해 첫 9 틱 동안 accel_cmd 를 -1.00 -> -3.00 으로
    # 틱당 0.25 씩(= 0.25/0.05 = 5.0, jerk_limit 에 정확히 포화) 내리는데, 네
    # 정책이 완전히 같은 값이라 자르지 않으면 j_max 가 정책이 아니라 시나리오
    # 초기 조건을 잰다. **이벤트 구간 열(_ev)은 이 절단의 영향을 받지 않는다** --
    # main 이 절단 없이 한 번 더 집계해 그 열만 되돌려 놓는다(EVENT_KEYS).
    w_lo = w0 if window_start_s is None else w0 + float(window_start_s)
    keep = np.nonzero((all_t >= w_lo - 1e-9) & (all_t <= w1 + 1e-9))[0]
    steps = [steps[i] for i in keep]
    t = all_t[keep]
    t0 = w0  # 제어 시작 = 시간 기준. 서버 경과 시계라 스폰 전 ~5 s 가 이미 지나 있다
    row["t0_sim"] = round(t0, 3)
    row["window_start_s"] = None if window_start_s is None else round(float(window_start_s), 3)
    row["window_s"] = round(float(t[-1]) - t0, 3)
    row["n_steps"] = len(steps)

    # --- 세로방향 좌표계: s_actor = ego_s + (y_ego - y_actor) ---
    ego_st = np.asarray(ego["state_trajectory"], dtype=float)
    ego_s = np.array([s["ego_s"] for s in steps], dtype=float)
    ego_v = np.array([s["ego_v"] for s in steps], dtype=float)
    ego_y = np.interp(t, ego_st[:, 0], ego_st[:, 2])
    ego_x = np.interp(t, ego_st[:, 0], ego_st[:, 1])
    resid = ego_s - (float(np.mean(ego_s + ego_y)) - ego_y)
    row["s_fit_resid_max"] = round(float(np.max(np.abs(resid))), 3)
    row["s_fit_resid_rms"] = round(float(np.sqrt(np.mean(resid ** 2))), 3)
    lag = max(1, int(round(S_MAP_LAG_S / np.median(np.diff(t)))))
    map_err = np.abs((ego_s[lag:] - ego_s[:-lag]) - (ego_y[:-lag] - ego_y[lag:]))
    map_err = np.concatenate([map_err, np.full(lag, map_err[-1])])
    row["s_map_err_window"] = round(float(np.max(map_err)), 3)

    # --- 유효성 ---
    solve_path = [s.get("solve_path") for s in steps]
    row["fallback_ratio"] = round(
        sum(1 for p in solve_path if p == "fallback") / float(len(steps)), 4)
    row["slack_steps"] = sum(1 for s in steps if not s.get("feasible"))
    pairs = set()
    ego_hit = False
    for e in data.get("_collision_log", []):
        ts = float(e.get("time_s", 0.0))
        if w0 - 1e-9 <= ts <= w1 + COLLISION_PAD_S:
            roles = frozenset((str(e.get("actor_role") or e.get("actor_key")),
                               str(e.get("other_actor_role") or e.get("other_actor_type"))))
            pairs.add(roles)
            ego_hit = ego_hit or any("ego" in r for r in roles)
    row["collisions_in_window"] = len(pairs)
    row["collision_pairs"] = "; ".join(sorted("+".join(sorted(p)) for p in pairs))
    row["ego_collision"] = ego_hit

    # --- 승차감 (창의 첫 샘플 = 스폰 과도 차분이므로 버림) ---
    accel = np.array(_finite([s.get("actual_accel") for s in steps])[1:])
    jerk = np.array(_finite([s.get("actual_jerk") for s in steps])[1:])
    row["a_avg"] = float(np.mean(np.abs(accel))) if accel.size else None
    row["a_min"] = float(np.min(accel)) if accel.size else None
    row["j_avg"] = float(np.mean(np.abs(jerk))) if jerk.size else None
    row["j_max"] = float(np.max(np.abs(jerk))) if jerk.size else None
    row["j_p99"] = float(np.percentile(np.abs(jerk), 99.0)) if jerk.size else None
    dt = float(np.median(np.diff(t)))
    width = max(2, int(round(CTRL_DT / dt)))
    accel_f = _moving_avg(accel, width)
    jerk_f = (np.abs(accel_f[width:] - accel_f[:-width]) / (width * dt)
              if accel_f.size > width else np.array([]))
    row["a_min_filt"] = float(np.min(accel_f)) if accel_f.size else None
    row["j_avg_filt"] = float(np.mean(jerk_f)) if jerk_f.size else None
    row["j_max_filt"] = float(np.max(jerk_f)) if jerk_f.size else None
    # 명령 저크: 제어기가 실제로 낸 값(로그 command_jerk = Δaccel_cmd/Δt).
    # 원시값. 표는 아래 대역 제한판(j_*_cmd_filt)을 쓴다.
    cmd_jerk = np.abs(np.array(_finite([s.get("command_jerk") for s in steps])[1:]))
    row["j_avg_cmd"] = float(np.mean(cmd_jerk)) if cmd_jerk.size else None
    row["j_max_cmd"] = float(np.max(cmd_jerk)) if cmd_jerk.size else None
    # 명령 저크가 한계에 붙어 있던 시간 비율. j_max 는 컷인에서 전 셀이 한계에
    # 닿아(원시 j_max 가 4 정책 18/18 셀 모두 5.00) **검열된 지표**라 정책을 못
    # 가른다. 붙어 있던 비율은 가른다: 컷인 aggressive 에서 SCC 12.4 % · LSTM 12.6 %
    # 대 STDAN 7.9 % · Proposed 9.6 % (2026-09-10). 표 열은 아직 안 바꿨고 CSV 에만
    # 낸다 -- 논문 표에 j_max 대신 이걸 쓸지는 따로 정할 것.
    lim = np.array(_finite([s.get("command_jerk_limit") for s in steps]), dtype=float)
    lim = float(np.median(lim)) if lim.size else float("nan")
    row["j_at_limit"] = (float(np.mean(cmd_jerk >= 0.95 * lim))
                         if cmd_jerk.size and np.isfinite(lim) and lim > 0 else None)
    cmd_accel = np.array(_finite([s.get("accel_cmd") for s in steps])[1:])
    row["a_avg_cmd"] = float(np.mean(np.abs(cmd_accel))) if cmd_accel.size else None
    row["a_min_cmd"] = float(np.min(cmd_accel)) if cmd_accel.size else None
    # 대역 제한 명령 저크: 명령은 20 Hz 로 갱신되고 예측이 틱마다 조금씩 흔들려
    # 정상 추종 중에도 ±0.5 m/s^2 의 고주파 성분이 남는다 — 차량은 이를 따라가지
    # 못하지만 raw 명령 저크는 그대로 센다. 실측 저크와 같은 0.2 s 대역으로 눌러서
    # "차량이 실제로 따라갈 수 있는 명령 변화"만 남긴 값.
    cmd_full = np.array([s.get("accel_cmd") if s.get("accel_cmd") is not None else np.nan
                         for s in steps], dtype=float)
    cmd_f = _moving_avg(np.nan_to_num(cmd_full, nan=0.0), width)
    cmd_jf = (np.abs(cmd_f[width:] - cmd_f[:-width]) / (width * dt)
              if cmd_f.size > width else np.array([]))
    row["j_avg_cmd_filt"] = float(np.mean(cmd_jf)) if cmd_jf.size else None
    row["j_max_cmd_filt"] = float(np.max(cmd_jf)) if cmd_jf.size else None

    # --- 차선 내 선행차 / 안전거리 위반 ---
    ego_lane = ego["lane_trajectory"]
    ego_lane_t = np.array([e["time_s"] for e in ego_lane], dtype=float)
    ego_lane_id = np.array([ego_lane[k]["lane_id"]
                            for k in _nearest(ego_lane_t, t)], dtype=float)
    tracks = {}
    for key, entry in data.items():
        if key == ego_key or not isinstance(entry, dict):
            continue
        if "state_trajectory" in entry and "lane_trajectory" in entry:
            tracks[key] = _actor_track(entry, t, ego_s, ego_y)
    gap = np.full(len(t), np.inf)
    for s_a, x_a, lane_a in tracks.values():
        inlane = (lane_a == ego_lane_id) & (np.abs(x_a - ego_x) <= LANE_HALF_WIDTH)
        ahead = inlane & (s_a > ego_s)
        gap = np.where(ahead, np.minimum(gap, s_a - ego_s), gap)
    has_lead = np.isfinite(gap)
    bumper = np.where(has_lead, gap - VEH_LEN, np.nan)
    d_safe = D0 + TAU * ego_v
    delta = np.where(has_lead, np.maximum(0.0, (d_safe - bumper) / d_safe) * 100.0, 0.0)
    row["has_inlane_lead"] = bool(has_lead.any())
    row["inlane_steps"] = int(has_lead.sum())
    row["min_bumper_gap"] = float(np.nanmin(bumper)) if has_lead.any() else None
    row["delta_max"] = float(np.max(delta))
    row["delta_avg"] = float(np.mean(delta[has_lead])) if has_lead.any() else None
    row["delta_avg_window"] = float(np.mean(delta))

    # --- 유효성 (s 매핑은 "실제로 쓰인 스텝"에서만 따진다) ---
    row["s_map_err"] = round(float(np.max(map_err[has_lead])), 3) if has_lead.any() else 0.0
    reasons = []
    if not row["ran_successfully"]:
        reasons.append("ran_successfully=False")
    if row["fallback_ratio"] > FALLBACK_TOL:
        reasons.append("fallback_ratio=%.3f" % row["fallback_ratio"])
    if row["s_map_err"] > S_MAP_TOL:
        reasons.append("s_map_err=%.2f m" % row["s_map_err"])
    row["valid"] = not reasons
    row["invalid_reason"] = "; ".join(reasons)

    # --- cut-in 응답 ---
    tv_key = next((k for k in data if k.startswith("target_cutin")), None)
    t_cross = None
    row["t_trigger"] = None
    row["trigger_mode"] = None
    row["lane_change_completed"] = None
    row["passed"] = None
    row["t_pass"] = None
    trig_abs = None
    if tv_key is not None:
        tv_log = data[tv_key].get("policy_log", {})
        trig = tv_log.get("trigger_time_s")
        trig_abs = None if trig is None else float(trig)
        row["t_trigger"] = None if trig is None else round(float(trig) - t0, 3)
        row["trigger_mode"] = tv_log.get("trigger_mode")
        row["lane_change_completed"] = tv_log.get("lane_change_completed")
        s_tv, x_tv, lane_tv = tracks[tv_key]
        cross = np.nonzero((lane_tv == ego_lane_id)
                           & (np.abs(x_tv - ego_x) <= LANE_HALF_WIDTH))[0]
        if cross.size:
            t_cross = float(t[cross[0]])
        row["passed"] = float(ego_s[-1] > s_tv[-1])
        ahead_tv = np.nonzero(ego_s > s_tv)[0]
        row["t_pass"] = round(float(t[ahead_tv[0]]) - t0, 3) if ahead_tv.size else None
    row["t_cross"] = None if t_cross is None else round(t_cross - t0, 3)
    row["t_cross_minus_trigger"] = (
        None if t_cross is None or row["t_trigger"] is None
        else round(row["t_cross"] - row["t_trigger"], 3))

    # cut-in 차량이 옛 차선 앞차를 스치고 지나가는 구간의 접촉 (시나리오 건전성)
    row["tv_lead_contact"] = False
    if trig_abs is not None:
        hi = t_cross + 2.0 if t_cross is not None else trig_abs + 3.0
        for e in data.get("_collision_log", []):
            if not trig_abs - 0.5 <= float(e.get("time_s", 0.0)) <= hi:
                continue
            roles = "%s|%s" % (e.get("actor_role"), e.get("other_actor_role"))
            if "target_cutin" in roles and "traffic_cutin_lane_lead" in roles:
                row["tv_lead_contact"] = True
                break

    cmd = np.array([s.get("accel_cmd") if s.get("accel_cmd") is not None else 0.0
                    for s in steps], dtype=float)
    i_on = _first_run_start(cmd <= ONSET_ACCEL, ONSET_HOLD_STEPS)
    t_onset = None if i_on is None else float(t[i_on])
    # t_onset 이 0 에 가까우면 창 시작 전에 이미 밟고 있었을 수 있다 = 좌측 절단
    row["t_onset"] = None if t_onset is None else round(t_onset - t0, 3)
    row["dt_ant_ctrl"] = (round(t_cross - t_onset, 3)
                          if t_cross is not None and t_onset is not None and t_onset < t_cross
                          else None)

    # Δt_ant: cut-out 과 같은 예측 선제성. TV 의 모드 중 지평선 끝이 ego 차선
    # **안**인 것들의 확률합이 임계를 1.0 s 이상 넘는 첫 시점부터 실제 진입까지.
    row["dt_ant"] = None
    row["t_pred"] = None
    row["cutin_call_max"] = None
    if tv_key in tracks and len(steps) == t.size:
        s_tv, _x, _lane = tracks[tv_key]
        call = _target_call(steps, s_tv, _predicts_lane_entry)
        finite = call[np.isfinite(call)]
        row["cutin_call_max"] = round(float(np.max(finite)), 4) if finite.size else None
        pred_abs = _prediction_onset(t, call, float(t[0]), dt)
        row["t_pred"] = None if pred_abs is None else round(pred_abs - t0, 3)
        if pred_abs is not None and t_cross is not None:
            row["dt_ant"] = max(0.0, round(t_cross - pred_abs, 3))

    row["T_rec"] = None
    row["T_rec_censored"] = None
    if t_cross is not None:
        i_cross = int(np.searchsorted(t, t_cross))
        hold = max(1, int(round(REC_HOLD_S / dt)))
        if delta[i_cross] == 0.0:
            row["T_rec"] = 0.0
        else:
            ok = delta[i_cross:] == 0.0
            i_ok = _first_run_start(ok, hold)
            if i_ok is None:
                row["T_rec_censored"] = round(float(t[-1]) - t_cross, 3)
            else:
                row["T_rec"] = round(float(t[i_cross + i_ok]) - t_cross, 3)

    row["v_avg"] = float(np.mean(ego_v))
    row["v_min"] = float(np.min(ego_v))
    if is_cutout_group(group):
        cutout_response(row, data, group, run_dir.name, sweep, tracks, t, t0, dt,
                        cmd, ego_s, ego_x, ego_v, ego_lane_id, steps, d_safe)
    # 이벤트 구간 명령 저크: 창 전체 평균은 이벤트가 끝난 뒤의 정상 추종 구간(창의
    # 절반 이상)이 지배한다. 예측기를 쓰는 정책은 그 구간에서 틱마다 예측이 조금씩
    # 흔들려 ±0.2 m/s^2 의 미세 진동이 남고, 그것이 컷인 응답 자체의 매끄러움을 덮는다.
    # [t_trigger-5 s, t_trigger+9 s] 로 잘라 기동 구간만 본다(트리거 없는 03/04 는 전 구간).
    row["j_avg_cmd_ev"] = row["j_avg_cmd"]
    row["j_max_cmd_ev"] = row["j_max_cmd"]
    # 이벤트 구간 + 대역 제한 (표 지표 j_*_cmd_filt 과 같은 식, 창만 자른 것).
    # j_*_cmd_ev 는 대역 제한 없는 원시 명령 저크이고, j_*_cmd_filt 은 대역
    # 제한이지만 런 전체 평균이라 정상 추종 구간이 지배한다. 기동만 보려면
    # 둘을 겹친 이 열을 쓴다. cmd_jf[k] 는 [k,k+w) 평균과 [k+w,k+2w) 평균의
    # 차이라 t[k+w] 부근이 중심이다.
    row["a_avg_ev"] = row["a_avg_cmd"]
    row["a_min_ev"] = row["a_min_cmd"]
    row["j_at_limit_ev"] = row["j_at_limit"]
    row["j_avg_cmd_ev_filt"] = row["j_avg_cmd_filt"]
    row["j_max_cmd_ev_filt"] = row["j_max_cmd_filt"]
    if row.get("t_trigger") is not None and len(steps) == t.size:
        lo = t0 + row["t_trigger"] + EVENT_PRE_S
        hi = t0 + row["t_trigger"] + EVENT_POST_S
        cj = np.array([s.get("command_jerk") if s.get("command_jerk") is not None
                       else np.nan for s in steps], dtype=float)
        cj = np.abs(cj[(t >= lo) & (t <= hi)])
        cj = cj[np.isfinite(cj)]
        if cj.size:
            row["j_avg_cmd_ev"] = float(np.mean(cj))
            row["j_max_cmd_ev"] = float(np.max(cj))
            if np.isfinite(lim) and lim > 0:
                row["j_at_limit_ev"] = float(np.mean(cj >= 0.95 * lim))
        ev_cmd = (t >= lo) & (t <= hi)
        if ev_cmd.any():
            row["a_avg_ev"] = float(np.mean(np.abs(cmd_full[ev_cmd])))
            row["a_min_ev"] = float(np.nanmin(cmd_full[ev_cmd]))
        if cmd_jf.size:
            t_j = t[width:width + cmd_jf.size]
            ev_j = (t_j >= lo) & (t_j <= hi)
            if ev_j.any():
                row["j_avg_cmd_ev_filt"] = float(np.mean(cmd_jf[ev_j]))
                row["j_max_cmd_ev_filt"] = float(np.max(cmd_jf[ev_j]))

    # 이벤트 구간 안전·속도. delta_max 는 컷인 응답 안에서 나므로 창을 잘라도 거의
    # 같지만, delta_avg 와 v_avg 는 평균이라 이벤트 뒤의 조용한 정상주행이 값을
    # 끌어내린다(창 전체 delta_avg 는 회복 후 0 인 스텝이 절반 이상이다).
    # 충돌도 같은 창으로 세어 컷인과 무관한 뒤쪽 경로주행 충돌을 뺀다.
    # T_rec 은 이미 t_cross 기준이고 gap_err_* 는 [t_trigger-1 s, t_out] 로 따로 좁아
    # 이벤트판이 따로 없다.
    row["j_avg_filt_ev"] = row["j_avg_filt"]
    row["j_max_filt_ev"] = row["j_max_filt"]
    row["T_rec_ev"] = row["T_rec"]
    # Δt_ant 는 예측 시각 기준이라 이벤트판이 따로 없었는데, ``--window-start-s``
    # 로 창 앞을 자르면 그 구간에 있던 예측을 못 보게 되어 값이 깎인다(컷인
    # 2.10 -> 1.59). 이벤트 표는 절단의 영향을 받지 않아야 하므로 별도 열로 둔다.
    row["dt_ant_ev"] = row.get("dt_ant")
    row["min_bumper_gap_sublv_ev"] = row.get("min_bumper_gap_sublv")
    row["delta_max_ev"] = row["delta_max"]
    row["delta_avg_ev"] = row["delta_avg"]
    row["min_bumper_gap_ev"] = row["min_bumper_gap"]
    row["v_avg_ev"] = row["v_avg"]
    row["v_min_ev"] = row["v_min"]
    row["collisions_in_event"] = row["collisions_in_window"]
    row["ego_collision_ev"] = row["ego_collision"]
    if row.get("t_trigger") is not None:
        lo = t0 + row["t_trigger"] + EVENT_PRE_S
        hi = t0 + row["t_trigger"] + EVENT_POST_S
        ev = (t >= lo) & (t <= hi)
        if ev.any():
            row["delta_max_ev"] = float(np.max(delta[ev]))
            lead_ev = ev & has_lead
            row["delta_avg_ev"] = (float(np.mean(delta[lead_ev])) if lead_ev.any() else None)
            row["min_bumper_gap_ev"] = (float(np.nanmin(bumper[lead_ev]))
                                        if lead_ev.any() else None)
            row["v_avg_ev"] = float(np.mean(ego_v[ev]))
            row["v_min_ev"] = float(np.min(ego_v[ev]))
            sub = tracks.get(next((k for k in data if k.startswith(
                "target_lead_after_cutout")), None))
            if sub is not None:
                s_sub, x_sub, lane_sub = sub
                m = (ev & (lane_sub == ego_lane_id)
                     & (np.abs(x_sub - ego_x) <= LANE_HALF_WIDTH) & (s_sub > ego_s))
                row["min_bumper_gap_sublv_ev"] = (
                    float(np.min(s_sub[m] - ego_s[m]) - VEH_LEN) if m.any() else None)
        # 실측 0.2 s 대역 저크의 이벤트판. accel 은 첫 샘플을 버리고 만들므로
        # jerk_f[k] 의 시각은 t[1+width+k] 다. 중간에 결측이 있어 길이가 어긋나면
        # 정렬을 믿을 수 없으니 창 전체 값을 그대로 둔다.
        if jerk_f.size and accel.size == len(steps) - 1:
            t_jf = t[1 + width:1 + width + jerk_f.size]
            ev_jf = (t_jf >= lo) & (t_jf <= hi)
            if ev_jf.any():
                row["j_avg_filt_ev"] = float(np.mean(jerk_f[ev_jf]))
                row["j_max_filt_ev"] = float(np.max(jerk_f[ev_jf]))
        # T_rec 의 이벤트판: 회복을 창 끝이 아니라 t_trigger+9 s 에서 자른다.
        if t_cross is not None:
            i_cross = int(np.searchsorted(t, t_cross))
            hold = max(1, int(round(REC_HOLD_S / dt)))
            keep = t[i_cross:] <= hi
            ok = (delta[i_cross:] == 0.0) & keep
            row["T_rec_ev"] = None
            if delta[i_cross] == 0.0:
                row["T_rec_ev"] = 0.0
            else:
                i_ok = _first_run_start(ok, hold)
                if i_ok is not None:
                    row["T_rec_ev"] = round(float(t[i_cross + i_ok]) - t_cross, 3)
        pairs_ev, ego_hit_ev = set(), False
        for e in data.get("_collision_log", []):
            ts = float(e.get("time_s", 0.0))
            if not lo - 1e-9 <= ts <= hi + 1e-9:
                continue
            roles = frozenset((str(e.get("actor_role") or e.get("actor_key")),
                               str(e.get("other_actor_role") or e.get("other_actor_type"))))
            pairs_ev.add(roles)
            ego_hit_ev = ego_hit_ev or any("ego" in r for r in roles)
        row["collisions_in_event"] = len(pairs_ev)
        row["ego_collision_ev"] = ego_hit_ev

    for key in ("a_avg", "a_avg_cmd", "a_min", "a_min_filt", "a_min_cmd",
                "j_avg", "j_avg_filt", "j_avg_cmd", "j_avg_cmd_ev", "j_avg_cmd_filt", "j_max",
                "j_max_filt", "j_max_cmd", "j_max_cmd_ev", "j_max_cmd_filt", "j_p99", "delta_max", "delta_avg",
                "delta_avg_window", "min_bumper_gap", "v_avg", "v_min"):
        if row.get(key) is not None:
            row[key] = round(row[key], 4)
    return row


def pooled_cutout_group(group):
    """논문 cut-out 표의 두 행: subLV 유무만 구분하고 트리거 변형은 합친다."""
    if not is_cutout_group(group):
        return group
    return "cutout_no_sublv" if "no_sublv" in group else "cutout_sublv"


def aggregate(rows, group_key=None):
    """(group, policy) 셀별 metric -> (mean, std, n).

    ``group_key`` 로 그룹을 합칠 수 있다(cut-out 표의 subLV 행은 17 m/13 m 트리거를
    한 행으로 합산한다).
    """
    cells = collections.OrderedDict()
    for row in rows:
        if not row["valid"]:
            continue
        group = row["group"] if group_key is None else group_key(row["group"])
        cells.setdefault((group, row["policy"]), []).append(row)
    out = collections.OrderedDict()
    for key in sorted(cells):
        sel = cells[key]
        stats = collections.OrderedDict()
        for metric in (AGG_METRICS_CUTOUT if is_cutout_group(key[0]) else AGG_METRICS):
            vals = _finite([r.get(metric) for r in sel])
            if vals:
                stats[metric] = (float(np.mean(vals)),
                                 float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                 len(vals))
            else:
                stats[metric] = (None, None, 0)
        stats["_n_runs"] = len(sel)
        stats["_collisions"] = sum(1 for r in sel if r.get("collisions_in_window"))
        stats["_ego_collisions"] = sum(1 for r in sel if r.get("ego_collision"))
        stats["_ego_collisions_ev"] = sum(1 for r in sel if r.get("ego_collision_ev"))
        # cut-out 셀에서는 같은 열이 LV↔subLV 접촉 수를 뜻한다
        stats["_tv_lead_contacts"] = sum(
            1 for r in sel if r.get("tv_lead_contact") or r.get("lv_sublv_contact"))
        stats["_label"] = sel[0]["policy_label"]
        out[key] = stats
    return out


def fmt(value, spec="%.2f"):
    return "-" if value is None else spec % value


def write_runs_csv(path, rows):
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(str(path), "w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path, agg):
    with open(str(path), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "policy", "policy_label", "metric",
                         "n", "mean", "std", "n_runs", "ego_collision_runs",
                         "tv_lead_contact_runs"])
        for (group, policy), stats in agg.items():
            for metric in [m for m in stats if not m.startswith("_")]:
                mean, std, n = stats[metric]
                writer.writerow([group, policy, stats["_label"], metric, n,
                                 "" if mean is None else "%.4f" % mean,
                                 "" if std is None else "%.4f" % std,
                                 stats["_n_runs"], stats["_ego_collisions"],
                                 stats["_tv_lead_contacts"]])


def write_tex(path, agg):
    lines = ["% aggregate_cutin_table.py 자동 생성 — table_cutin.tex 열 순서 그대로",
             "% a_avg, j_avg, j_max, delta_max, delta_avg, dt_ant, T_rec (셀 평균)",
             "% j_avg/j_max 는 **대역 제한 명령 저크** — accel_cmd 를 0.2 s 대역으로",
             "% 누른 뒤의 Δ/Δt 다. 원시 명령 저크는 20 Hz 예측 지터를 그대로 세서",
             "% 예측 기반 정책만 불리해지고 j_max 가 한계 10 에 포화한다.",
             "% 괄호 없는 값이 표에 들어간다."]
    groups = [g for g in sorted({k[0] for k in agg})
              if "no_cutin_decel" not in g and not is_cutout_group(g)]
    for gi, group in enumerate(groups):
        cells = [k for k in agg if k[0] == group]
        label = SCENARIO_LABEL.get(group, group.replace("_", "\\_"))
        lines.append("")
        lines.append("\\multirow{%d}{*}{\\shortstack[l]{%s}}" % (len(cells), label))
        for key in cells:
            stats = agg[key]
            vals = [fmt(stats[m][0]) for m in TEX_METRICS]
            lines.append(" & %s & %s \\\\ %% n=%d"
                         % (stats["_label"], " & ".join(vals), stats["_n_runs"]))
        lines.append("\\midrule" if gi < len(groups) - 1 else "\\bottomrule")
    for group in sorted({k[0] for k in agg if "no_cutin_decel" in k[0]}):
        lines.append("")
        lines.append("%% --- %s: v_avg (m/s), a_min (m/s^2), passed (fraction), "
                     "t_pass (s rel. t0) ---" % group)
        for key in [k for k in agg if k[0] == group]:
            stats = agg[key]
            vals = [fmt(stats[m][0]) for m in TEX_METRICS_03]
            lines.append(" %s & %s \\\\ %% n=%d"
                         % (stats["_label"], " & ".join(vals), stats["_n_runs"]))
    open(str(path), "w").write("\n".join(lines) + "\n")


def write_cutout_tex(path, agg):
    lines = ["% aggregate_cutin_table.py 자동 생성 — table_cutout.tex 열 순서 그대로",
             "% a_avg, j_avg, j_max, dt_ant, gap부족_max, gap초과_max, v_avg",
             "% (셀 평균; subLV 행은 트리거 17 m + 13 m 합산)",
             "% j_avg/j_max 는 **대역 제한 명령 저크**(accel_cmd 0.2 s 대역 후 Δ/Δt),",
             "% dt_ant = t_out - t_onset (선제성 없으면 0.00), v_avg 는 [t_trigger, 창끝] 평균.",
             "% gap부족/초과 = 안전거리 안으로 들어간 쪽 / 뒤처진 쪽의 최대치. 합쳐서 재면",
             "% 완화해서 붙는 정책과 못 따라가는 정책이 같은 값으로 찍힌다."]
    groups = sorted({k[0] for k in agg if is_cutout_group(k[0])})
    for gi, group in enumerate(groups):
        cells = [k for k in agg if k[0] == group]
        lines.append("")
        lines.append("\\multirow{%d}{*}{\\shortstack[l]{%s}} %% %s"
                     % (len(cells), cutout_label(group), group))
        for key in cells:
            stats = agg[key]
            vals = [fmt(stats[m][0]) for m in TEX_METRICS_CUTOUT]
            lines.append(" & %s & %s \\\\ %% n=%d"
                         % (stats["_label"], " & ".join(vals), stats["_n_runs"]))
        lines.append("\\midrule" if gi < len(groups) - 1 else "\\bottomrule")
    open(str(path), "w").write("\n".join(lines) + "\n")


def write_event_tex(path, agg):
    lines = ["% aggregate_cutin_table.py 자동 생성 — 이벤트 구간만 자른 표",
             "% 창 = [t_trigger - 1 s, t_trigger + 9 s] (기동 + 회복)",
             "% a_avg_ev, a_min_ev, j_avg, j_max (0.2 s 대역 명령 저크), j_avg 원시,\n"
             "% delta_max_ev, delta_avg_ev, min_bumper_gap_ev, v_avg_ev, v_min_ev, dt_ant",
             "% 표의 j_*_cmd_filt 은 같은 식이되 런 전체 평균이라 정상 추종 구간이 지배한다."]
    groups = sorted({k[0] for k in agg if agg[k]["a_avg_ev"][0] is not None
                     and not is_cutout_group(k[0])})
    for gi, group in enumerate(groups):
        cells = [k for k in agg if k[0] == group and agg[k]["a_avg_ev"][0] is not None]
        lines.append("")
        lines.append("\\multirow{%d}{*}{\\shortstack[l]{%s}} %% %s"
                     % (len(cells), cutout_label(group) if is_cutout_group(group)
                        else SCENARIO_LABEL.get(group, group), group))
        for key in cells:
            stats = agg[key]
            vals = [fmt(stats[EV_OF.get(m, m)][0]) for m in TEX_METRICS]
            lines.append(" & %s & %s \\\\ %% n=%d"
                         % (stats["_label"], " & ".join(vals), stats["_n_runs"]))
        lines.append("\\midrule" if gi < len(groups) - 1 else "\\bottomrule")
    open(str(path), "w").write("\n".join(lines) + "\n")


def _cell(stats, metric, event):
    """전체구간/이벤트 구간 공통 셀. event 면 같은 열의 이벤트판을 읽는다."""
    key = EV_OF.get(metric, metric) if event else metric
    mean, std, n = stats[key]
    cell = ("-" if mean is None
            else "%.2f ± %.2f%s" % (mean, std, "" if n == stats["_n_runs"]
                                    else " (n=%d)" % n))
    raw = RAW_OF_FILT.get(metric)
    if raw is not None:
        raw_mean = stats[EV_OF.get(raw, raw) if event else raw][0]
        if raw_mean is not None:
            cell += " (%.2f)" % raw_mean
    return cell


def markdown_table_cutout(agg, event=False):
    head = ("| 시나리오 | 제어기 | n | a_avg | j_avg 명령대역(실측대역) | j_max 명령대역(실측대역) "
            "| Δt_ant | gap부족_max | gap초과_max | v_avg | t_out-t_trig | gap@trig "
            "| 정상상태 | δ_max | subLV 최소간격 | ego충돌 | LV-subLV접촉 |")
    lines = [head, "|" + "---|" * 17]
    for (group, _policy), stats in agg.items():
        if not is_cutout_group(group):
            continue
        cells = [_cell(stats, m, event) for m in TEX_METRICS_CUTOUT]
        for metric in ("t_out_minus_trigger", "gap_at_trigger",
                       "settled_before_trigger", "delta_max", "min_bumper_gap_sublv"):
            mean = stats[EV_OF.get(metric, metric) if event else metric][0]
            cells.append("-" if mean is None else "%.2f" % mean)
        lines.append("| %s | %s | %d | %s | %d | %d |"
                     % (group, stats["_label"], stats["_n_runs"], " | ".join(cells),
                        stats["_ego_collisions_ev" if event else "_ego_collisions"],
                        stats["_tv_lead_contacts"]))
    return "\n".join(lines)


def markdown_table(agg, event=False):
    head = ("| 시나리오 | 제어기 | n | a_avg | j_avg 명령대역(실측대역) | j_max 명령대역(실측대역) "
            "| δ_max | δ_avg | Δt_ant | T_rec | ego충돌 | TV-앞차접촉 |")
    lines = [head, "|" + "---|" * 12]
    for (group, _policy), stats in agg.items():
        if is_cutout_group(group):
            continue
        cells = [_cell(stats, m, event) for m in TEX_METRICS]
        lines.append("| %s | %s | %d | %s | %d | %d |"
                     % (group, stats["_label"], stats["_n_runs"], " | ".join(cells),
                        stats["_ego_collisions_ev" if event else "_ego_collisions"],
                        stats["_tv_lead_contacts"]))
    return "\n".join(lines)


def write_metrics_md(path, agg, rows, argv_note):
    doc, _, doc_cutout = __doc__.split("[정의]", 1)[1].partition("[cut-out 정의]")
    doc = doc.strip()
    bad = [r for r in rows if not r["valid"]]
    parts = ["# cut-in 표 지표 정의와 집계 (aggregate_cutin_table.py)", "",
             "생성 명령: `%s`" % argv_note, "",
             "## 1. 정의", "", "```", doc, "```", "",
             "## 2. 집계 (유효 런만, mean ± std)", "", markdown_table(agg), "",
             "j_avg / j_max 는 제어주기(0.2 s) 대역으로 다시 잰 값이고 괄호 안이 "
             "요청대로 잰 **원시** max|actual_jerk| 다.",
             "원시값은 이상적 액추에이터의 1틱 deadbeat 보정 때문에 측정 가속도가 "
             "명령 주위에서 틱마다 진동해서(세게 밟는 구간에서 명령 -2.19 → -1.21 인데",
             "측정 -3.98, -0.31, -3.89, +0.44 … -12.76) 40~300 m/s^3 까지 튄다 — "
             "명령 저크는 jerk_limit 1.5 m/s^3 에 묶여 있으니 제어기가 낸 값이 아니다.", "",
             "## 3. 제외된 런 (%d / %d) — 런 단위 상세는 table_cutin_runs.csv 의 "
             "valid, invalid_reason 열" % (len(bad), len(rows)), ""]
    if bad:
        buckets = collections.OrderedDict()
        for r in bad:
            key = (r["group"], r["policy_label"], r.get("ego_speed"),
                   re.sub(r"[0-9.]+", "*", r["invalid_reason"]))
            buckets.setdefault(key, []).append(r["run"])
        parts.append("| 그룹 | 제어기 | ego | 런 수 | 사유 |")
        parts.append("|---|---|---|---|---|")
        for key, runs in buckets.items():
            parts.append("| %s | %s | %s | %d | %s |"
                         % (key[0], key[1], key[2], len(runs), key[3]))
    else:
        parts.append("없음.")
    coll = [r for r in rows if r["valid"] and r.get("collisions_in_window")]
    ego_coll = [r for r in coll if r["ego_collision"]]
    parts += ["", "## 4. 충돌 (유효 런은 집계에 그대로 포함)", "",
              "**ego 충돌 %d건.**" % len(ego_coll)]
    for r in ego_coll:
        parts.append("- %s/%s/%s — %s" % (r["policy"], r["group"], r["run"],
                                          r["collision_pairs"]))
    bg = [r for r in coll if not r["ego_collision"]]
    if bg:
        parts += ["", "배경 차량끼리의 충돌(ego 무관)이 있는 유효 런 %d개:" % len(bg), "",
                  "| 충돌쌍 | 런 수 |", "|---|---|"]
        by_pair = collections.Counter(r["collision_pairs"] for r in bg)
        for pair, n in by_pair.most_common():
            parts.append("| %s | %d |" % (pair, n))
    tvl = [r for r in rows if r.get("tv_lead_contact")]
    parts += ["", "### TV–옛차선 앞차 접촉 (tv_lead_contact, %d런; 제외하지 않음)" % len(tvl),
              "", "cut-in 차량이 옛 차선 앞차를 횡방향 여유 ~0 m 로 스쳐 지나가는 구간이라 "
              "cm 단위 차이로 접촉이 잡힌다. ego 와는 무관하지만 그 런의 TV 궤적은 "
              "교란됐을 수 있으니 t_cross 이후 값을 읽을 때 참고할 것.", ""]
    if tvl:
        for r in tvl:
            parts.append("- %s/%s/%s (ego %s, TV %s)"
                         % (r["policy_label"], r["group"], r["run"],
                            r.get("ego_speed"), r.get("target_speed")))
    else:
        parts.append("없음.")
    if any(is_cutout_group(k[0]) for k in agg):
        cutout_rows = [r for r in rows if is_cutout_group(r["group"])]
        parts += ["", "## 5. cut-out", "",
                  "표 열 순서는 overleaf/table_cutout.tex 그대로 "
                  "(a_avg, j_avg, j_max, Δt_ant, v_avg) 이고 tex 블록은 "
                  "table_cutout_rows.tex 에 따로 쓴다. 런 단위 열은 "
                  "table_cutin_runs.csv 에 같이 들어간다.", "",
                  "### 5.1 정의", "", "```", doc_cutout.strip(), "```", "",
                  "### 5.2 집계 (유효 런만, mean ± std)", "",
                  markdown_table_cutout(agg), "",
                  "gap@trig / subLV 최소간격은 범퍼 간격 [m], 정상상태는 "
                  "settled_before_trigger 를 만족한 런의 비율이다.", ""]
        no_trig = [r for r in cutout_rows if r["valid"] and r.get("t_trigger") is None]
        no_out = [r for r in cutout_rows
                  if r["valid"] and r.get("t_trigger") is not None
                  and r.get("t_out") is None]
        not_inlane = [r for r in cutout_rows
                      if r["valid"] and r.get("lv_inlane_at_trigger") is False]
        parts += ["### 5.3 cut-out 이 성립하지 않은 유효 런", "",
                  "여기 걸린 런은 지표가 나오더라도 cut-out 을 잰 게 아니다. "
                  "lv_inlane_at_trigger=False 는 트리거 시점에 LV 가 이미 ego 차선 밖이라 "
                  "t_out 이 t_trigger 와 같아지고(Δt_ant 0.00) 시나리오 자체가 틀린 것이다.", ""]
        if no_trig or no_out or not_inlane:
            for r in no_trig:
                parts.append("- 트리거 없음: %s/%s/%s" % (r["policy"], r["group"], r["run"]))
            for r in no_out:
                parts.append("- 트리거는 걸렸지만 창 안에서 차선을 안 벗어남: %s/%s/%s"
                             % (r["policy"], r["group"], r["run"]))
            for r in not_inlane:
                parts.append("- 트리거 시점에 LV 가 ego 차선에 없음(lv_inlane_at_trigger=False): "
                             "%s/%s/%s (t_out-t_trigger=%s)"
                             % (r["policy"], r["group"], r["run"],
                                r.get("t_out_minus_trigger")))
        else:
            parts.append("없음.")
        contact = [r for r in cutout_rows if r.get("lv_sublv_contact")]
        parts += ["", "### 5.4 LV–subLV 접촉 (lv_sublv_contact, %d런; 제외하지 않음)"
                  % len(contact), ""]
        if contact:
            for r in contact:
                parts.append("- %s/%s/%s" % (r["policy_label"], r["group"], r["run"]))
        else:
            parts.append("없음.")
    open(str(path), "w").write("\n".join(parts) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--ego-speeds", default="", help="예: 14,17 (기본: 전부)")
    ap.add_argument("--groups", default="", help="예: 01_cutin_normal (기본: 전부)")
    ap.add_argument("--window-start-s", type=float, default=None,
                    help="채점 창의 앞을 제어 시작 이후 이 초수만큼 잘라낸다 "
                         "(기본: 자르지 않음). 3.0 이면 스폰 정렬 과도가 빠져 최대 열"
                         "(j_max)이 이벤트 구간 값과 일치한다. 평균 열은 수렴하지 않는다.")
    ap.add_argument("--window-s", type=float, default=None,
                    help="채점 창을 제어 시작 이후 이 초수로 자른다 (기본: 스윕 전체 25 s). "
                         "런마다 t0 가 14.4~14.6 s 라 17.5 는 절대 시각 약 32 s 에 해당한다.")
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    speeds = [float(v) for v in args.ego_speeds.split(",") if v.strip()]
    groups = [g for g in args.groups.split(",") if g.strip()]

    rows = []
    for pkl in sorted(root.rglob("scenario_result.pkl")):
        rel = pkl.parent.relative_to(root).parts
        if len(rel) != 3 or any(p.startswith("_") for p in rel):
            continue  # _discarded_* 같은 보관 디렉터리는 건너뛴다
        if groups and rel[1] not in groups:
            continue
        row = collect_run(rel[0], rel[1], pkl.parent, window_s=args.window_s,
                          window_start_s=args.window_start_s)
        if args.window_start_s is not None:
            base = collect_run(rel[0], rel[1], pkl.parent, window_s=args.window_s)
            for key in EVENT_KEYS:      # 이벤트 열은 절단 전 창에서 잰 값을 쓴다
                row[key] = base.get(key)
        if speeds and row.get("ego_speed") not in speeds:
            continue
        rows.append(row)
    if not rows:
        print("no runs under %s" % root)
        return 1

    agg = aggregate(rows)
    has_cutout = any(is_cutout_group(k[0]) for k in agg)
    write_runs_csv(root / "table_cutin_runs.csv", rows)
    write_summary_csv(root / "table_cutin_summary.csv", agg)
    write_tex(root / "table_cutin_rows.tex", agg)
    if has_cutout:
        write_cutout_tex(root / "table_cutout_rows.tex",
                         aggregate(rows, group_key=pooled_cutout_group))
    has_event = any(r.get("t_trigger") is not None for r in rows)
    if has_event:
        write_event_tex(root / "table_event_rows.tex", agg)
    note = "aggregate_cutin_table.py %s%s%s%s%s" % (
        root, " --ego-speeds " + args.ego_speeds if speeds else "",
        " --groups " + args.groups if groups else "",
        " --window-s %g" % args.window_s if args.window_s is not None else "",
        " --window-start-s %g" % args.window_start_s if args.window_start_s is not None else "")
    write_metrics_md(root / "METRICS.md", agg, rows, note)
    print("%d runs (%d valid) -> table_cutin_runs.csv, table_cutin_summary.csv, "
          "table_cutin_rows.tex, %sMETRICS.md in %s\n"
          % (len(rows), sum(1 for r in rows if r["valid"]),
             "table_cutout_rows.tex, " if has_cutout else "", root))
    print(markdown_table(agg))
    if has_cutout:
        print("\ncut-out:")
        print(markdown_table_cutout(agg))
    if has_event:
        print("\n이벤트 구간 [t_trigger-5 s, t_trigger+9 s] — 열은 위 표와 같다:")
        print(markdown_table(agg, event=True))
        if has_cutout:
            print("\ncut-out (이벤트 구간):")
            print(markdown_table_cutout(agg, event=True))
    bad = [r for r in rows if not r["valid"]]
    if bad:
        print("\n제외 %d런:" % len(bad))
        for r in bad:
            print("  %s/%s/%s ego=%s : %s" % (r["policy"], r["group"], r["run"],
                                              r.get("ego_speed"), r["invalid_reason"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
