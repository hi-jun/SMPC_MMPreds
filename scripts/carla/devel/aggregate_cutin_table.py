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
  충돌 판정과 모든 상대시각(t_trigger/t_cross/t_onset/t_pass)도 같은 t0 를 쓴다.
  **`_spawn_settle_log.sim_elapsed_s + _cruise_warmup_log.sim_elapsed_s` 를 t0 로
  쓰면 안 된다.** CARLA 의 time_s 는 서버 경과 시계라 시나리오 스폰 전에 이미 ~5 s
  가 지나 있다 — 저 합(2026-09-04 스윕에서 9.90 s)은 실제 제어 시작(첫 스텝 14.4~
  15.2 s)보다 4.4~5.3 s 이르다. 그걸 창의 시작으로 쓰면 모든 런에서 뒤쪽 4~5 s
  (회복 구간)가 잘리고 창 내용이 런마다 달라진다. 옛 aggregate_confchance.py 와
  RESULT.md 의 상대시각이 이 방식이라 이 스크립트 값과 4.4~5.3 s 어긋난다.

승차감 (측정값 actual_accel/actual_jerk, 창의 첫 샘플 1개 버림 — 첫 차분이
스폰 과도를 포함한다: analyze_acc_scenario_result.py 와 동일)
  a_avg = mean|actual_accel| [m/s^2],  j_avg = mean|actual_jerk|,
  j_max = max|actual_jerk| [m/s^3],  j_p99 = |jerk| 99퍼센타일, a_min = min accel.
  ※ **원시 j_avg/j_max 는 액추에이터 잡음이 지배한다.** 이상적 액추에이터의 1틱
  deadbeat 보정 때문에 측정 가속도가 명령 주위에서 틱마다 진동한다 — 세게 밟는
  구간(02_cutin_aggressive/aggressive_cutin_0017, chance0.6, ego 17)에서 명령이
  -2.19 → -1.21 로 매끈한데 측정값은 -3.98, -0.31, -3.89, +0.44, -2.20, +0.25,
  -12.76, +2.52 로 튄다(|actual_accel-accel_cmd| > 1.0 인 스텝이 그 런의 17%).
  명령 저크는 jerk_limit 1.5 m/s^3 에 묶여 있으니 저 값들은 제어기가 낸 게 아니다.
  그대로 쓰면 j_max 가 40~300 m/s^3, a_min 이 -12.8 m/s^2 (a_min 한계 -3.0) 로
  나와 승차감 비교가 불가능하다. 그래서 **제어 주기(0.2 s) 대역**으로 다시 계산한
    a_filt[k] = 4틱(0.2 s) 이동평균 = Δv/0.2s,  j_filt[k] = (a_filt[k+4]-a_filt[k])/0.2
  를 a_min_filt / j_avg_filt / j_max_filt 로 함께 내고 **LaTeX 표에는 filt 값을 쓴다**
  (0.05 s 틱 채터는 승객이 저크로 느끼지 않는다; 제어기는 0.2 s 마다만 갱신한다).
  위 런 기준 a_min -12.76→-3.57(명령 -2.79), j_avg 15.5→1.00(명령 0.81), j_max 306→20.8.
  a_avg 는 절대값 평균이라 잡음에 둔감해서(0.975 vs 명령 0.824) 원시값 그대로 쓴다.
  원시 j_avg / j_max / j_p99 / a_min 도 CSV·요약에 그대로 남는다.

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
  Δt_ant = t_cross - t_onset  (t_onset < t_cross 일 때만; 아니면 None → 표에 '-'.
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

Δt_ant (제어기 반응 선제성; cut-in 과 같은 뜻이되 기준점이 t_out 이다)
  t_settle_end = **t_out 전에 시작한** 마지막 "정상상태 구간"(|accel_cmd| < 0.15 가
    1.0 s 이상 연속)의 끝 시각. 그런 구간이 없으면 창 시작.
    ※ "t_out 전에 **끝나는**" 이 아니라 "t_out 전에 **시작한**" 이다. 반응이 아예
    없는 제어기(SCC 등)는 정상상태 구간이 t_out 을 걸쳐 이어지는데, 끝나는 시각으로
    걸러버리면 그 구간이 통째로 빠지고 훨씬 이른(트리거 전 과도) 구간이 기준이 돼
    Δt_ant 가 크게 나오는 가짜 선제성이 생긴다. 시작 시각으로 고르면 그 런은
    t_settle_end > t_out → t_onset > t_out → Δt_ant = 0.00 (선제성 없음) 이 된다.
    t_settle_end > t_out 인 런은 "t_out 까지 제어기가 아무 반응을 안 했다"는 뜻이다.
  t_onset = t_settle_end 이후 처음으로 2스텝 연속
    accel_cmd >= +0.3 m/s^2 (kind cutout_no_sublv: 비워진 차선으로 가속) 또는
    accel_cmd <= -0.3 m/s^2 (kind cutout_sublv: subLV 때문에 감속) 인 시각.
    쓴 부호는 onset_sign 열에 남긴다.
  Δt_ant = t_out - t_onset (t_onset < t_out 일 때). t_onset 이 t_out 이후면 **0.00**
    = 선제성 없음(table_cutout.tex 의 SCC 행 0.00 과 같은 뜻). 창 안에 onset 자체가
    없을 때만 None → 표에 '-'.
  ※ 정상상태를 기준점으로 잡는 이유: cut-out 은 트리거 전이 정속 추종 구간이라
    창 시작 직후의 스폰 과도를 onset 으로 잘못 집기 쉽다.

속도
  v_avg = **[t_trigger, 창끝] 의 ego 속도 평균** — table_cutout.tex 의 v_avg 열.
    (cut-in 쪽 v_avg 는 창 전체 평균이다. 같은 이름이지만 구간이 다르다.)
  v_avg_window = 창 전체 평균, v_avg_post = [t_out, 창끝] 평균.

subLV (kind cutout_sublv 만)
  min_bumper_gap_sublv = subLV 가 ego 차선에 있고 앞설 때의 최소 범퍼 간격.
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
CUTOUT_OUT_HOLD_STEPS = 2     # lane_id 한 샘플 튐 방지 (0.1 s)
SETTLE_ACCEL = 0.15
SETTLE_MIN_S = 1.0
SETTLE_LOOKBACK_S = 3.0
CUTOUT_CONTACT_PRE_S = 0.5
CUTOUT_CONTACT_POST_S = 2.0

SWEEP_PARAMS = ("ego_speed", "target_speed", "lane_change_distance",
                "lane_change_time_s", "target_lead_gap", "trigger_distance",
                "ego_gap_at_trigger")
TEX_METRICS = ("a_avg", "j_avg_filt", "j_max_filt", "delta_max", "delta_avg",
               "dt_ant", "T_rec")
TEX_METRICS_03 = ("v_avg", "a_min_filt", "passed", "t_pass")
TEX_METRICS_CUTOUT = ("a_avg", "j_avg_filt", "j_max_filt", "dt_ant", "v_avg")
RAW_OF_FILT = {"j_avg_filt": "j_avg", "j_max_filt": "j_max"}
AGG_METRICS = ("a_avg", "a_min", "a_min_filt", "j_avg", "j_avg_filt", "j_max",
               "j_max_filt", "j_p99", "delta_max", "delta_avg", "delta_avg_window",
               "min_bumper_gap", "t_cross_minus_trigger", "dt_ant", "T_rec",
               "v_avg", "v_min", "passed", "t_pass")
AGG_METRICS_CUTOUT = AGG_METRICS + (
    "t_out_minus_trigger", "v_avg_window", "v_avg_post", "gap_at_trigger",
    "v_ego_at_trigger", "v_lv_at_trigger", "min_bumper_gap_sublv",
    "settled_before_trigger")
SCENARIO_LABEL = {"01_cutin_normal": "Normal \\\\ Cut-in",
                  "02_cutin_aggressive": "Aggressive \\\\ Cut-in",
                  "03_no_cutin_decel": "No Cut-in \\\\ (adj. decel)",
                  "04_no_cutin_decel_onset": "No Cut-in \\\\ (decel onset 22/28/34)"}


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
                    cmd, ego_s, ego_x, ego_v, ego_lane_id):
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
                "lv_inlane_at_trigger"):
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
    onset = (cmd <= -CUTOUT_ONSET_ACCEL) if sublv_expected else (cmd >= CUTOUT_ONSET_ACCEL)
    i_from = 0 if settle_end is None else int(np.searchsorted(t, settle_end))
    i_on = _first_run_start(onset[i_from:], ONSET_HOLD_STEPS)
    t_onset = None if i_on is None else float(t[i_from + i_on])
    row["t_onset"] = None if t_onset is None else round(t_onset - t0, 3)
    row["dt_ant"] = None
    if t_onset is not None and out_abs is not None:
        row["dt_ant"] = round(out_abs - t_onset, 3) if t_onset < out_abs else 0.0

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


def collect_run(policy, group, run_dir):
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
    w1 = w0 + float(sweep.get("max_sim_time_s") or DEFAULT_WINDOW_S)
    keep = np.nonzero(all_t <= w1 + 1e-9)[0]
    steps = [steps[i] for i in keep]
    t = all_t[keep]
    t0 = w0  # 제어 시작 = 시간 기준. 서버 경과 시계라 스폰 전 ~5 s 가 이미 지나 있다
    row["t0_sim"] = round(t0, 3)
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
    # t_onset 이 0 에 가까우면 창 시작 전에 이미 밟고 있었을 수 있다 = Δt_ant 좌측 절단
    row["t_onset"] = None if t_onset is None else round(t_onset - t0, 3)
    row["dt_ant"] = (round(t_cross - t_onset, 3)
                     if t_cross is not None and t_onset is not None and t_onset < t_cross
                     else None)

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
                        cmd, ego_s, ego_x, ego_v, ego_lane_id)
    for key in ("a_avg", "a_min", "a_min_filt", "j_avg", "j_avg_filt", "j_max",
                "j_max_filt", "j_p99", "delta_max", "delta_avg",
                "delta_avg_window", "min_bumper_gap", "v_avg", "v_min"):
        if row.get(key) is not None:
            row[key] = round(row[key], 4)
    return row


def aggregate(rows):
    """(group, policy) 셀별 metric -> (mean, std, n)."""
    cells = collections.OrderedDict()
    for row in rows:
        if not row["valid"]:
            continue
        cells.setdefault((row["group"], row["policy"]), []).append(row)
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
             "% j_avg/j_max 는 제어주기(0.2 s) 대역으로 다시 잰 j_avg_filt/j_max_filt 다 —",
             "% 원시 actual_jerk 는 1틱 액추에이터 채터(|j| 40~300)가 지배한다."]
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
             "% a_avg, j_avg, j_max, dt_ant, v_avg (셀 평균)",
             "% j_avg/j_max 는 제어주기(0.2 s) 대역으로 다시 잰 j_avg_filt/j_max_filt,",
             "% dt_ant = t_out - t_onset (선제성 없으면 0.00), v_avg 는 [t_trigger, 창끝] 평균."]
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


def markdown_table_cutout(agg):
    head = ("| 시나리오 | 제어기 | n | a_avg | j_avg 필터(원시) | j_max 필터(원시) "
            "| Δt_ant | v_avg | t_out-t_trig | gap@trig | 정상상태 | δ_max "
            "| subLV 최소간격 | ego충돌 | LV-subLV접촉 |")
    lines = [head, "|" + "---|" * 15]
    for (group, _policy), stats in agg.items():
        if not is_cutout_group(group):
            continue
        cells = []
        for metric in TEX_METRICS_CUTOUT:
            mean, std, n = stats[metric]
            cell = ("-" if mean is None
                    else "%.2f ± %.2f%s" % (mean, std, "" if n == stats["_n_runs"]
                                            else " (n=%d)" % n))
            raw = RAW_OF_FILT.get(metric)
            if raw is not None and stats[raw][0] is not None:
                cell += " (%.2f)" % stats[raw][0]
            cells.append(cell)
        for metric in ("t_out_minus_trigger", "gap_at_trigger",
                       "settled_before_trigger", "delta_max", "min_bumper_gap_sublv"):
            mean = stats[metric][0]
            cells.append("-" if mean is None else "%.2f" % mean)
        lines.append("| %s | %s | %d | %s | %d | %d |"
                     % (group, stats["_label"], stats["_n_runs"], " | ".join(cells),
                        stats["_ego_collisions"], stats["_tv_lead_contacts"]))
    return "\n".join(lines)


def markdown_table(agg):
    head = ("| 시나리오 | 제어기 | n | a_avg | j_avg 필터(원시) | j_max 필터(원시) "
            "| δ_max | δ_avg | Δt_ant | T_rec | ego충돌 | TV-앞차접촉 |")
    lines = [head, "|" + "---|" * 12]
    for (group, _policy), stats in agg.items():
        if is_cutout_group(group):
            continue
        cells = []
        for metric in TEX_METRICS:
            mean, std, n = stats[metric]
            cell = ("-" if mean is None
                    else "%.2f ± %.2f%s" % (mean, std, "" if n == stats["_n_runs"]
                                            else " (n=%d)" % n))
            raw = RAW_OF_FILT.get(metric)
            if raw is not None and stats[raw][0] is not None:
                cell += " (%.2f)" % stats[raw][0]
            cells.append(cell)
        lines.append("| %s | %s | %d | %s | %d | %d |"
                     % (group, stats["_label"], stats["_n_runs"], " | ".join(cells),
                        stats["_ego_collisions"], stats["_tv_lead_contacts"]))
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
        row = collect_run(rel[0], rel[1], pkl.parent)
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
        write_cutout_tex(root / "table_cutout_rows.tex", agg)
    note = "aggregate_cutin_table.py %s%s%s" % (
        root, " --ego-speeds " + args.ego_speeds if speeds else "",
        " --groups " + args.groups if groups else "")
    write_metrics_md(root / "METRICS.md", agg, rows, note)
    print("%d runs (%d valid) -> table_cutin_runs.csv, table_cutin_summary.csv, "
          "table_cutin_rows.tex, %sMETRICS.md in %s\n"
          % (len(rows), sum(1 for r in rows if r["valid"]),
             "table_cutout_rows.tex, " if has_cutout else "", root))
    print(markdown_table(agg))
    if has_cutout:
        print("\ncut-out:")
        print(markdown_table_cutout(agg))
    bad = [r for r in rows if not r["valid"]]
    if bad:
        print("\n제외 %d런:" % len(bad))
        for r in bad:
            print("  %s/%s/%s ego=%s : %s" % (r["policy"], r["group"], r["run"],
                                              r.get("ego_speed"), r["invalid_reason"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
