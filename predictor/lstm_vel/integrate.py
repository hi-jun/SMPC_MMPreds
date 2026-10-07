import torch


def integrate_velocity(pred_vel, dt):
    """(Steps, B, 2) 속도(ft/s) -> 현재 위치 기준 상대 위치(ft).

    학습 시 GT와 동일한 규약: fut_pos[k] = sum_{i<=k} v[i] * dt
    """
    return torch.cumsum(pred_vel * dt, dim=0)
