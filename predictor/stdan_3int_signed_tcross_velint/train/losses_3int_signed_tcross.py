from __future__ import annotations

import torch


def integrate_velocity_distribution(v_pred, dt=0.1):
    """Convert predicted future velocity distribution to relative position distribution."""
    if v_pred.shape[-1] == 2:
        return torch.cumsum(v_pred * dt, dim=0)

    mu_v = v_pred[:, :, 0:2]
    inv_sig_v = v_pred[:, :, 2:4]
    rho_v = v_pred[:, :, 4].clamp(-0.999, 0.999)

    mu_p = torch.cumsum(mu_v * dt, dim=0)
    v_variance = 1.0 / (torch.pow(inv_sig_v, 2) + 1e-6)
    p_variance = torch.cumsum(v_variance * (dt ** 2), dim=0)
    inv_sig_p = 1.0 / (torch.sqrt(p_variance) + 1e-6)

    sigma_v_x = 1.0 / (inv_sig_v[:, :, 0] + 1e-6)
    sigma_v_y = 1.0 / (inv_sig_v[:, :, 1] + 1e-6)
    cov_v_xy = rho_v * sigma_v_x * sigma_v_y
    cov_p_xy = torch.cumsum(cov_v_xy * (dt ** 2), dim=0)
    rho_p = cov_p_xy * inv_sig_p[:, :, 0] * inv_sig_p[:, :, 1]
    rho_p = rho_p.clamp(-0.999, 0.999)
    return torch.cat([mu_p, inv_sig_p, rho_p.unsqueeze(-1)], dim=-1)


def integrate_velocity_modes(modes, dt=0.1):
    if isinstance(modes, (list, tuple)):
        return [integrate_velocity_distribution(mode, dt=dt) for mode in modes]
    return integrate_velocity_distribution(modes, dt=dt)


def velocity_metrics(v_pred, fut_pos, mask, dt=0.1):
    valid = mask[:, :, 0] > 0.5
    if v_pred.shape[-1] >= 5:
        pred_vel = v_pred[:, :, 0:2]
    else:
        pred_vel = v_pred[:, :, 0:2]
    gt_vel = torch.zeros_like(pred_vel)
    gt_vel[0] = fut_pos[0] / dt
    if fut_pos.shape[0] > 1:
        gt_vel[1:] = (fut_pos[1:] - fut_pos[:-1]) / dt
    diff = pred_vel - gt_vel
    speed_error = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=-1) + 1e-9)
    mae = torch.sum(speed_error * valid) / torch.sum(valid).clamp_min(1.0)
    mse = torch.sum(torch.sum(torch.pow(diff, 2), dim=-1) * valid) / torch.sum(valid).clamp_min(1.0)
    return mae, mse


def masked_nll(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    mu_x = y_pred[:, :, 0]
    mu_y = y_pred[:, :, 1]
    sig_x = y_pred[:, :, 2]
    sig_y = y_pred[:, :, 3]
    rho = y_pred[:, :, 4].clamp(-0.999, 0.999)
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out = 0.5 * torch.pow(ohr, 2) * (
        torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2)
        + torch.pow(sig_y, 2) * torch.pow(y - mu_y, 2)
        - 2 * rho * sig_x * sig_y * (x - mu_x) * (y - mu_y)
    ) - torch.log(sig_x * sig_y * ohr + 1e-9) + 1.8379
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    denom = torch.sum(mask).clamp_min(1.0)
    return torch.sum(acc * mask) / denom


def masked_mse(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    out = torch.pow(y_gt[:, :, 0] - y_pred[:, :, 0], 2) + torch.pow(y_gt[:, :, 1] - y_pred[:, :, 1], 2)
    acc[:, :, 0] = out
    acc[:, :, 1] = out
    denom = torch.sum(mask).clamp_min(1.0)
    return torch.sum(acc * mask) / denom


def classification_loss(prob, one_hot_target):
    value = torch.log(torch.sum(prob * one_hot_target, dim=-1) + 1e-9)
    return -torch.mean(value)


def endpoint_loss(y_pred, y_gt, mask):
    valid = mask[-1, :, 0] > 0.5
    if not torch.any(valid):
        return y_pred.sum() * 0.0
    diff = y_pred[-1, valid, 0:2] - y_gt[-1, valid, 0:2]
    return torch.mean(torch.sum(torch.pow(diff, 2), dim=-1))


def t_cross_loss(t_cross_pred, t_cross_target, t_cross_mask):
    mask = t_cross_mask.view_as(t_cross_pred)
    valid_count = torch.sum(mask)
    if valid_count.item() == 0:
        return t_cross_pred.sum() * 0.0
    return torch.sum(torch.pow(t_cross_pred - t_cross_target.view_as(t_cross_pred), 2) * mask) / valid_count


def trajectory_metrics(y_pred, y_gt, mask):
    valid = mask[:, :, 0] > 0.5
    dist = torch.sqrt(torch.sum(torch.pow(y_pred[:, :, 0:2] - y_gt[:, :, 0:2], 2), dim=-1) + 1e-9)
    ade = torch.sum(dist * valid) / torch.sum(valid).clamp_min(1.0)
    final_valid = valid[-1]
    fde = torch.sum(dist[-1] * final_valid) / torch.sum(final_valid).clamp_min(1.0)
    return ade, fde


def t_cross_metrics(t_cross_pred, t_cross_target, t_cross_mask, norm_denominator):
    mask = t_cross_mask.view_as(t_cross_pred)
    valid_count = torch.sum(mask)
    if valid_count.item() == 0:
        zero = t_cross_pred.sum() * 0.0
        return zero, zero, zero
    pred_raw = t_cross_pred * norm_denominator
    target_raw = t_cross_target.view_as(t_cross_pred) * norm_denominator
    diff = (pred_raw - target_raw) * mask
    bias = torch.sum(diff) / valid_count
    mae = torch.sum(torch.abs(diff)) / valid_count
    mse = torch.sum(torch.pow(diff, 2)) / valid_count
    return mae, mse, bias
