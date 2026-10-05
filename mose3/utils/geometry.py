import torch


def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def fit_focal_from_bearings(bearings, conf=None, conf_thresh=0.1,
                            f_min=0.25, f_max=10.0, min_valid=256):
    """Per-video focal lengths (fx, fy) [B, 2] in pixels, least-squares fit to a bearing field [B, N, H, W, 2].

    The principal point is the image centre. Fits are clamped to [f_min, f_max] half-diagonals; a failed
    fit falls back to one half-diagonal. Inputs are detached.
    """
    bearings = bearings.detach()
    if conf is not None:
        conf = conf.detach()
    B, N, H, W, _ = bearings.shape
    # pixel offsets from the image centre
    px = (torch.arange(W, device=bearings.device, dtype=bearings.dtype) + 0.5 - 0.5 * W)
    py = (torch.arange(H, device=bearings.device, dtype=bearings.dtype) + 0.5 - 0.5 * H)
    px, py = torch.meshgrid(px, py, indexing='xy')
    half_diag = 0.5 * (H ** 2 + W ** 2) ** 0.5

    valid = torch.isfinite(bearings).all(dim=-1)
    if conf is not None:
        valid = valid & (torch.sigmoid(conf[..., 0]) > conf_thresh)
    w = valid.to(bearings.dtype)
    b = torch.nan_to_num(bearings, nan=0.0, posinf=0.0, neginf=0.0)

    n_valid = w.sum((1, 2, 3))
    # f_x = sum(w px^2) / sum(w px b_x); f_y likewise
    grid_sq = torch.stack([(w * px * px).sum((1, 2, 3)),
                           (w * py * py).sum((1, 2, 3))], dim=-1)
    proj = torch.stack([(w * px * b[..., 0]).sum((1, 2, 3)),
                        (w * py * b[..., 1]).sum((1, 2, 3))], dim=-1)

    f_raw = grid_sq / proj
    ok = torch.isfinite(f_raw) & (f_raw > 0) & (n_valid[:, None] >= min_valid)
    f_safe = torch.nan_to_num(f_raw, nan=half_diag, posinf=half_diag, neginf=half_diag)
    f = torch.where(ok, f_safe.clamp(f_min * half_diag, f_max * half_diag),
                    torch.full_like(f_safe, half_diag))
    return f
