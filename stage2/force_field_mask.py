from __future__ import annotations

import torch


def hand_proximity_mask(
    anchor_positions_world: torch.Tensor,
    hand_points_world: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Return anchors within ``radius`` of any current hand landmark."""
    if anchor_positions_world.ndim != 3 or anchor_positions_world.shape[-1] != 3:
        raise ValueError("anchor_positions_world must have shape (B, A, 3)")
    if hand_points_world.ndim != 3 or hand_points_world.shape[-1] != 3:
        raise ValueError("hand_points_world must have shape (B, H, 3)")
    if anchor_positions_world.shape[0] != hand_points_world.shape[0]:
        raise ValueError("Anchor and hand batches must match")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if hand_points_world.shape[1] == 0:
        return torch.zeros(anchor_positions_world.shape[:2], dtype=torch.bool,
                           device=anchor_positions_world.device)
    distances = torch.cdist(anchor_positions_world, hand_points_world)
    return distances.amin(dim=-1) <= radius


def select_hard_mask_topk_with_fallback(
    local_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    active_top_k: int,
    global_scores: torch.Tensor,
) -> torch.Tensor:
    """Select top-k local anchors; use global top-k only if a row has no candidates."""
    if local_scores.ndim != 2 or candidate_mask.shape != local_scores.shape:
        raise ValueError("local_scores and candidate_mask must have matching (B, A) shapes")
    if global_scores.shape != local_scores.shape:
        raise ValueError("global_scores must match local_scores shape")
    if active_top_k < 0:
        raise ValueError("active_top_k must be non-negative")
    candidate_mask = candidate_mask.to(device=local_scores.device, dtype=torch.bool)
    batch_size, num_anchors = local_scores.shape
    selected = torch.zeros_like(candidate_mask)
    if num_anchors == 0:
        return selected
    empty_rows = ~candidate_mask.any(dim=-1)
    if active_top_k == 0:
        selected.copy_(candidate_mask)
    else:
        quota = min(active_top_k, num_anchors)
        candidate_scores = local_scores.masked_fill(~candidate_mask, -torch.inf)
        candidate_count = candidate_mask.sum(dim=-1)
        candidate_quota = candidate_count.clamp_max(quota)
        local_order = candidate_scores.argsort(dim=-1, descending=True)
        ranks = torch.arange(num_anchors, device=local_scores.device).expand(batch_size, -1)
        local_selected = torch.zeros_like(selected)
        local_selected.scatter_(1, local_order, ranks < candidate_quota[:, None])
        selected |= local_selected
        global_order = global_scores.argsort(dim=-1, descending=True)
        global_selected = torch.zeros_like(selected)
        global_selected.scatter_(1, global_order, ranks < quota)
        selected |= global_selected & empty_rows[:, None]
    if active_top_k == 0:
        selected[empty_rows] = True
    return selected
