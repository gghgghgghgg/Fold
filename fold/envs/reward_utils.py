import os
import torch
from fold.envs.math_utils import quat_conjugate, quat_mul, matrix_from_quat

@torch.jit.script
def rotation_distance(object_rot, target_rot):
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention

@torch.jit.script
def position_distance(object_pos, target_pos): # (B, num_links, 3) -> (B, num_links)
    return torch.norm(object_pos - target_pos, p=2, dim=-1)

@torch.jit.script
def chamfer_distance(pts1, pts2, pts1_valid, pts2_valid):
    """
    pts1: (B, N1, 3), pts2: (B, N2, 3)
    pts1_valid: (B, N1), pts2_valid: (B, N2) -> does not consider invalid points in returned distance
    return: (B)
    """

    dist1 = torch.cdist(pts1, pts2, p=2.0) # shape (B, N1, N2)
    dist2 = torch.cdist(pts2, pts1, p=2.0)  # shape (B, N2, N1)
    pts2_valid_expanded = pts2_valid.unsqueeze(1).expand(-1, pts1.shape[1], -1) # shape (B, N1, N2)
    dist1[~pts2_valid_expanded] = 100.0

    pts1_valid_expanded = pts1_valid.unsqueeze(1).expand(-1, pts2.shape[1], -1)  # shape (B, N2, N1)
    dist2[~pts1_valid_expanded] = 100.0

    min_dists1 = torch.min(dist1, dim=-1).values # shape (B, N1)
    min_dists2 = torch.min(dist2, dim=-1).values  # shape (B, N2)
    chamfer_forward = torch.zeros(min_dists1.shape[0], device=min_dists1.device)
    chamfer_backward = torch.zeros(min_dists2.shape[0], device=min_dists2.device)

    num_valids1 = pts1_valid.sum(dim=-1)
    num_valids2 = pts2_valid.sum(dim=-1)
    # if both pts have zero valid points, set chamfer to zero
    both_zero = (num_valids1 == 0) & (num_valids2 == 0)
    chamfer_forward[both_zero] = 0.0
    chamfer_backward[both_zero] = 0.0

    # if only one of them has zero valid points, set chamfer to max_distance
    one_zero = (num_valids1 == 0) ^ (num_valids2 == 0)
    chamfer_forward[one_zero] = 100.0
    chamfer_backward[one_zero] = 100.0

    # if both pts have some valid points, set chamfer to the min distance of only valid points
    both_has_valid = (num_valids1 > 0) & (num_valids2 > 0)
    if both_has_valid.sum() > 0:
        valid_sums1 = torch.sum(min_dists1 * pts1_valid, dim=-1)
        valid_sums2 = torch.sum(min_dists2 * pts2_valid, dim=-1)
        chamfer_forward[both_has_valid] = (valid_sums1 / num_valids1)[both_has_valid]
        chamfer_backward[both_has_valid] = (valid_sums2 / num_valids2)[both_has_valid]
    return (chamfer_forward + chamfer_backward)/2.0 # shape (B)


@torch.jit.script
def transform_contact(
    contact_positions, # shape (N, T, 3)
    new_frame, # shape (N, 7)
):
    new_quat = new_frame[:, 3:]
    matrices = matrix_from_quat(new_quat) # shape (N, 3, 3)
    offsets = new_frame[:, :3].unsqueeze(1)
    transformed = contact_positions - offsets
    transformed = torch.bmm(transformed, matrices.transpose(1, 2))
    return transformed # shape (N, T, 3)
