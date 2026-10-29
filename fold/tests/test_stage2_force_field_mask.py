import torch

from stage2.force_field_mask import hand_proximity_mask, select_hard_mask_topk_with_fallback
from stage2.realtime_force_field import realtime_force_field_config_from_payload


def test_hand_proximity_mask_uses_inclusive_radius_and_empty_hand():
    anchors = torch.tensor([[[0.05, 0, 0], [0.051, 0, 0]]])
    hand = torch.zeros((1, 1, 3))
    assert hand_proximity_mask(anchors, hand, 0.05).tolist() == [[True, False]]
    assert not hand_proximity_mask(anchors, hand[:, :0], 0.05).any()


def test_hard_mask_excludes_non_candidates_even_if_they_score_higher():
    local_scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    global_scores = torch.tensor([[100.0, 90.0, 80.0, 70.0]])
    nearby = torch.tensor([[False, True, True, False]])
    selected = select_hard_mask_topk_with_fallback(local_scores, nearby, 3, global_scores)
    assert selected.tolist() == [[False, True, True, False]]


def test_empty_rows_fall_back_globally_but_nonempty_rows_never_fill_globally():
    local_scores = torch.tensor([[0.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
    global_scores = torch.tensor([[9.0, 8.0, 7.0], [1.0, 4.0, 3.0]])
    candidates = torch.tensor([[False, True, False], [False, False, False]])
    selected = select_hard_mask_topk_with_fallback(local_scores, candidates, 2, global_scores)
    assert selected.tolist() == [[False, True, False], [False, True, True]]


def test_zero_topk_selects_all_candidates_or_all_global_on_empty_rows():
    local_scores = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    global_scores = torch.tensor([[9.0, 8.0, 7.0], [1.0, 4.0, 3.0]])
    candidates = torch.tensor([[True, False, True], [False, False, False]])
    selected = select_hard_mask_topk_with_fallback(local_scores, candidates, 0, global_scores)
    assert selected.tolist() == [[True, False, True], [True, True, True]]


def test_legacy_saved_field_config_keeps_global_anchor_behavior():
    config = realtime_force_field_config_from_payload({
        "data_path": "field.npz",
        "encoder_checkpoint": "encoder.pt",
        "frame_offset": 30,
        "wrench": {},
    })
    assert config.hand_mask_radius == 0.0


def test_new_saved_field_config_restores_hard15_radius():
    config = realtime_force_field_config_from_payload({
        "data_path": "field.npz",
        "encoder_checkpoint": "encoder.pt",
        "frame_offset": 30,
        "hand_mask_radius": 0.15,
        "wrench": {},
    })
    assert config.hand_mask_radius == 0.15
