from types import SimpleNamespace

import torch

from stage2.network_builder import SemanticFlatObsEncoder
from stage2.obs_spec import build_stage2_obs_spec


def test_force_field_is_an_independent_exact_slice():
    obs_spec = {
        "obs_dim": 6,
        "fields": [
            {"name": "original_a", "start": 0, "end": 2, "dim": 2},
            {"name": "original_b", "start": 2, "end": 3, "dim": 1},
            {
                "name": "force_command_realtime_force_field",
                "start": 3,
                "end": 6,
                "dim": 3,
            },
        ],
        "actor_groups": [
            {"name": "overlapping", "start": 0, "end": 6, "dim": 6},
        ],
    }
    encoder = SemanticFlatObsEncoder(
        obs_spec,
        field_hidden_units=[1],
        activation_name="identity",
        use_layer_norm=False,
    )
    for field_encoder in encoder.encoders.values():
        torch.nn.init.ones_(field_encoder[0].weight)
        torch.nn.init.zeros_(field_encoder[0].bias)

    encoded = encoder(torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]]))

    assert list(encoder.encoders) == [
        "original_a",
        "original_b",
        "force_command_realtime_force_field",
    ]
    assert encoder.output_dim == 3
    torch.testing.assert_close(encoded, torch.tensor([[3.0, 4.0, 56.0]]))


def test_stage2_spec_appends_force_field_without_actor_groups():
    obs_dim_info = {
        "qpos_target_dim": 1,
        "qpos_dim": 1,
        "qvel_dim": 1,
        "kpt_dim": 1,
        "wrist_dim": 1,
    }
    env = SimpleNamespace(
        robots={
            "left": SimpleNamespace(obs_dim_info=obs_dim_info),
            "right": SimpleNamespace(obs_dim_info=obs_dim_info),
        },
        object_names=["object"],
        objects={"object": SimpleNamespace(n_links=1, num_joints=1)},
        observe_tip_dist=False,
        observe_contact_force=False,
        realtime_force_field_command=SimpleNamespace(latent_dim=128),
        obs_dim=161,
    )

    obs_spec = build_stage2_obs_spec(env)
    force_field = obs_spec["fields"][-1]

    assert "actor_groups" not in obs_spec
    assert force_field == {
        "name": "force_command_realtime_force_field",
        "group": "force_command",
        "source": "environment",
        "field": "realtime_force_field",
        "start": 33,
        "end": 161,
        "dim": 128,
    }
    assert all(
        current["end"] == following["start"]
        for current, following in zip(obs_spec["fields"], obs_spec["fields"][1:])
    )
