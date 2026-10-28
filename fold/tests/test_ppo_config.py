import pytest

from fold.rl.ppo_config import configure_ppo_rollout


@pytest.mark.parametrize(
    ("num_envs", "horizon", "expected_minibatch"),
    [
        (8, 4, 32),
        (8, 16, 64),
        (8, 10, 40),
        (8, 3, 24),
    ],
)
def test_configure_ppo_rollout_sets_effective_horizon_and_divisible_minibatch(
    num_envs, horizon, expected_minibatch
):
    config = {"num_steps_per_env": 32}

    configure_ppo_rollout(config, num_envs, horizon)

    assert config["horizon_length"] == horizon
    assert config["minibatch_size"] == expected_minibatch
    assert (num_envs * horizon) % config["minibatch_size"] == 0
    assert "num_steps_per_env" not in config


@pytest.mark.parametrize(
    ("num_envs", "horizon"),
    [(0, 16), (8, 0)],
)
def test_configure_ppo_rollout_rejects_nonpositive_dimensions(num_envs, horizon):
    with pytest.raises(ValueError):
        configure_ppo_rollout({}, num_envs, horizon)
