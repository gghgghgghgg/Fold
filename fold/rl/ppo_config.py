def configure_ppo_rollout(config: dict, num_envs: int, horizon: int, target_steps_per_minibatch: int = 8) -> None:
    """Set rl-games rollout keys and choose a minibatch that divides the batch."""
    num_envs = int(num_envs)
    horizon = int(horizon)
    target_steps_per_minibatch = int(target_steps_per_minibatch)
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive; got {num_envs}.")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive; got {horizon}.")
    if target_steps_per_minibatch <= 0:
        raise ValueError(
            "target_steps_per_minibatch must be positive; "
            f"got {target_steps_per_minibatch}."
        )

    max_steps = min(horizon, target_steps_per_minibatch)
    steps_per_minibatch = max(steps for steps in range(1, max_steps + 1) if horizon % steps == 0)
    minibatch_size = num_envs * steps_per_minibatch
    batch_size = num_envs * horizon
    if batch_size % minibatch_size:
        raise RuntimeError(
            f"minibatch_size {minibatch_size} does not divide rollout batch {batch_size}."
        )

    config.pop("num_steps_per_env", None)
    config["horizon_length"] = horizon
    config["minibatch_size"] = minibatch_size
