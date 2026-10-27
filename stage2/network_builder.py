from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from collections.abc import Sequence

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder
from stage1.network_builder import Stage1SemanticNetwork


def _activation_factory(name: str):
    name = str(name).lower()
    if name == "elu":
        return nn.ELU
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name in {"silu", "swish"}:
        return nn.SiLU
    if name in {"identity", "none"}:
        return nn.Identity
    raise ValueError(f"Unsupported stage2 activation: {name}")


def _build_field_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation_name: str,
    use_layer_norm: bool,
) -> nn.Module:
    if len(hidden_dims) == 0:
        return nn.Identity()

    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    activation_ctor = _activation_factory(activation_name)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        if use_layer_norm:
            layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(activation_ctor())
        in_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class SemanticFlatObsEncoder(nn.Module):
    def __init__(
        self,
        obs_spec: dict,
        field_hidden_units: Sequence[int],
        activation_name: str,
        use_layer_norm: bool,
    ):
        super().__init__()
        if len(field_hidden_units) == 0:
            raise ValueError("stage2_field_units must contain at least one hidden size.")

        self.obs_dim = int(obs_spec["obs_dim"])
        self.fields = list(obs_spec["fields"])
        self.output_dim = len(self.fields) * int(field_hidden_units[-1])
        self.encoders = nn.ModuleDict()

        for field in self.fields:
            self.encoders[field["name"]] = _build_field_mlp(
                input_dim=int(field["dim"]),
                hidden_dims=field_hidden_units,
                activation_name=activation_name,
                use_layer_norm=use_layer_norm,
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected obs dim {self.obs_dim}, got {obs.shape[-1]}.")

        encoded_fields = []
        for field in self.fields:
            field_obs = obs[:, field["start"] : field["end"]]
            encoded_fields.append(self.encoders[field["name"]](field_obs))
        return torch.cat(encoded_fields, dim=-1)


def _extract_a2c_network_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Unexpected checkpoint format at {checkpoint_path}")
    model_state = ckpt["model"]
    prefix = "a2c_network."
    extracted = {k[len(prefix) :]: v for k, v in model_state.items() if k.startswith(prefix)}
    if not extracted:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a2c_network weights")
    return extracted


class Stage2ResidualNetwork(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop("actions_num")
        self.value_size = kwargs.pop("value_size", 1)
        self.num_seqs = kwargs.pop("num_seqs", 1)

        NetworkBuilder.BaseNetwork.__init__(self)
        self.load(params)

        if self.separate:
            raise NotImplementedError("stage2 residual network only supports separate=False.")

        obs_spec = params.get("stage2_obs_spec")
        if obs_spec is None:
            raise ValueError("Missing stage2_obs_spec in network config.")

        base_network_config = deepcopy(params.get("base_network_config"))
        if base_network_config is None:
            raise ValueError("Missing base_network_config in stage2 network config.")
        base_checkpoint = params.get("base_checkpoint")
        if not base_checkpoint:
            raise ValueError("Missing base_checkpoint in stage2 network config.")

        self.architecture = params.get("stage2_architecture", "semantic")
        self.residual_scale = float(params.get("stage2_residual_scale", 0.1))
        if self.architecture == "semantic":
            field_hidden_units = params.get("stage2_field_units", [128, 64])
            activation_name = params.get("stage2_encoder_activation", self.activation)
            use_layer_norm = params.get("stage2_use_layer_norm", True)
            encoder_obs_spec = obs_spec
            if obs_spec.get("actor_groups"):
                encoder_obs_spec = {**obs_spec, "fields": obs_spec["actor_groups"]}
            self.semantic_encoder = SemanticFlatObsEncoder(
                obs_spec=encoder_obs_spec,
                field_hidden_units=field_hidden_units,
                activation_name=activation_name,
                use_layer_norm=use_layer_norm,
            )
            encoder_dim = self.semantic_encoder.output_dim
        elif self.architecture == "maniptrans":
            self.semantic_encoder = nn.Identity()
            encoder_dim = int(obs_spec["obs_dim"])
        else:
            raise ValueError(f"Unsupported stage2 architecture: {self.architecture}")
        trunk_out_dim = encoder_dim if len(self.units) == 0 else int(self.units[-1])

        if len(self.units) == 0:
            self.actor_mlp = nn.Identity()
        else:
            self.actor_mlp = self._build_mlp(
                input_size=encoder_dim,
                units=self.units,
                activation=self.activation,
                norm_func_name=self.normalization,
                dense_func=torch.nn.Linear,
                d2rl=self.is_d2rl,
                norm_only_first_layer=self.norm_only_first_layer,
            )

        self.value = self._build_value_layer(trunk_out_dim, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        if not self.is_continuous:
            raise NotImplementedError("stage2 residual network only supports continuous control.")

        self.residual_mu = torch.nn.Linear(trunk_out_dim, actions_num)
        self.residual_mu_act = self.activations_factory.create(self.space_config["mu_activation"])
        self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])
        sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
        if self.fixed_sigma:
            self.sigma = nn.Parameter(
                torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.sigma = torch.nn.Linear(trunk_out_dim, actions_num)

        self.base_obs_dim = int(base_network_config["stage1_obs_spec"]["obs_dim"])
        self.base_policy = Stage1SemanticNetwork(
            base_network_config,
            actions_num=actions_num,
            value_size=self.value_size,
            num_seqs=self.num_seqs,
        )
        self.base_policy.load_state_dict(_extract_a2c_network_state(base_checkpoint), strict=True)
        self.base_policy.eval()
        for param in self.base_policy.parameters():
            param.requires_grad = False

        mlp_init = self.init_factory.create(**self.initializer)
        for module in self.semantic_encoder.modules():
            if isinstance(module, nn.Linear):
                mlp_init(module.weight)
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
        for module in self.actor_mlp.modules():
            if isinstance(module, nn.Linear):
                mlp_init(module.weight)
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
        if isinstance(self.value, nn.Linear):
            mlp_init(self.value.weight)
            if getattr(self.value, "bias", None) is not None:
                torch.nn.init.zeros_(self.value.bias)

        torch.nn.init.zeros_(self.residual_mu.weight)
        if getattr(self.residual_mu, "bias", None) is not None:
            torch.nn.init.zeros_(self.residual_mu.bias)

        if self.fixed_sigma:
            sigma_init(self.sigma)
        else:
            sigma_init(self.sigma.weight)
            if getattr(self.sigma, "bias", None) is not None:
                torch.nn.init.zeros_(self.sigma.bias)

    def forward(self, obs_dict):
        if isinstance(obs_dict, dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
        else:
            obs = obs_dict
            states = None

        if obs.shape[-1] < self.base_obs_dim:
            raise ValueError(f"Stage2 obs dim {obs.shape[-1]} smaller than base obs dim {self.base_obs_dim}.")

        base_obs = obs[:, : self.base_obs_dim]
        with torch.no_grad():
            base_mu, _, _, _ = self.base_policy({"obs": base_obs, "rnn_states": states})

        encoded = self.semantic_encoder(obs)
        trunk = self.actor_mlp(encoded)
        residual_mu = self.residual_mu_act(self.residual_mu(trunk))
        mu = torch.clamp(base_mu + self.residual_scale * residual_mu, -1.0, 1.0)
        value = self.value_act(self.value(trunk))
        if self.fixed_sigma:
            logstd = self.sigma_act(self.sigma)
        else:
            logstd = self.sigma_act(self.sigma(trunk))

        return (
            torch.nan_to_num(mu),
            torch.nan_to_num(logstd),
            torch.nan_to_num(value),
            states,
        )


class Stage2ResidualBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        return Stage2ResidualNetwork(self.params, **kwargs)
