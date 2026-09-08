import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ace_ego_0.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from ace_ego_0.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from ace_ego_0.model.modules.urdf.fusion_module import GatedUrdfTokenFusion
from ace_ego_0.model.modules.urdf.robot_urdf_registry import (
    get_robot_urdf_spec,
    list_supported_robot_names,
    resolve_robot_name,
    resolve_urdf_spec_overrides,
)
from ace_ego_0.model.modules.urdf.urdf_cache import load_or_build_urdf_graph_cache
from ace_ego_0.model.modules.urdf.urdf_encoder import UrdfGraphEncoder

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

logger = logging.getLogger(__name__)
_LOGGED_UNKNOWN_URDF_ROBOTS: set[str] = set()
_LOGGED_LEARNABLE_URDF_ROBOTS: set[str] = set()


def _is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _resolve_action_horizon(action_config) -> int:
    """Resolve action horizon from config and enforce consistency."""
    configured_horizon = getattr(action_config, "action_horizon", None)
    future_window = int(getattr(action_config, "future_action_window_size", 0))
    past_window = int(getattr(action_config, "past_action_window_size", 0))
    derived_horizon = future_window + past_window + 1

    if configured_horizon is None:
        return derived_horizon

    configured_horizon = int(configured_horizon)
    if configured_horizon != derived_horizon:
        raise ValueError(
            "Inconsistent action horizon in action_model config: "
            f"`action_horizon`={configured_horizon}, "
            f"`future_action_window_size + past_action_window_size + 1`={derived_horizon}."
        )
    return configured_horizon


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


def _resolve_state_injection_mode(action_config) -> str:
    """Resolve one canonical state injection mode with backward-compatible aliases."""
    explicit_mode = action_config.get("state_injection_mode", None)
    if explicit_mode is None:
        state_design_preset = action_config.get("state_design_preset", None)
        if state_design_preset is not None and str(state_design_preset).lower() in {"rdt2", "rdt2_like", "rdt2_v1"}:
            explicit_mode = "timestep_conditioning"

    if explicit_mode is not None:
        normalized_mode = str(explicit_mode).lower()
        alias_map = {
            "none": "disabled",
            "prepend_token": "token",
            "raw_state_token": "token",
            "rdt2": "timestep_conditioning",
            "rdt2_like": "timestep_conditioning",
        }
        normalized_mode = alias_map.get(normalized_mode, normalized_mode)
        valid_modes = {"disabled", "token", "timestep_conditioning"}
        if normalized_mode not in valid_modes:
            raise ValueError(
                "Unsupported `framework.action_model.state_injection_mode`: "
                f"{explicit_mode!r}. Expected one of {sorted(valid_modes)}."
            )
        return normalized_mode

    if bool(action_config.get("enable_state_timestep_conditioning", False)):
        return "timestep_conditioning"
    if bool(action_config.get("enable_state_input", False)):
        return "token"
    return "disabled"


def _resolve_urdf_unknown_robot_policy(action_config) -> str:
    """Resolve how URDF conditioning should handle unsupported robot identifiers."""
    explicit_policy = getattr(action_config, "urdf_unknown_robot_policy", "error")
    normalized_policy = str(explicit_policy).lower()
    valid_policies = {"error", "zero_tokens", "learnable_tokens"}
    if normalized_policy not in valid_policies:
        raise ValueError(
            "Unsupported `framework.action_model.urdf_unknown_robot_policy`: "
            f"{explicit_policy!r}. Expected one of {sorted(valid_policies)}."
        )
    return normalized_policy


def _normalize_learnable_urdf_robot_name(robot_name: str) -> str:
    """Normalize one configured fallback robot name for learnable URDF-token lookup."""
    normalized_robot_name = str(robot_name).strip().lower()
    if not normalized_robot_name:
        raise ValueError("`framework.action_model.urdf_learnable_robot_names` cannot contain empty names.")
    return normalized_robot_name


def _resolve_learnable_urdf_robot_names(action_config, *, global_config=None) -> tuple[str, ...]:
    """Resolve configured unsupported robot identifiers that should use learnable URDF embeddings."""
    configured_robot_names = getattr(action_config, "urdf_learnable_robot_names", ())
    if isinstance(configured_robot_names, str):
        raise TypeError(
            "`framework.action_model.urdf_learnable_robot_names` must be a sequence of robot names, not one string."
        )

    normalized_robot_names = tuple(
        _normalize_learnable_urdf_robot_name(robot_name) for robot_name in configured_robot_names
    )
    if len(set(normalized_robot_names)) != len(normalized_robot_names):
        raise ValueError(
            "`framework.action_model.urdf_learnable_robot_names` contains duplicate names after normalization."
        )

    return normalized_robot_names


def _resolve_learnable_supported_urdf_robot_names(action_config) -> tuple[str, ...]:
    """Resolve supported robot identifiers that should use learnable URDF embeddings."""
    configured_robot_names = getattr(action_config, "urdf_learnable_supported_robot_names", ())
    if isinstance(configured_robot_names, str):
        raise TypeError(
            "`framework.action_model.urdf_learnable_supported_robot_names` must be a sequence of robot names, "
            "not one string."
        )

    normalized_robot_names = tuple(
        _normalize_learnable_urdf_robot_name(resolve_robot_name(robot_name)) for robot_name in configured_robot_names
    )
    if len(set(normalized_robot_names)) != len(normalized_robot_names):
        raise ValueError(
            "`framework.action_model.urdf_learnable_supported_robot_names` contains duplicate names after "
            "canonicalization."
        )
    return normalized_robot_names


def _resolve_urdf_probe_robot_name(global_config, *, urdf_spec_overrides=None) -> str:
    """Select a graph-backed robot from the active data mix for encoder sizing.

    The probe only determines the graph feature width; it must not inspect an
    unrelated robot registered elsewhere in the process. This matters for
    single-embodiment runs whose prebuilt cache is mounted without every source
    URDF file.
    """
    vla_data_cfg = getattr(getattr(global_config, "datasets", None), "vla_data", None)
    if vla_data_cfg is not None:
        manifest_robot_type = vla_data_cfg.get("manifest_robot_type", None)
        if manifest_robot_type is not None and str(manifest_robot_type).strip():
            try:
                return resolve_robot_name(
                    str(manifest_robot_type),
                    urdf_spec_overrides=urdf_spec_overrides,
                )
            except KeyError:
                logger.warning(
                    "Cannot use unsupported manifest_robot_type `%s` as the URDF encoder probe.",
                    manifest_robot_type,
                )
        for key in ("robot_name", "robot_type"):
            robot_identifier = vla_data_cfg.get(key, None)
            if robot_identifier is None or not str(robot_identifier).strip():
                continue
            try:
                return resolve_robot_name(str(robot_identifier), urdf_spec_overrides=urdf_spec_overrides)
            except KeyError:
                logger.warning("Cannot use unsupported %s `%s` as the URDF encoder probe.", key, robot_identifier)

    if urdf_spec_overrides:
        return next(iter(urdf_spec_overrides.values())).robot_name

    supported_robot_names = list_supported_robot_names()
    if not supported_robot_names:
        raise RuntimeError("URDF conditioning is enabled, but no supported robot names are registered.")
    return supported_robot_names[0]


def _build_state_encoder(
    *,
    input_dim: int,
    output_dim: int,
    encoder_type: str,
    encoder_num_layers: int,
) -> nn.Module:
    """Build a lightweight encoder for state tokens or conditioning vectors."""
    normalized_encoder_type = str(encoder_type).lower()
    if normalized_encoder_type == "linear" or encoder_num_layers <= 1:
        return nn.Linear(input_dim, output_dim)

    if normalized_encoder_type != "mlp":
        raise ValueError(
            f"Unsupported `framework.action_model.state_encoder_type`: {encoder_type!r}. Expected 'linear' or 'mlp'."
        )

    hidden_dim = output_dim
    modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
    for _ in range(max(int(encoder_num_layers) - 2, 0)):
        modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


def _config_contains_key(config, key: str) -> bool:
    """Return whether one config object explicitly contains the requested key."""
    try:
        return key in config
    except TypeError:
        return hasattr(config, key)


def _should_use_legacy_token_state_encoder(action_config, *, state_injection_mode: str) -> bool:
    """Preserve the historical raw-state token MLP unless new encoder knobs are explicitly requested."""
    if state_injection_mode != "token":
        return False

    explicit_new_encoder_keys = ("state_encoder_type", "state_encoder_num_layers")
    return not any(_config_contains_key(action_config, key) for key in explicit_new_encoder_keys)


class StateTimestepConditioner(nn.Module):
    """Fuse one compact state vector with one timestep embedding for broadcast conditioning."""

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        state_encoder_type: str,
        state_encoder_num_layers: int,
        conditioning_mode: str,
    ) -> None:
        super().__init__()
        self.conditioning_mode = str(conditioning_mode).lower()
        self.state_encoder = _build_state_encoder(
            input_dim=int(state_dim),
            output_dim=int(hidden_dim),
            encoder_type=state_encoder_type,
            encoder_num_layers=state_encoder_num_layers,
        )

        if self.conditioning_mode == "concat":
            self.condition_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif self.conditioning_mode == "add":
            self.condition_proj = None
        else:
            raise ValueError(
                "Unsupported `framework.action_model.state_timestep_mode`: "
                f"{conditioning_mode!r}. Expected 'concat' or 'add'."
            )

    def _collapse_state_context(self, state: torch.Tensor) -> torch.Tensor:
        """Collapse optional state history into one per-sample vector for conditioning."""
        if state.ndim == 2:
            return state
        if state.ndim == 3:
            return state[:, -1, :]
        raise ValueError(f"Unsupported `state` shape for timestep conditioning: {tuple(state.shape)}.")

    def forward(self, *, timestep_embedding: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor | None:
        """Return one broadcastable conditioning vector with shape ``[B, hidden_dim]``."""
        if state is None:
            return None

        collapsed_state = self._collapse_state_context(state)
        state_hidden = self.state_encoder(collapsed_state)
        if self.conditioning_mode == "concat":
            return self.condition_proj(torch.cat([timestep_embedding, state_hidden], dim=-1))
        return timestep_embedding + state_hidden


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}  # fallback defaults


class FlowMatchingActionExpert(nn.Module):
    """Flow-matching action expert used by ACE-Ego-0."""

    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg
        vl_hidden_dim = int(global_config.framework.qwenvl.vl_hidden_dim)
        num_vl_layers = int(global_config.framework.qwenvl.num_vl_layers)

        action_hidden_dim = int(diffusion_model_cfg.get("input_embedding_dim", vl_hidden_dim))
        action_num_layers = int(diffusion_model_cfg.get("num_layers", num_vl_layers))
        attention_head_dim = int(diffusion_model_cfg.get("attention_head_dim", DiTConfig["attention_head_dim"]))
        configured_num_heads = diffusion_model_cfg.get("num_attention_heads")
        if configured_num_heads is None:
            action_num_heads = action_hidden_dim // attention_head_dim
        else:
            action_num_heads = int(configured_num_heads)

        if action_hidden_dim != action_num_heads * attention_head_dim:
            raise ValueError(
                "action_model.diffusion_model_cfg has inconsistent dimensions: "
                f"input_embedding_dim={action_hidden_dim}, "
                f"num_attention_heads={action_num_heads}, "
                f"attention_head_dim={attention_head_dim}. "
                "Expected input_embedding_dim == num_attention_heads * attention_head_dim."
            )
        if action_num_layers > num_vl_layers:
            raise ValueError(
                "action_model.diffusion_model_cfg.num_layers cannot exceed VLM layers: "
                f"num_layers={action_num_layers}, num_vl_layers={num_vl_layers}."
            )

        configured_cross_dim = diffusion_model_cfg.get("cross_attention_dim")
        if configured_cross_dim is not None and int(configured_cross_dim) != vl_hidden_dim:
            raise ValueError(
                "action_model.diffusion_model_cfg.cross_attention_dim must match VLM hidden size: "
                f"cross_attention_dim={configured_cross_dim}, vl_hidden_dim={vl_hidden_dim}."
            )

        diffusion_model_cfg.update(
            {
                "num_layers": action_num_layers,
                "input_embedding_dim": action_hidden_dim,
                "attention_head_dim": attention_head_dim,
                "num_attention_heads": action_num_heads,
                "cross_attention_dim": vl_hidden_dim,
            }
        )
        self.input_embedding_dim = action_hidden_dim
        self.model = DiT(**diffusion_model_cfg)  # TODO better way is copy LLM from VLM
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = _resolve_action_horizon(action_config)
        self.num_inference_timesteps = action_config.num_inference_timesteps
        self.state_injection_mode = _resolve_state_injection_mode(action_config)
        self.enable_state_input = self.state_injection_mode != "disabled"
        self.enable_state_timestep_conditioning = self.state_injection_mode == "timestep_conditioning"
        self.state_timestep_mode = str(getattr(action_config, "state_timestep_mode", "concat")).lower()
        self.use_legacy_token_state_encoder = _should_use_legacy_token_state_encoder(
            action_config,
            state_injection_mode=self.state_injection_mode,
        )
        self.enable_urdf_input = bool(getattr(action_config, "enable_urdf_input", False))
        self.urdf_use_body_token = bool(getattr(action_config, "urdf_use_body_token", True))
        self.urdf_use_chain_token = bool(getattr(action_config, "urdf_use_chain_token", True))
        self.urdf_unknown_robot_policy = _resolve_urdf_unknown_robot_policy(action_config)
        self.urdf_learnable_robot_names = _resolve_learnable_urdf_robot_names(
            action_config,
            global_config=global_config,
        )
        self.urdf_learnable_supported_robot_names = _resolve_learnable_supported_urdf_robot_names(action_config)
        self.urdf_spec_overrides = resolve_urdf_spec_overrides(getattr(action_config, "urdf_robot_specs", None))
        self.urdf_auto_build_cache = bool(getattr(action_config, "urdf_auto_build_cache", True))
        self.urdf_cache_dir = Path(str(getattr(action_config, "urdf_cache_dir", "urdf_cache"))).expanduser()
        self.urdf_encoder_hidden_dim = int(getattr(action_config, "urdf_encoder_hidden_dim", 256))
        self.urdf_encoder_layers = int(getattr(action_config, "urdf_encoder_layers", 2))
        configured_urdf_token_dim = int(getattr(action_config, "urdf_token_dim", self.input_embedding_dim))
        self.urdf_token_dim = configured_urdf_token_dim if configured_urdf_token_dim > 0 else self.input_embedding_dim
        self._urdf_graph_cache_by_robot: dict[str, dict[str, Any]] = {}
        self.last_urdf_diagnostics: dict[str, torch.Tensor] = {}

        if self.enable_state_input and not action_config.state_dim:
            raise ValueError("`framework.action_model.state_dim` must be set when `enable_state_input` is true.")

        state_encoder_type = str(getattr(action_config, "state_encoder_type", "mlp"))
        state_encoder_num_layers = int(getattr(action_config, "state_encoder_num_layers", 2))
        self.state_encoder = None
        self.state_timestep_conditioner = None
        if self.state_injection_mode == "token" and action_config.state_dim:
            if self.use_legacy_token_state_encoder:
                self.state_encoder = MLP(
                    input_dim=int(action_config.state_dim),
                    hidden_dim=1024,
                    output_dim=self.input_embedding_dim,
                )
            else:
                self.state_encoder = _build_state_encoder(
                    input_dim=int(action_config.state_dim),
                    output_dim=self.input_embedding_dim,
                    encoder_type=state_encoder_type,
                    encoder_num_layers=state_encoder_num_layers,
                )
        elif self.state_injection_mode == "timestep_conditioning" and action_config.state_dim:
            self.state_timestep_conditioner = StateTimestepConditioner(
                state_dim=int(action_config.state_dim),
                hidden_dim=self.input_embedding_dim,
                state_encoder_type=state_encoder_type,
                state_encoder_num_layers=state_encoder_num_layers,
                conditioning_mode=self.state_timestep_mode,
            )
        if self.enable_urdf_input and not (self.urdf_use_body_token or self.urdf_use_chain_token):
            raise ValueError("At least one of `urdf_use_body_token` or `urdf_use_chain_token` must be true.")
        if self.urdf_unknown_robot_policy == "learnable_tokens" and not self.urdf_learnable_robot_names:
            raise ValueError(
                "`framework.action_model.urdf_learnable_robot_names` must be configured when "
                "`framework.action_model.urdf_unknown_robot_policy=learnable_tokens`."
            )
        self.urdf_condition_token_count = int(self.urdf_use_body_token) + int(self.urdf_use_chain_token)

        self.urdf_encoder = None
        self.urdf_token_projection = None
        self.urdf_fusion = None
        learnable_urdf_embedding_names = tuple(
            dict.fromkeys((*self.urdf_learnable_robot_names, *self.urdf_learnable_supported_robot_names))
        )
        self.learnable_urdf_embeddings = nn.ParameterDict(
            {
                robot_name: nn.Parameter(torch.randn(self.input_embedding_dim) * 0.02)
                for robot_name in learnable_urdf_embedding_names
            }
        )
        if self.enable_urdf_input:
            probe_robot_name = _resolve_urdf_probe_robot_name(
                global_config,
                urdf_spec_overrides=self.urdf_spec_overrides,
            )
            probe_spec = get_robot_urdf_spec(
                probe_robot_name,
                require_urdf_file=False,
                urdf_spec_overrides=self.urdf_spec_overrides,
            )
            probe_payload = load_or_build_urdf_graph_cache(
                probe_spec,
                self.urdf_cache_dir,
                auto_build=self.urdf_auto_build_cache,
            )
            self._urdf_graph_cache_by_robot[probe_spec.robot_name] = probe_payload
            probe_feature_dim = int(probe_payload["feature_dim"])
            self.urdf_encoder = UrdfGraphEncoder(
                input_dim=probe_feature_dim,
                hidden_dim=self.urdf_encoder_hidden_dim,
                output_dim=self.urdf_token_dim,
                num_layers=self.urdf_encoder_layers,
            )
            self.urdf_token_projection = (
                nn.Identity()
                if self.urdf_token_dim == self.input_embedding_dim
                else nn.Linear(self.urdf_token_dim, self.input_embedding_dim)
            )
            self.urdf_fusion = GatedUrdfTokenFusion(token_dim=self.input_embedding_dim)

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config
        if _is_main_process():
            action_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            dit_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            non_dit_params = action_total_params - dit_params
            logger.info(
                "Action model parameters (trainable): total=%s, dit=%s, non_dit=%s | "
                "hidden=%d, layers=%d, heads=%d, head_dim=%d, cross_attention_dim=%d",
                f"{action_total_params:,}",
                f"{dit_params:,}",
                f"{non_dit_params:,}",
                action_hidden_dim,
                action_num_layers,
                action_num_heads,
                attention_head_dim,
                vl_hidden_dim,
            )
            logger.info(
                "URDF conditioning enabled: %s | cache_dir=%s | body_token=%s | chain_token=%s | "
                "unknown_robot_policy=%s | learnable_robot_names=%s | learnable_supported_robot_names=%s",
                self.enable_urdf_input,
                self.urdf_cache_dir,
                self.urdf_use_body_token,
                self.urdf_use_chain_token,
                self.urdf_unknown_robot_policy,
                list(self.urdf_learnable_robot_names),
                list(self.urdf_learnable_supported_robot_names),
            )
            logger.info(
                "Action model state route: mode=%s, legacy_token_encoder=%s",
                self.state_injection_mode,
                self.use_legacy_token_state_encoder,
            )

    def _encode_state_features(self, state: torch.Tensor | None) -> torch.Tensor | None:
        """Encode optional raw state inputs into one sequence token."""
        if self.state_injection_mode != "token" or self.state_encoder is None or state is None:
            return None

        if state.ndim == 2:
            state = state.unsqueeze(1)

        return self.state_encoder(state)

    def _apply_state_timestep_conditioning(
        self,
        action_side_embeddings: torch.Tensor,
        *,
        timestep_embedding: torch.Tensor,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        """Broadcast one fused state+timestep condition over the action-side sequence when configured."""
        if self.state_injection_mode != "timestep_conditioning" or self.state_timestep_conditioner is None:
            return action_side_embeddings

        conditioning = self.state_timestep_conditioner(
            timestep_embedding=timestep_embedding,
            state=state,
        )
        if conditioning is None:
            return action_side_embeddings

        conditioning_expanded = conditioning.unsqueeze(1).expand(-1, action_side_embeddings.shape[1], -1)
        return action_side_embeddings + conditioning_expanded

    def _load_urdf_graph_cache(self, robot_name: str) -> dict[str, Any]:
        """Load one cached URDF graph payload for a robot in the current batch."""
        canonical_robot_name = resolve_robot_name(
            robot_name,
            urdf_spec_overrides=self.urdf_spec_overrides,
        )
        cached_payload = self._urdf_graph_cache_by_robot.get(canonical_robot_name)
        if cached_payload is not None:
            return cached_payload

        robot_spec = get_robot_urdf_spec(
            canonical_robot_name,
            require_urdf_file=False,
            urdf_spec_overrides=self.urdf_spec_overrides,
        )
        cached_payload = load_or_build_urdf_graph_cache(
            robot_spec,
            self.urdf_cache_dir,
            auto_build=self.urdf_auto_build_cache,
        )
        self._urdf_graph_cache_by_robot[canonical_robot_name] = cached_payload
        return cached_payload

    @staticmethod
    def _move_urdf_graph_to_device(
        graph_payload: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Move the tensor-valued parts of one URDF graph payload to the current device."""
        return {
            "node_features": graph_payload["node_features"].to(device=device, dtype=dtype),
            "adjacency_norm": graph_payload["adjacency_norm"].to(device=device, dtype=dtype),
            "left_chain_mask": graph_payload["left_chain_mask"].to(device=device),
            "right_chain_mask": graph_payload["right_chain_mask"].to(device=device),
        }

    def _select_urdf_tokens(self, urdf_tokens: torch.Tensor) -> torch.Tensor:
        """Keep only the URDF tokens enabled by the current configuration."""
        selected_tokens: list[torch.Tensor] = []
        if self.urdf_use_body_token:
            selected_tokens.append(urdf_tokens[:, 0:1, :])
        if self.urdf_use_chain_token:
            selected_tokens.append(urdf_tokens[:, 1:2, :])
        return torch.cat(selected_tokens, dim=1)

    def _zero_urdf_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return zero URDF conditioning tokens for unsupported embodiments."""
        return torch.zeros(
            (batch_size, self.urdf_condition_token_count, self.input_embedding_dim),
            device=device,
            dtype=dtype,
        )

    def _log_unknown_urdf_robot_names(self, robot_names: list[str]) -> None:
        """Warn once per unknown robot identifier when the zero-token fallback is enabled."""
        if not robot_names or not _is_main_process():
            return

        for robot_name in robot_names:
            if robot_name in _LOGGED_UNKNOWN_URDF_ROBOTS:
                continue
            logger.warning(
                "Falling back to zero URDF tokens for unsupported robot identifier %s because "
                "`framework.action_model.urdf_unknown_robot_policy=zero_tokens`.",
                robot_name,
            )
            _LOGGED_UNKNOWN_URDF_ROBOTS.add(robot_name)

    def _log_learnable_urdf_robot_names(self, robot_names: list[str]) -> None:
        """Log once per robot identifier when learnable URDF conditioning is used."""
        if not robot_names or not _is_main_process():
            return

        for robot_name in robot_names:
            if robot_name in _LOGGED_LEARNABLE_URDF_ROBOTS:
                continue
            logger.info(
                "Using learnable URDF conditioning for robot identifier %s.",
                robot_name,
            )
            _LOGGED_LEARNABLE_URDF_ROBOTS.add(robot_name)

    def _resolve_learnable_urdf_embedding_name(self, robot_name: str) -> str:
        """Resolve one unsupported robot identifier into a configured learnable-URDF embedding key."""
        normalized_robot_name = _normalize_learnable_urdf_robot_name(robot_name)
        if normalized_robot_name not in self.learnable_urdf_embeddings:
            raise ValueError(
                "Unsupported robot identifier for learnable URDF fallback: "
                f"{robot_name!r}. Add it to `framework.action_model.urdf_learnable_robot_names` "
                "or switch `framework.action_model.urdf_unknown_robot_policy` to `error` or `zero_tokens`."
            )
        return normalized_robot_name

    def _cache_learnable_urdf_embedding(
        self,
        learnable_embeddings_by_robot: dict[str, torch.Tensor],
        robot_name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Cache one learnable URDF embedding tensor for the current batch."""
        if robot_name in learnable_embeddings_by_robot:
            return
        learnable_embeddings_by_robot[robot_name] = self.learnable_urdf_embeddings[robot_name].to(
            device=device,
            dtype=dtype,
        )

    def _fuse_projected_urdf_embeddings(
        self,
        *,
        body_batch: torch.Tensor,
        chain_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse projected URDF body and chain embeddings into the selected conditioning tokens."""
        if self.urdf_fusion is None:
            raise RuntimeError("URDF fusion was requested before initialization completed.")

        fused_tokens, diagnostics = self.urdf_fusion(body_batch, chain_batch)
        return self._select_urdf_tokens(fused_tokens), diagnostics

    def _encode_urdf_condition_tokens(
        self,
        robot_names: list[str] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Encode one batch of robot identities into action-expert conditioning tokens."""
        if not self.enable_urdf_input:
            self.last_urdf_diagnostics = {}
            return None

        if robot_names is None:
            raise ValueError("`robot_names` must be provided when `enable_urdf_input` is true.")
        if self.urdf_encoder is None or self.urdf_token_projection is None or self.urdf_fusion is None:
            raise RuntimeError("URDF modules were not initialized even though URDF input is enabled.")

        ordered_robot_sources: list[tuple[str, str]] = []
        body_embeddings_by_robot: dict[str, torch.Tensor] = {}
        chain_embeddings_by_robot: dict[str, torch.Tensor] = {}
        unknown_robot_names: list[str] = []
        learnable_robot_names: list[str] = []
        learnable_embeddings_by_robot: dict[str, torch.Tensor] = {}
        for robot_name in robot_names:
            try:
                canonical_robot_name = resolve_robot_name(
                    robot_name,
                    urdf_spec_overrides=self.urdf_spec_overrides,
                )
            except KeyError:
                if self.urdf_unknown_robot_policy == "zero_tokens":
                    ordered_robot_sources.append(("zero", str(robot_name)))
                    unknown_robot_names.append(str(robot_name))
                    continue
                if self.urdf_unknown_robot_policy == "learnable_tokens":
                    learnable_robot_name = self._resolve_learnable_urdf_embedding_name(str(robot_name))
                    ordered_robot_sources.append(("learnable", learnable_robot_name))
                    learnable_robot_names.append(str(robot_name))
                    self._cache_learnable_urdf_embedding(
                        learnable_embeddings_by_robot,
                        learnable_robot_name,
                        device=device,
                        dtype=dtype,
                    )
                    continue
                if self.urdf_unknown_robot_policy != "zero_tokens":
                    raise
            learnable_supported_robot_name = _normalize_learnable_urdf_robot_name(canonical_robot_name)
            if learnable_supported_robot_name in self.urdf_learnable_supported_robot_names:
                ordered_robot_sources.append(("learnable", learnable_supported_robot_name))
                learnable_robot_names.append(str(robot_name))
                self._cache_learnable_urdf_embedding(
                    learnable_embeddings_by_robot,
                    learnable_supported_robot_name,
                    device=device,
                    dtype=dtype,
                )
                continue

            ordered_robot_sources.append(("supported", canonical_robot_name))
            if canonical_robot_name in body_embeddings_by_robot:
                continue

            graph_payload = self._load_urdf_graph_cache(canonical_robot_name)
            graph_tensors = self._move_urdf_graph_to_device(graph_payload, device=device, dtype=dtype)
            body_embedding, chain_embedding = self.urdf_encoder(graph_tensors)
            body_embeddings_by_robot[canonical_robot_name] = self.urdf_token_projection(body_embedding)
            chain_embeddings_by_robot[canonical_robot_name] = self.urdf_token_projection(chain_embedding)

        self._log_unknown_urdf_robot_names(unknown_robot_names)
        self._log_learnable_urdf_robot_names(learnable_robot_names)

        supported_robot_names = [name for source_kind, name in ordered_robot_sources if source_kind == "supported"]
        diagnostics: dict[str, torch.Tensor] = {}
        supported_selected_tokens: torch.Tensor | None = None
        if supported_robot_names:
            body_batch = torch.stack([body_embeddings_by_robot[name] for name in supported_robot_names], dim=0)
            chain_batch = torch.stack([chain_embeddings_by_robot[name] for name in supported_robot_names], dim=0)
            supported_selected_tokens, diagnostics = self._fuse_projected_urdf_embeddings(
                body_batch=body_batch,
                chain_batch=chain_batch,
            )

        learnable_selected_tokens_by_robot: dict[str, torch.Tensor] = {}
        if learnable_embeddings_by_robot:
            ordered_learnable_robot_names = list(learnable_embeddings_by_robot.keys())
            learnable_batch = torch.stack(
                [learnable_embeddings_by_robot[name] for name in ordered_learnable_robot_names],
                dim=0,
            )
            learnable_selected_tokens, learnable_diagnostics = self._fuse_projected_urdf_embeddings(
                body_batch=learnable_batch,
                chain_batch=learnable_batch,
            )
            learnable_selected_tokens_by_robot = {
                name: learnable_selected_tokens[index : index + 1]
                for index, name in enumerate(ordered_learnable_robot_names)
            }
            diagnostics = dict(diagnostics)
            diagnostics.update(
                {f"learnable_{diagnostic_name}": value for diagnostic_name, value in learnable_diagnostics.items()}
            )
            diagnostics["learnable_robot_count"] = torch.tensor(
                len(learnable_robot_names),
                device=device,
                dtype=torch.int64,
            )

        if unknown_robot_names:
            diagnostics = dict(diagnostics)
            diagnostics["unknown_robot_count"] = torch.tensor(
                len(unknown_robot_names),
                device=device,
                dtype=torch.int64,
            )
        self.last_urdf_diagnostics = diagnostics

        zero_tokens = self._zero_urdf_tokens(batch_size=1, device=device, dtype=dtype)
        batch_tokens: list[torch.Tensor] = []
        supported_index = 0
        for source_kind, robot_name in ordered_robot_sources:
            if source_kind == "zero":
                batch_tokens.append(zero_tokens)
                continue
            if source_kind == "learnable":
                learnable_selected_tokens = learnable_selected_tokens_by_robot.get(robot_name)
                if learnable_selected_tokens is None:
                    raise RuntimeError(
                        "Learnable URDF robot names were present but no surrogate conditioning tokens were computed."
                    )
                batch_tokens.append(learnable_selected_tokens)
                continue
            if supported_selected_tokens is None:
                raise RuntimeError("Supported URDF robot names were present but no conditioning tokens were computed.")
            batch_tokens.append(supported_selected_tokens[supported_index : supported_index + 1])
            supported_index += 1
        return torch.cat(batch_tokens, dim=0)

    def _run_transformer_block(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one DiT block for inference."""
        if encoder_hidden_states is None:

            def _self_attention_layer_forward(
                current_hidden_states: torch.Tensor,
                current_temb: torch.Tensor,
            ) -> torch.Tensor:
                return layer(
                    hidden_states=current_hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=None,
                    temb=current_temb,
                )

            return _self_attention_layer_forward(hidden_states, temb)

        def _layer_forward(
            current_hidden_states: torch.Tensor,
            current_encoder_hidden_states: torch.Tensor,
            current_temb: torch.Tensor,
        ) -> torch.Tensor:
            return layer(
                hidden_states=current_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=current_encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                temb=current_temb,
            )

        return _layer_forward(hidden_states, encoder_hidden_states, temb)

    def _resolve_layer_encoder_hidden_states(
        self,
        *,
        layer_idx: int,
        vl_embs_list: list[torch.Tensor],
    ) -> torch.Tensor | None:
        """Return VLM context for cross-attention or none for an interleaved self-attention layer."""
        if bool(getattr(self.model.config, "interleave_self_attention", False)) and layer_idx % 2 == 1:
            return None
        return vl_embs_list[layer_idx]

    @staticmethod
    def _compute_inference_prediction(model_prediction: torch.Tensor) -> torch.Tensor:
        """Return the velocity prediction used by the released flow-matching checkpoints."""
        return model_prediction

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        robot_names: list[str] | None = None,
        action_horizon: int | None = None,
    ) -> torch.Tensor:
        """Predict one normalized action trajectory with flow matching."""
        resolved_action_horizon = self.action_horizon if action_horizon is None else int(action_horizon)
        if resolved_action_horizon <= 0:
            raise ValueError(f"Inference action horizon must be positive, got {resolved_action_horizon}.")
        if self.config.add_pos_embed and resolved_action_horizon > int(self.config.max_seq_len):
            raise ValueError(
                "Inference action horizon exceeds action-model positional embedding capacity: "
                f"action_horizon={resolved_action_horizon}, max_seq_len={self.config.max_seq_len}."
            )

        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, resolved_action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self._encode_state_features(state)
        urdf_tokens = self._encode_urdf_condition_tokens(robot_names, device=device, dtype=actions.dtype)

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)

            # Encode timestep
            temb = self.model.timestep_encoder(timesteps_tensor)
            action_side_embeddings = torch.cat((future_tokens, action_features), dim=1)
            action_side_embeddings = self._apply_state_timestep_conditioning(
                action_side_embeddings,
                timestep_embedding=temb,
                state=state,
            )
            sequence_segments: list[torch.Tensor] = []
            if state_features is not None:
                sequence_segments.append(state_features)
            if urdf_tokens is not None:
                sequence_segments.append(urdf_tokens)
            sequence_segments.append(action_side_embeddings)
            sa_embs = torch.cat(sequence_segments, dim=1)

            # Run the same layerwise cross/self-attention schedule used during training.
            model_output = sa_embs
            for layer_idx, layer in enumerate(self.model.transformer_blocks):
                encoder_hidden_states = self._resolve_layer_encoder_hidden_states(
                    layer_idx=layer_idx,
                    vl_embs_list=vl_embs_list,
                )
                model_output = self._run_transformer_block(layer, model_output, encoder_hidden_states, temb)
            # TODO miss self att and _process_output
            model_output = self.model.norm_out(model_output)
            pred = self.action_decoder(model_output)
            model_prediction = pred[:, -resolved_action_horizon:]
            pred_velocity = self._compute_inference_prediction(model_prediction)
            # Euler integration
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Build the ACE-Ego-0 flow-matching action expert.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        FlowMatchingActionExpert: Initialized action expert.
    """
    return FlowMatchingActionExpert(global_config=config)
