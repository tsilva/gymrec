"""Gymrec-owned adapters for the two supported native Gymnasium runtimes."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np

CONTRACT_VERSION = 1
STABLE_RETRO_PROVIDER_ID = "stable-retro-turbo"
MARIO_TURBO_PROVIDER_ID = "supermariobrosnes-turbo"
BUILTIN_ACTION_SETS = {
    "simple": (
        (),
        ("RIGHT",),
        ("RIGHT", "B"),
        ("RIGHT", "A"),
        ("RIGHT", "A", "B"),
        ("A",),
        ("LEFT",),
    ),
    "right": (
        ("RIGHT",),
        ("RIGHT", "B"),
        ("RIGHT", "A"),
        ("RIGHT", "A", "B"),
    ),
}
_MANAGED_CONFIG_KEYS = frozenset(
    {"game", "num_envs", "num_threads", "render_mode", "autoreset_mode"}
)


def _json_value(value):
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value.name)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_json_value(item) for item in value]
    return value


def _lane_info(infos):
    if not isinstance(infos, Mapping):
        return {}
    result = {}
    for key, value in infos.items():
        if str(key).startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask).reshape(-1)[0]):
            continue
        if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
            value = value[0]
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, memoryview))
            and len(value) == 1
        ):
            value = value[0]
        result[str(key)] = _json_value(value)
    return result


class SingleLaneEnv(gym.Env):
    """Expose one lane of a native Gymnasium VectorEnv as a scalar environment."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vector_env):
        if int(vector_env.num_envs) != 1:
            raise ValueError("Gymrec providers require exactly one environment lane")
        self.vector_env = vector_env
        self.action_space = vector_env.single_action_space
        self.observation_space = vector_env.single_observation_space
        self.render_mode = "rgb_array"
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, infos = self.vector_env.reset(seed=seed, options=options)
        self._needs_reset = False
        return observations[0], _lane_info(infos)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError(
                "reset() must be called before step() or after a terminal step"
            )
        if isinstance(self.action_space, gym.spaces.Discrete):
            scalar_action = int(np.asarray(action).reshape(-1)[0])
            if not self.action_space.contains(scalar_action):
                raise ValueError(
                    f"Action {scalar_action!r} is not in {self.action_space}"
                )
            batched_action = np.asarray(
                [scalar_action], dtype=self.action_space.dtype
            )
        else:
            scalar_action = np.asarray(action, dtype=self.action_space.dtype)
            if not self.action_space.contains(scalar_action):
                raise ValueError(f"Action {action!r} is not in {self.action_space}")
            batched_action = scalar_action[np.newaxis, ...]
        observations, rewards, terminated, truncated, infos = self.vector_env.step(
            batched_action
        )
        is_terminated = bool(terminated[0])
        is_truncated = bool(truncated[0])
        self._needs_reset = is_terminated or is_truncated
        return (
            observations[0],
            float(rewards[0]),
            is_terminated,
            is_truncated,
            _lane_info(infos),
        )

    def render(self):
        return self.vector_env.render()

    def close(self):
        self.vector_env.close()


def _file_sha256(path):
    if not path:
        return None
    try:
        with open(path, "rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except (FileNotFoundError, OSError, TypeError):
        return None


def _normalize_policy_action(task):
    if task is None:
        return None, None
    if not isinstance(task, Mapping):
        raise ValueError("config.task must be an object")
    ignored = sorted(str(key) for key in task if key != "action")
    if ignored:
        warnings.warn(
            "Gymrec uses only config.task.action for mechanical policy-action "
            f"conversion and ignores native-task sections: {', '.join(ignored)}",
            UserWarning,
            stacklevel=4,
        )
    action = task.get("action")
    if action is None:
        return None, None
    if not isinstance(action, Mapping):
        raise ValueError("config.task.action must be an object")
    extra = sorted(set(action) - {"set", "actions"})
    if extra:
        raise ValueError(
            "config.task.action has unsupported keys: " + ", ".join(extra)
        )
    explicit = action.get("actions")
    name = action.get("set")
    if explicit is not None and name is not None:
        raise ValueError("config.task.action cannot define both set and actions")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(
            explicit, (str, bytes, bytearray, memoryview)
        ):
            raise ValueError("config.task.action.actions must be an array")
        normalized = []
        for labels in explicit:
            if not isinstance(labels, Sequence) or isinstance(
                labels, (str, bytes, bytearray, memoryview)
            ):
                raise ValueError(
                    "Each config.task.action.actions entry must be an array of controls"
                )
            normalized.append(tuple(str(label).upper() for label in labels))
        if not normalized:
            raise ValueError("config.task.action.actions must not be empty")
        return tuple(normalized), {
            "actions": [list(labels) for labels in normalized]
        }
    if name in (None, "native"):
        return None, None
    normalized_name = str(name).lower()
    try:
        policy_actions = BUILTIN_ACTION_SETS[normalized_name]
    except KeyError as exc:
        names = ", ".join(sorted(BUILTIN_ACTION_SETS))
        raise ValueError(
            f"Unknown Gymrec action set {name!r}; expected native, {names}, "
            "or explicit actions"
        ) from exc
    return policy_actions, {"set": normalized_name}


def _prepare_config(config):
    kwargs = copy.deepcopy(dict(config))
    task = kwargs.pop("task", None)
    policy_actions, effective_action = _normalize_policy_action(task)
    for key in sorted(_MANAGED_CONFIG_KEYS.intersection(kwargs)):
        raise ValueError(f"config.{key} is managed by Gymrec")
    effective = copy.deepcopy(kwargs)
    if effective_action is not None:
        effective["task"] = {"action": effective_action}
    return kwargs, policy_actions, effective


class ProviderSession:
    """Shared Gymrec session behavior over one native runtime lane."""

    def __init__(
        self,
        *,
        provider_id,
        environment_id,
        effective_config,
        vector_env,
        system,
        buttons,
        policy_actions,
        fps,
        assets,
    ):
        self.provider_id = provider_id
        self.environment_id = environment_id
        self.effective_config = effective_config
        self.env = SingleLaneEnv(vector_env)
        self.control_profile = f"stable_retro.{system}"
        self.fps = max(float(fps), 1.0)
        self._buttons = tuple(
            str(button).upper() if button is not None else "" for button in buttons
        )
        self._policy_actions = policy_actions
        self.provenance = {
            "distribution": provider_id,
            "version": importlib.metadata.version(provider_id),
            "assets": _json_value(assets),
        }

    def policy_observation(self, observation):
        return observation

    def recording_observation(self, observation):
        frame = self.env.render()
        return observation if frame is None else frame

    def adapt_policy_action(self, action):
        if self._policy_actions is None:
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                return int(np.asarray(action).reshape(-1)[0])
            return action
        index = int(np.asarray(action).reshape(-1)[0])
        if index < 0 or index >= len(self._policy_actions):
            raise ValueError(
                f"Policy action {index} is outside the configured action set"
            )
        return self.action_from_labels(self._policy_actions[index])

    def validate_policy(self, policy):
        policy_action = getattr(policy, "action_space", None)
        if self._policy_actions is not None:
            if getattr(policy_action, "n", None) != len(self._policy_actions):
                raise ValueError(
                    "Policy action space does not match the Gymrec action set"
                )
        elif policy_action is not None:
            expected_n = getattr(self.env.action_space, "n", None)
            actual_n = getattr(policy_action, "n", None)
            if expected_n is not None and actual_n != expected_n:
                raise ValueError(
                    "Policy action space does not match the native environment"
                )
            expected_shape = getattr(self.env.action_space, "shape", None)
            actual_shape = getattr(policy_action, "shape", None)
            if expected_n is None and actual_shape != expected_shape:
                raise ValueError(
                    "Policy action space does not match the native environment"
                )
        policy_observation = getattr(policy, "observation_space", None)
        if policy_observation is not None and getattr(
            policy_observation, "shape", None
        ) != getattr(self.env.observation_space, "shape", None):
            raise ValueError("Policy observation space does not match the provider")

    def action_from_labels(self, labels):
        requested = {str(label).upper() for label in labels}
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            meanings = getattr(self.env.vector_env, "action_meanings", ())
            action_buttons = getattr(self.env.vector_env, "ACTION_BUTTONS", None)
            if action_buttons is None:
                try:
                    from supermariobrosnes_turbo import ACTION_BUTTONS

                    action_buttons = ACTION_BUTTONS
                except ImportError:
                    action_buttons = {}
            for index, meaning in enumerate(meanings):
                if {
                    str(label).upper()
                    for label in action_buttons.get(str(meaning), ())
                } == requested:
                    return index
            raise ValueError(
                f"No configured action matches controls {sorted(requested)!r}"
            )
        if not isinstance(self.env.action_space, gym.spaces.MultiBinary):
            raise ValueError(
                "Named controls require a MultiBinary or named Discrete action space"
            )
        action = np.zeros(
            self.env.action_space.n, dtype=self.env.action_space.dtype
        )
        for label in requested:
            try:
                action[self._buttons.index(label)] = 1
            except ValueError as exc:
                raise ValueError(f"Control label {label!r} is unavailable") from exc
        return action


def _resolve_stable_enum(value, enum_type):
    if not isinstance(value, str):
        return value
    name = value.split(".")[-1].upper()
    try:
        return enum_type[name]
    except KeyError as exc:
        raise ValueError(f"Unknown {enum_type.__name__} value {value!r}") from exc


def _stable_path(data, environment_id, value, default_name, inttype):
    if value is None:
        value = default_name
    raw = str(value)
    if raw.endswith(".json") or Path(raw).is_absolute():
        return raw
    filename = raw if raw.endswith(".json") else f"{raw}.json"
    return data.get_file_path(environment_id, filename, inttype)


def _stable_default_state_name(data, environment_id, inttype):
    metadata_path = data.get_file_path(environment_id, "metadata.json", inttype)
    try:
        with open(metadata_path, encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, TypeError, json.JSONDecodeError):
        return None
    players = metadata.get("default_player_state")
    if isinstance(players, Sequence) and not isinstance(players, str) and players:
        return players[0]
    return metadata.get("default_state")


def _stable_state_hashes(data, stable_retro, environment_id, state, inttype):
    if state == stable_retro.State.NONE:
        return {}
    if state == stable_retro.State.DEFAULT:
        state = _stable_default_state_name(data, environment_id, inttype)
    if state is None:
        return {}
    values = tuple(state.keys()) if isinstance(state, Mapping) else state
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        values = (values,)
    hashes = {}
    for value in values:
        if isinstance(value, (bytes, bytearray, memoryview)):
            digest = hashlib.sha256(bytes(value)).hexdigest()
            label = f"inline-{digest[:12]}"
        else:
            raw = str(value)
            path = Path(raw).expanduser()
            if not path.exists():
                filename = raw if raw.endswith(".state") else f"{raw}.state"
                resolved = data.get_file_path(environment_id, filename, inttype)
                path = Path(resolved) if resolved else path
            digest = _file_sha256(path)
            label = raw
        hashes[label] = digest
    return hashes


class StableRetroProvider:
    provider_id = STABLE_RETRO_PROVIDER_ID
    contract_version = CONTRACT_VERSION

    def create(self, *, environment_id, config, render_mode):
        del render_mode
        import stable_retro
        from stable_retro import data

        kwargs, policy_actions, effective = _prepare_config(config)
        state = kwargs.pop("state", stable_retro.State.DEFAULT)
        if "use_restricted_actions" in kwargs:
            kwargs["use_restricted_actions"] = _resolve_stable_enum(
                kwargs["use_restricted_actions"], stable_retro.Actions
            )
        if "obs_type" in kwargs:
            kwargs["obs_type"] = _resolve_stable_enum(
                kwargs["obs_type"], stable_retro.Observations
            )
        inttype = _resolve_stable_enum(
            kwargs.get("inttype", data.Integrations.STABLE), data.Integrations
        )
        if "inttype" in kwargs:
            kwargs["inttype"] = inttype
        constructor = stable_retro.RetroVecEnv
        if "autoreset_mode" in inspect.signature(constructor).parameters:
            kwargs["autoreset_mode"] = "Disabled"
        vector_env = constructor(
            environment_id,
            state=state,
            render_mode="rgb_array",
            num_envs=1,
            num_threads=1,
            **kwargs,
        )
        rom_path = getattr(vector_env, "rom_path", None) or kwargs.get("rom_path")
        if not rom_path:
            rom_path = data.get_original_romfile_path(environment_id, inttype)
        system = getattr(vector_env, "system", None) or stable_retro.get_romfile_system(
            rom_path
        )
        buttons = getattr(vector_env, "buttons", None) or data.EMU_INFO[system][
            "buttons"
        ]
        info_path = getattr(vector_env, "info_path", None) or _stable_path(
            data, environment_id, kwargs.get("info"), "data", inttype
        )
        scenario_path = getattr(vector_env, "scenario_path", None) or _stable_path(
            data, environment_id, kwargs.get("scenario"), "scenario", inttype
        )
        initial_states = getattr(vector_env, "initial_state_assets", None)
        assets = {
            "rom_sha256": _file_sha256(rom_path),
            "info_sha256": _file_sha256(info_path),
            "scenario_sha256": _file_sha256(scenario_path),
            "state_sha256": (
                list(initial_states)
                if initial_states is not None
                else _stable_state_hashes(
                    data, stable_retro, environment_id, state, inttype
                )
            ),
        }
        frame_skip = max(int(kwargs.get("frame_skip", 1)), 1)
        return ProviderSession(
            provider_id=self.provider_id,
            environment_id=environment_id,
            effective_config=effective,
            vector_env=vector_env,
            system=system,
            buttons=buttons,
            policy_actions=policy_actions,
            fps=60.0 / frame_skip,
            assets=assets,
        )

    def catalog(self):
        from stable_retro import data

        return tuple(sorted(data.list_games(data.Integrations.STABLE)))


def _mario_state_hashes(state, state_dir):
    if state is None:
        return {}
    from supermariobrosnes_turbo.env import _resolve_state_path

    values = tuple(state.keys()) if isinstance(state, Mapping) else state
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        values = (values,)
    hashes = {}
    for value in values:
        if isinstance(value, (bytes, bytearray, memoryview)):
            digest = hashlib.sha256(bytes(value)).hexdigest()
            label = f"inline-{digest[:12]}"
        else:
            digest = _file_sha256(_resolve_state_path(value, state_dir))
            label = str(value)
        hashes[label] = digest
    return hashes


def _resolve_mario_state(state, state_dir):
    if state is None or isinstance(state, (bytes, bytearray, memoryview)):
        return state
    from supermariobrosnes_turbo.env import _resolve_state_path

    if isinstance(state, Mapping):
        return {
            _resolve_state_path(value, state_dir): weight
            for value, weight in state.items()
        }
    if isinstance(state, Sequence) and not isinstance(state, str):
        return [_resolve_state_path(value, state_dir) for value in state]
    return _resolve_state_path(state, state_dir)


class MarioTurboProvider:
    provider_id = MARIO_TURBO_PROVIDER_ID
    contract_version = CONTRACT_VERSION

    def create(self, *, environment_id, config, render_mode):
        del render_mode
        from supermariobrosnes_turbo import (
            NES_BUTTONS,
            SuperMarioBrosNesTurboVecEnv,
            resolve_required_rom_path,
        )

        kwargs, policy_actions, effective = _prepare_config(config)
        state = kwargs.pop("state", "Level1-1")
        state_dir = kwargs.pop("state_dir", None)
        rom_path = resolve_required_rom_path(kwargs.get("rom_path"), environment_id)
        vector_env = SuperMarioBrosNesTurboVecEnv(
            environment_id,
            state=_resolve_mario_state(state, state_dir),
            render_mode="rgb_array",
            num_envs=1,
            num_threads=1,
            **kwargs,
        )
        frame_skip = max(int(kwargs.get("frame_skip", 1)), 1)
        return ProviderSession(
            provider_id=self.provider_id,
            environment_id=environment_id,
            effective_config=effective,
            vector_env=vector_env,
            system="Nes",
            buttons=NES_BUTTONS,
            policy_actions=policy_actions,
            fps=60.0 / frame_skip,
            assets={
                "rom_sha256": _file_sha256(rom_path),
                "state_sha256": _mario_state_hashes(state, state_dir),
            },
        )

    def catalog(self):
        return ("SuperMarioBros-Nes-v0",)


PROVIDERS = {
    STABLE_RETRO_PROVIDER_ID: StableRetroProvider(),
    MARIO_TURBO_PROVIDER_ID: MarioTurboProvider(),
}
SUPPORTED_PROVIDER_IDS = frozenset(PROVIDERS)
