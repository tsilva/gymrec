import asyncio
import json
import sys
import types

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode

import main


@pytest.fixture
def temp_storage_dir(tmp_path):
    main._lazy_init()
    previous_local_dir = main.CONFIG["storage"]["local_dir"]
    main.CONFIG["storage"]["local_dir"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        main.CONFIG["storage"]["local_dir"] = previous_local_dir


class FakeTurboVectorEnv:
    metadata = {"render_modes": ["rgb_array"], "autoreset_mode": AutoresetMode.DISABLED}
    num_envs = 1
    single_action_space = gym.spaces.MultiBinary(9)
    single_observation_space = gym.spaces.Box(0, 255, shape=(2, 3, 3), dtype=np.uint8)
    render_mode = "rgb_array"
    autoreset_mode = AutoresetMode.DISABLED

    def __init__(self, terminal=True):
        self.terminal = terminal
        self.actions = []
        self.reset_calls = []
        self.closed = False
        self.rendered = np.full((2, 3, 3), 9, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append((seed, options))
        observations = np.full((1, 2, 3, 3), 2, dtype=np.uint8)
        infos = {
            "x_pos": np.array([10]),
            "_x_pos": np.array([True]),
            "hidden": np.array([99]),
            "_hidden": np.array([False]),
        }
        return observations, infos

    def step(self, actions):
        self.actions.append(np.array(actions, copy=True))
        observations = np.full((1, 2, 3, 3), 3, dtype=np.uint8)
        infos = {
            "x_pos": np.array([11]),
            "_x_pos": np.array([True]),
            "label": np.array(["lane-zero"], dtype=object),
            "_label": np.array([True]),
        }
        return (
            observations,
            np.array([1.5], dtype=np.float32),
            np.array([self.terminal]),
            np.array([False]),
            infos,
        )

    def render(self):
        return self.rendered

    def close(self):
        self.closed = True


class FakeStableRetroEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.action_space = gym.spaces.MultiBinary(9)
        self.observation_space = gym.spaces.Box(0, 255, shape=(2, 3, 3), dtype=np.uint8)
        self.actions = []
        self._stable_retro = True
        self._gymrec_backend = "stable-retro"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((2, 3, 3), dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        return np.zeros((2, 3, 3), dtype=np.uint8), 0.0, False, False, {}


class FakeMarioRecipeEnv(gym.Env):
    metadata = {}
    action_space = gym.spaces.MultiBinary(9)
    observation_space = gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8)

    def __init__(self):
        self._step = 0
        self._gymrec_policy_observations = True

    @staticmethod
    def _info(x, score=0, lives=3, level=(0, 0)):
        return {
            "xscrollHi": x // 256,
            "xscrollLo": x % 256,
            "score": score,
            "lives": lives,
            "levelHi": level[0],
            "levelLo": level[1],
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), self._info(0)

    def step(self, action):
        del action
        self._step += 1
        if self._step == 1:
            info = self._info(10, score=100)
        else:
            info = self._info(0, score=100, level=(0, 1))
        return np.zeros((4, 84, 84), dtype=np.uint8), 0.0, False, False, info

    def render(self):
        return np.full((224, 240, 3), 7, dtype=np.uint8)


def test_runtime_backend_selection_is_explicit_and_backward_compatible(monkeypatch):
    monkeypatch.setattr(main, "_get_stableretro_envs", lambda: [main.SUPERMARIOBROS_NES_ENV_ID])

    assert (
        main.resolve_runtime_backend(
            main.SUPERMARIOBROS_NES_ENV_ID,
            requested="supermariobrosnes-turbo",
        )
        == "supermariobrosnes-turbo"
    )
    assert main.resolve_runtime_backend(main.SUPERMARIOBROS_NES_ENV_ID) == "stable-retro"
    assert (
        main.resolve_runtime_backend(
            main.SUPERMARIOBROS_NES_ENV_ID,
            metadata={"capture_backend": "supermariobrosnes-turbo"},
        )
        == "supermariobrosnes-turbo"
    )


def test_runtime_backend_rejects_turbo_for_unsupported_environment(monkeypatch):
    monkeypatch.setattr(main, "_get_stableretro_envs", lambda: [])

    with pytest.raises(ValueError, match="only supports SuperMarioBros-Nes-v0"):
        main.resolve_runtime_backend(
            "SonicTheHedgehog-Genesis-v0",
            requested="supermariobrosnes-turbo",
        )


def test_explicit_playback_backend_overrides_capture_backend(monkeypatch):
    monkeypatch.setattr(main, "_get_stableretro_envs", lambda: [main.SUPERMARIOBROS_NES_ENV_ID])

    assert (
        main.resolve_runtime_backend(
            main.SUPERMARIOBROS_NES_ENV_ID,
            requested="stable-retro",
            recorded_backend="supermariobrosnes-turbo",
        )
        == "stable-retro"
    )


def test_older_metadata_without_capture_backend_uses_stable_retro(monkeypatch):
    monkeypatch.setattr(main, "_get_stableretro_envs", lambda: [main.SUPERMARIOBROS_NES_ENV_ID])

    assert (
        main.resolve_runtime_backend(
            main.SUPERMARIOBROS_NES_ENV_ID,
            metadata={"stable_retro_state": "Level1-1"},
        )
        == "stable-retro"
    )


def test_turbo_adapter_batches_actions_and_extracts_lane_zero():
    vector_env = FakeTurboVectorEnv()
    env = main.SuperMarioBrosNesTurboEnvAdapter(vector_env)
    action = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)

    observation, reset_info = env.reset(seed=123, options={"example": True})
    next_observation, reward, terminated, truncated, info = env.step(action)

    assert vector_env.reset_calls == [(123, {"example": True})]
    assert observation.shape == (2, 3, 3)
    assert reset_info == {"x_pos": 10}
    assert vector_env.actions[0].shape == (1, 9)
    np.testing.assert_array_equal(vector_env.actions[0][0], action)
    assert next_observation.shape == (2, 3, 3)
    assert reward == pytest.approx(1.5)
    assert terminated is True
    assert truncated is False
    assert info == {"x_pos": 11, "label": "lane-zero"}


def test_turbo_adapter_preserves_disabled_autoreset_render_and_close():
    vector_env = FakeTurboVectorEnv()
    env = main.SuperMarioBrosNesTurboEnvAdapter(vector_env)
    action = np.zeros(9, dtype=np.int8)

    assert env.autoreset_mode is AutoresetMode.DISABLED
    env.reset()
    env.step(action)
    with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
        env.step(action)
    assert env.render() is vector_env.rendered
    env.close()
    assert vector_env.closed is True


def test_recipe_policy_observation_uses_raw_provider_render_for_recording():
    env = FakeMarioRecipeEnv()
    observation, _ = env.reset()

    frame = main._recording_observation(env, observation)

    assert frame.shape == (224, 240, 3)
    assert np.all(frame == 7)


def test_mario_recipe_task_applies_score_reward_and_level_success():
    task = {
        "id": "mario",
        "action": {"set": "simple"},
        "signals": {
            "x": ["xscrollHi", "xscrollLo"],
            "score": "score",
            "lives": "lives",
            "level": ["levelHi", "levelLo"],
        },
        "events": {
            "life_loss": {"signal": "lives", "operation": "decrease"},
            "level_change": {"signal": "level", "operation": "change"},
        },
        "termination": {
            "failure": ["life_loss"],
            "success": ["level_change"],
            "max_episode_steps": 4500,
        },
        "reward": {
            "reward_mode": "score",
            "progress_reward_scale": 1.0,
            "score_progress_clipped": False,
            "completion_reward": 0.0,
            "death_penalty": 25.0,
            "time_penalty": 0.0,
            "use_native_reward": False,
            "clip_rewards": False,
        },
    }
    env = main.MarioRecipeTaskEnv(FakeMarioRecipeEnv(), task)
    env.reset(seed=10000)

    _, reward, terminated, truncated, info = env.step(np.zeros(9, dtype=np.int8))
    assert reward == pytest.approx(11.0)
    assert terminated is False
    assert truncated is False
    assert info["progress_delta"] == 10
    assert info["score_delta"] == 100

    _, reward, terminated, truncated, info = env.step(np.zeros(9, dtype=np.int8))
    assert reward == pytest.approx(0.0)
    assert terminated is True
    assert truncated is False
    assert info["task_outcome"] == "success"
    assert info["task_events"] == ["level_change"]


def test_turbo_backend_delegates_rom_discovery_to_package(monkeypatch):
    constructor_kwargs = {}

    def fake_constructor(env_id, **kwargs):
        constructor_kwargs.update(kwargs)
        assert env_id == main.SUPERMARIOBROS_NES_ENV_ID
        return FakeTurboVectorEnv()

    fake_module = types.SimpleNamespace(
        Actions=types.SimpleNamespace(ALL="all"),
        SuperMarioBrosNesTurboVecEnv=fake_constructor,
    )
    monkeypatch.setitem(sys.modules, "supermariobrosnes_turbo", fake_module)

    env = main._create_env__supermariobrosnes_turbo(main.SUPERMARIOBROS_NES_ENV_ID)

    assert "rom_path" not in constructor_kwargs
    assert constructor_kwargs["state"] == "Level1-1"
    env.close()


def test_capture_backend_metadata_round_trip(temp_storage_dir):
    main._lazy_init()
    episode_id = bytes.fromhex("00" * 16)
    session_id = bytes.fromhex("01" * 16)
    dataset = main.Dataset.from_dict(
        {
            "episode_id": [episode_id],
            "seed": [123],
            "actions": [[]],
            "rewards": [None],
            "terminations": [None],
            "truncations": [None],
            "infos": [None],
            "session_id": [session_id],
            "collector": ["human"],
            "gymrec_version": ["test"],
            "storage_format": [main.STORAGE_FORMAT_IMAGES],
            "logical_env_id": [main.SUPERMARIOBROS_NES_ENV_ID],
            "capture_backend": ["supermariobrosnes-turbo"],
            "state_name": ["Level1-1"],
            "rom_sha256": [main.SUPERMARIOBROS_NES_ROM_SHA256],
            "nes_button_order": [json.dumps(main.NES_BUTTON_ORDER)],
            "action_encoding": [main.NES_ACTION_ENCODING],
            "frame_skip": [1],
            "sticky_action_prob": [0.0],
            "collector_contract_id": [None],
            "policy_mode": [None],
            "policy_seed": [None],
            "observations": [np.zeros((1, 1, 3), dtype=np.uint8)],
        }
    )
    dataset = dataset.cast_column("episode_id", main.Value("binary"))
    dataset = dataset.cast_column("session_id", main.Value("binary"))
    metadata = {
        "env_id": main.SUPERMARIOBROS_NES_ENV_ID,
        "capture_backend": "supermariobrosnes-turbo",
        "state_name": "Level1-1",
        "rom_sha256": main.SUPERMARIOBROS_NES_ROM_SHA256,
        "nes_button_order": list(main.NES_BUTTON_ORDER),
        "action_encoding": main.NES_ACTION_ENCODING,
        "frameskip": 1,
        "sticky_actions": 0.0,
    }

    main.save_dataset_locally(dataset, main.SUPERMARIOBROS_NES_ENV_ID, metadata=metadata)
    loaded = main.load_local_metadata(main.SUPERMARIOBROS_NES_ENV_ID)

    assert loaded["capture_backend"] == "supermariobrosnes-turbo"
    assert loaded["state_name"] == "Level1-1"
    assert loaded["rom_sha256"] == main.SUPERMARIOBROS_NES_ROM_SHA256
    assert loaded["nes_button_order"] == list(main.NES_BUTTON_ORDER)
    assert loaded["recordings"][0]["capture_backend"] == "supermariobrosnes-turbo"


def test_dataset_capture_backend_is_optional_for_older_datasets():
    main._lazy_init()
    old_dataset = main.Dataset.from_dict({"actions": [[0], []]})
    new_dataset = main.Dataset.from_dict(
        {"actions": [[0], []], "capture_backend": ["supermariobrosnes-turbo"] * 2}
    )

    assert main._capture_backend_from_dataset(old_dataset) is None
    assert main._capture_backend_from_dataset(new_dataset) == "supermariobrosnes-turbo"


def test_legacy_hub_schema_is_rejected_without_alignment():
    main._lazy_init()
    with pytest.raises(ValueError, match="canonical gymrec schema"):
        main._validate_remote_parquet_schema(
            "owner/legacy", ["actions"], main.STORAGE_FORMAT_IMAGES
        )


def test_environment_metadata_can_be_recovered_from_dataset_rows():
    main._lazy_init()
    dataset = main.Dataset.from_dict(
        {
            "logical_env_id": [main.SUPERMARIOBROS_NES_ENV_ID],
            "capture_backend": ["supermariobrosnes-turbo"],
            "state_name": ["Level1-1"],
            "rom_sha256": [main.SUPERMARIOBROS_NES_ROM_SHA256],
            "nes_button_order": [json.dumps(main.NES_BUTTON_ORDER)],
            "action_encoding": [main.NES_ACTION_ENCODING],
            "frame_skip": [1],
            "sticky_action_prob": [0.0],
        }
    )

    metadata = main._environment_metadata_from_dataset(dataset)

    assert metadata["capture_backend"] == "supermariobrosnes-turbo"
    assert metadata["state_name"] == "Level1-1"
    assert metadata["env_make_kwargs"] == {
        "frame_skip": 1,
        "sticky_action_prob": 0.0,
    }


def test_same_stored_action_vectors_reach_stable_retro_and_turbo():
    stored_actions = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
    ]

    async def replay_into(env):
        recorder = main.DatasetRecorderWrapper(
            env, headless=True, storage_format=main.STORAGE_FORMAT_IMAGES
        )
        recorder._ensure_screen = lambda _frame: None
        recorder._render_frame = lambda _frame: None
        recorder._print_keymappings = lambda: None
        recorder._input_loop = lambda: True
        await recorder.replay(
            fps=1000,
            total=len(stored_actions),
            episodes=[{"seed": 123, "items": iter(stored_actions)}],
        )
        recorder.close()

    stable_env = FakeStableRetroEnv()
    turbo_vector = FakeTurboVectorEnv(terminal=False)
    turbo_env = main.SuperMarioBrosNesTurboEnvAdapter(turbo_vector)
    asyncio.run(replay_into(stable_env))
    asyncio.run(replay_into(turbo_env))

    stable_actions = [action.tolist() for action in stable_env.actions]
    turbo_actions = [action[0].tolist() for action in turbo_vector.actions]
    assert stable_actions == stored_actions
    assert turbo_actions == stored_actions


def test_recorded_rows_carry_backend_neutral_nes_metadata():
    main._lazy_init()
    recorder = object.__new__(main.DatasetRecorderWrapper)
    recorder.episode_ids = [bytes.fromhex("00" * 16)]
    recorder.seeds = [1]
    recorder.frames = ["frame.webp"]
    recorder.actions = [[1, 0, 0, 0, 0, 0, 0, 1, 1]]
    recorder.rewards = [0.0]
    recorder.terminations = [False]
    recorder.truncations = [False]
    recorder.infos = [json.dumps({})]
    recorder.session_ids = [bytes.fromhex("01" * 16)]
    recorder.collector = "human"
    recorder._gymrec_version = "test"
    recorder.storage_format = main.STORAGE_FORMAT_IMAGES
    recorder.policy_seeds = [None]
    recorder.collector_contract = None
    recorder._env_metadata = {
        "env_id": main.SUPERMARIOBROS_NES_ENV_ID,
        "capture_backend": "supermariobrosnes-turbo",
        "state_name": "Level1-1",
        "rom_sha256": main.SUPERMARIOBROS_NES_ROM_SHA256,
        "nes_button_order": list(main.NES_BUTTON_ORDER),
        "action_encoding": main.NES_ACTION_ENCODING,
        "frameskip": 1,
        "sticky_actions": 0.0,
    }

    dataset = recorder._build_recorded_dataset()

    assert dataset["actions"][0] == recorder.actions[0]
    assert dataset["logical_env_id"] == [main.SUPERMARIOBROS_NES_ENV_ID]
    assert dataset["capture_backend"] == ["supermariobrosnes-turbo"]
    assert json.loads(dataset["nes_button_order"][0]) == list(main.NES_BUTTON_ORDER)
