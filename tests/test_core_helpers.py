import argparse
import asyncio
import json
import shutil
import uuid

import gymnasium as gym
import numpy as np
import pytest

import main


class DatasetLike:
    def __init__(self, columns):
        self._columns = columns
        self.column_names = list(columns)

    def __len__(self):
        if not self._columns:
            return 0
        return len(next(iter(self._columns.values())))

    def __getitem__(self, key):
        if isinstance(key, int):
            return {name: values[key] for name, values in self._columns.items()}
        return self._columns[key]


class OneStepTerminalEnv(gym.Env):
    metadata = {}
    action_space = gym.spaces.Discrete(1)
    observation_space = gym.spaces.Box(0, 255, shape=(1, 1, 3), dtype=np.uint8)

    def __init__(self):
        self.done = False
        self.reset_seeds = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.reset_seeds.append(seed)
        return np.zeros((1, 1, 3), dtype=np.uint8), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("step called after terminal state without reset")
        self.done = True
        return np.ones((1, 1, 3), dtype=np.uint8), 0.0, True, False, {}


class DummySpec:
    max_episode_steps = None
    kwargs = {
        "frameskip": 4,
        "repeat_action_probability": 0.25,
    }


class DummyAtariEnv:
    _env_id = "ALE/Breakout-v5"
    _gymrec_make_kwargs = {
        "frameskip": 1,
        "repeat_action_probability": 0.0,
    }
    spec = DummySpec()
    unwrapped = None
    metadata = {}
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(0, 255, shape=(1, 1, 3), dtype=np.uint8)
    reward_range = (-float("inf"), float("inf"))


class DummyAtariRenderFpsEnv:
    _env_id = "ALE/Breakout-v5"
    _gymrec_make_kwargs = {}
    spec = DummySpec()
    unwrapped = None
    metadata = {"render_fps": 30}
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(0, 255, shape=(1, 1, 3), dtype=np.uint8)
    reward_range = (-float("inf"), float("inf"))


class DummyVizDoomDictObservationEnv:
    _env_id = "VizdoomBasic-v0"
    _vizdoom = True
    _gymrec_make_kwargs = {}
    spec = None
    unwrapped = None
    metadata = {}
    action_space = gym.spaces.MultiBinary(3)
    observation_space = gym.spaces.Dict(
        {
            "screen": gym.spaces.Box(0, 255, shape=(120, 160, 3), dtype=np.uint8),
            "variables": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        }
    )
    reward_range = (-float("inf"), float("inf"))


class DummyStableRetroEnv:
    _env_id = "SuperMarioBros-Nes-v0"
    _stable_retro = True
    _gymrec_stable_retro_state = "Level1-1"
    _gymrec_make_kwargs = {"frame_skip": 4}
    spec = None
    unwrapped = None
    metadata = {}
    action_space = gym.spaces.MultiBinary(8)
    observation_space = gym.spaces.Box(0, 255, shape=(1, 1, 3), dtype=np.uint8)
    reward_range = (-float("inf"), float("inf"))


def make_minimal_dataset(episode_uuid=None, collector="random"):
    main._lazy_init()
    episode_uuid = episode_uuid or uuid.uuid4()
    session_uuid = uuid.uuid4()
    dataset = main.Dataset.from_dict(
        {
            "episode_id": [episode_uuid.bytes, episode_uuid.bytes],
            "seed": [123, 123],
            "actions": [[0], []],
            "rewards": [1.0, None],
            "terminations": [True, None],
            "truncations": [False, None],
            "infos": ["{}", None],
            "session_id": [session_uuid.bytes, session_uuid.bytes],
            "collector": [collector, collector],
            "gymrec_version": ["0.1.0+test", "0.1.0+test"],
            "storage_format": [main.STORAGE_FORMAT_IMAGES, main.STORAGE_FORMAT_IMAGES],
        }
    )
    dataset = dataset.cast_column("episode_id", main.Value("binary"))
    dataset = dataset.cast_column("session_id", main.Value("binary"))
    return dataset


def test_env_id_encoding_round_trips_special_characters():
    env_id = "ALE/Breakout-v5_custom"
    encoded = main._encode_env_id_for_hf(env_id)

    assert encoded == "ALE_slash_Breakout_dash_v5_underscore_custom"
    assert main._decode_hf_repo_name(encoded) == env_id


def test_human_record_env_make_kwargs_force_deterministic_supported_backends():
    assert main._human_record_env_make_kwargs("atari") == {
        "frameskip": 1,
        "repeat_action_probability": 0.0,
    }
    assert main._human_record_env_make_kwargs("stable-retro") == {
        "frame_skip": 1,
        "sticky_action_prob": 0.0,
    }
    assert main._human_record_env_make_kwargs("vizdoom") == {}


def test_env_make_kwargs_from_metadata_recovers_recording_contract():
    metadata = {"frameskip": 4, "sticky_actions": 0.25}

    assert main._env_make_kwargs_from_metadata("ALE/Breakout-v5", metadata, backend="atari") == {
        "frameskip": 4,
        "repeat_action_probability": 0.25,
    }
    assert main._env_make_kwargs_from_metadata(
        "SuperMarioBros-Nes-v0", metadata, backend="stable-retro"
    ) == {
        "frame_skip": 4,
        "sticky_action_prob": 0.25,
    }


def test_env_make_kwargs_from_metadata_prefers_saved_make_kwargs():
    metadata = {
        "frameskip": 4,
        "sticky_actions": 0.25,
        "env_make_kwargs": {"frameskip": 1, "repeat_action_probability": 0.0},
    }

    assert main._env_make_kwargs_from_metadata("ALE/Breakout-v5", metadata, backend="atari") == {
        "frameskip": 1,
        "repeat_action_probability": 0.0,
    }


def test_stable_retro_state_is_recovered_from_recording_metadata(monkeypatch):
    captured = {}

    class CreatedEnv:
        pass

    def fake_create(env_id, state=None, env_make_kwargs=None):
        captured["env_id"] = env_id
        captured["state"] = state
        captured["env_make_kwargs"] = env_make_kwargs
        return CreatedEnv()

    monkeypatch.setattr(main, "_get_stableretro_envs", lambda: ["SuperMarioBros-Nes-v0"])
    monkeypatch.setattr(main, "_create_env__stableretro", fake_create)

    env = main.create_env(
        "SuperMarioBros-Nes-v0",
        metadata={"stable_retro_state": "Level1-1", "env_make_kwargs": {"frame_skip": 4}},
    )

    assert env._env_id == "SuperMarioBros-Nes-v0"
    assert captured == {
        "env_id": "SuperMarioBros-Nes-v0",
        "state": "Level1-1",
        "env_make_kwargs": {"frame_skip": 4},
    }


def test_capture_env_metadata_preserves_stable_retro_state():
    if main.CONFIG is None:
        main.CONFIG = main._load_config()

    metadata = main._capture_env_metadata(DummyStableRetroEnv())

    assert metadata["stable_retro_state"] == "Level1-1"


def test_get_frameskip_and_metadata_prefer_actual_make_kwargs():
    env = DummyAtariEnv()
    if main.CONFIG is None:
        main.CONFIG = main._load_config()

    assert main.get_frameskip(env) == 1

    metadata = main._capture_env_metadata(env)
    assert metadata["frameskip"] == 1
    assert metadata["sticky_actions"] == 0.0
    assert metadata["env_make_kwargs"] == {
        "frameskip": 1,
        "repeat_action_probability": 0.0,
    }


def test_atari_default_fps_uses_sixty_hz_base_not_ale_render_metadata():
    if main.CONFIG is None:
        main.CONFIG = main._load_config()

    assert main.get_default_fps(DummyAtariEnv()) == 60
    assert main.get_default_fps(DummyAtariRenderFpsEnv()) == 15


def test_capture_env_metadata_handles_dict_observation_space():
    env = DummyVizDoomDictObservationEnv()
    if main.CONFIG is None:
        main.CONFIG = main._load_config()

    metadata = main._capture_env_metadata(env)

    assert metadata["observation_space_type"] == "Dict"
    assert "observation_shape" not in metadata
    assert "observation_dtype" not in metadata
    assert "Box" in metadata["observation_space_screen"]
    assert "Box" in metadata["observation_space_variables"]


def test_episode_selection_rejects_multiple_selectors():
    with pytest.raises(ValueError, match="Use only one"):
        main._select_episode_numbers(10, episode_range="1-2", first=1)


def test_episode_selection_handles_range_first_last_and_all():
    assert main._select_episode_numbers(5) == [1, 2, 3, 4, 5]
    assert main._select_episode_numbers(5, episode_range="2-4") == [2, 3, 4]
    assert main._select_episode_numbers(5, first=2) == [1, 2]
    assert main._select_episode_numbers(5, last=2) == [4, 5]


def test_dataset_storage_format_uses_current_columns():
    assert (
        main._dataset_storage_format(DatasetLike({"storage_format": ["lossless-video"]}))
        == main.STORAGE_FORMAT_LOSSLESS_VIDEO
    )
    assert (
        main._dataset_storage_format(DatasetLike({"video_path": ["observation_videos/a"]}))
        == main.STORAGE_FORMAT_LOSSLESS_VIDEO
    )
    assert (
        main._dataset_storage_format(DatasetLike({"observations": ["frame.webp"]}))
        == main.STORAGE_FORMAT_IMAGES
    )


def test_dataset_playback_episodes_verify_items_use_next_observation_row():
    dataset = DatasetLike(
        {
            "episode_id": ["ep1", "ep1", "ep1"],
            "observations": ["obs0", "obs1", "terminal_obs"],
            "actions": [[1], [2], []],
            "rewards": [0.0, 1.0, None],
            "terminations": [False, True, None],
            "truncations": [False, False, None],
        }
    )

    episodes, total_steps = main._dataset_playback_episodes(dataset, verify=True)

    assert total_steps == 2
    assert [list(episode["items"]) for episode in episodes] == [
        [
            ([1], "obs1", 0.0, False, False),
            ([2], "terminal_obs", 1.0, True, False),
        ]
    ]


def test_remote_storage_conflict_message_allows_same_or_empty_remote():
    assert (
        main._remote_storage_conflict_message(
            "BreakoutNoFrameskip-v4",
            "user/gymrec__BreakoutNoFrameskip_dash_v4",
            main.STORAGE_FORMAT_IMAGES,
            None,
        )
        is None
    )
    assert (
        main._remote_storage_conflict_message(
            "BreakoutNoFrameskip-v4",
            "user/gymrec__BreakoutNoFrameskip_dash_v4",
            main.STORAGE_FORMAT_IMAGES,
            main.STORAGE_FORMAT_IMAGES,
        )
        is None
    )


def test_remote_storage_conflict_message_rejects_legacy_video_remote():
    message = main._remote_storage_conflict_message(
        "BreakoutNoFrameskip-v4",
        "user/gymrec__BreakoutNoFrameskip_dash_v4",
        main.STORAGE_FORMAT_LOSSLESS_VIDEO,
        main.REMOTE_STORAGE_FORMAT_LEGACY_VIDEO,
    )

    assert "legacy lossless-video" in message
    assert "--replace" in message


def test_record_plan_rejects_human_headless():
    plan, error = main._make_record_plan(
        argparse.Namespace(agent="human", headless=True, episodes=None, max_steps=None)
    )

    assert plan is None
    assert "--headless" in error


def test_record_plan_defaults_agent_to_one_episode():
    plan, error = main._make_record_plan(
        argparse.Namespace(agent="random", headless=False, episodes=None, max_steps=50)
    )

    assert error is None
    assert plan.agent == "random"
    assert plan.max_episodes == 1
    assert plan.max_steps == 50


@pytest.mark.parametrize("episodes", [0, -1])
def test_record_plan_rejects_non_positive_episode_counts(episodes):
    plan, error = main._make_record_plan(
        argparse.Namespace(agent="random", headless=False, episodes=episodes, max_steps=None)
    )

    assert plan is None
    assert "--episodes must be >= 1" in error


def test_record_plan_requires_episodes_for_headless_agent():
    plan, error = main._make_record_plan(
        argparse.Namespace(agent="random", headless=True, episodes=None, max_steps=None)
    )

    assert plan is None
    assert "--episodes" in error


def test_record_plan_rejects_live_upload_dry_run():
    plan, error = main._make_record_plan(
        argparse.Namespace(
            agent="human",
            headless=False,
            episodes=None,
            max_steps=None,
            upload_live=True,
            dry_run=True,
        )
    )

    assert plan is None
    assert "--upload-live" in error
    assert "--dry-run" in error


def test_record_plan_preserves_live_upload_flag():
    plan, error = main._make_record_plan(
        argparse.Namespace(
            agent="random",
            headless=False,
            episodes=2,
            max_steps=None,
            upload_live=True,
            dry_run=False,
        )
    )

    assert error is None
    assert plan.upload_live is True


def test_agent_input_source_does_not_own_headless_state():
    source = main.AgentInputSource(lambda observation: "action", headless=True)

    assert not hasattr(source, "headless")
    assert source.get_action(None) == "action"


def test_huggingface_model_ref_parses_hf_scheme_and_urls():
    assert main.is_huggingface_model_ref("hf://tsilva/SuperMarioBros-Nes-v0_Level1-1")
    assert main.parse_huggingface_model_ref(
        "hf://tsilva/SuperMarioBros-Nes-v0_Level1-1/model.zip"
    ) == ("tsilva/SuperMarioBros-Nes-v0_Level1-1", "model.zip", None)
    assert main.parse_huggingface_model_ref(
        "https://huggingface.co/tsilva/SuperMarioBros-Nes-v0_Level1-1"
    ) == ("tsilva/SuperMarioBros-Nes-v0_Level1-1", None, None)
    assert main.parse_huggingface_model_ref(
        "https://huggingface.co/tsilva/SuperMarioBros-Nes-v0_Level1-1/blob/main/model.zip"
    ) == ("tsilva/SuperMarioBros-Nes-v0_Level1-1", "model.zip", "main")


def test_huggingface_metadata_extracts_policy_contract():
    metadata = {
        "env_config": {
            "game": "SuperMarioBros-Nes-v0",
            "state": "Level1-1",
            "action_set": "simple",
            "frame_skip": 4,
            "observation_size": 84,
            "obs_crop": [32, 0, 0, 0],
        },
        "environment": {"preprocessing": {"frame_stack": 4}},
    }

    assert main._metadata_str(metadata, ("env_config", "game")) == "SuperMarioBros-Nes-v0"
    assert main._metadata_str(metadata, ("env_config", "state")) == "Level1-1"
    assert main._metadata_int(metadata, ("env_config", "frame_skip"), default=1) == 4
    assert main._metadata_int(
        metadata, ("environment", "preprocessing", "frame_stack"), default=1
    ) == 4
    assert main._metadata_obs_crop(metadata) == (32, 0, 0, 0)


def test_huggingface_policy_source_defaults_to_stochastic_actions():
    source = main.HFPolicySource(
        ref="hf://tsilva/SuperMarioBros-Nes-v0_Level1-1",
        repo_id="tsilva/SuperMarioBros-Nes-v0_Level1-1",
        revision="main",
        checkpoint_filename="model.zip",
        model_path="/tmp/model.zip",
        metadata={},
        env_id="SuperMarioBros-Nes-v0",
        state="Level1-1",
        action_set="simple",
        frame_skip=4,
        frame_stack=4,
        observation_size=84,
        obs_crop=(32, 0, 0, 0),
    )

    assert source.deterministic is False


def test_stable_retro_simple_action_masks_match_nes_button_indices():
    masks = main._stable_retro_action_masks("SuperMarioBros-Nes-v0", "simple")

    assert masks == ((), (7,), (7, 0), (7, 8), (7, 8, 0), (8,), (6,))


def test_dataset_playback_episodes_preserve_episode_seeds_and_actions():
    episode_1 = uuid.uuid4().bytes
    episode_2 = uuid.uuid4().bytes
    dataset = DatasetLike(
        {
            "episode_id": [episode_1, episode_1, episode_2, episode_2],
            "seed": [111, 111, 222, 222],
            "actions": [[0], [], [0], []],
        }
    )

    episodes, total_steps = main._dataset_playback_episodes(dataset)

    assert total_steps == 2
    assert [episode["seed"] for episode in episodes] == [111, 222]
    assert [list(episode["items"]) for episode in episodes] == [[[0]], [[0]]]


def test_replay_resets_between_dataset_episodes():
    async def run_replay():
        env = OneStepTerminalEnv()
        recorder = main.DatasetRecorderWrapper(
            env, headless=True, storage_format=main.STORAGE_FORMAT_IMAGES
        )
        recorder._ensure_screen = lambda frame: None
        recorder._render_frame = lambda frame: None
        recorder._print_keymappings = lambda: None
        recorder._input_loop = lambda: True
        await recorder.replay(
            fps=1000,
            total=2,
            episodes=[
                {"seed": 111, "items": [[0]]},
                {"seed": 222, "items": [[0]]},
            ],
        )
        recorder.close()
        return env.reset_seeds

    assert asyncio.run(run_replay()) == [111, 222]


def test_live_upload_manifest_tracks_pending_failed_and_uploaded(tmp_path):
    if main.CONFIG is None:
        main.CONFIG = main._load_config()
    old_local_dir = main.CONFIG["storage"]["local_dir"]
    main.CONFIG["storage"]["local_dir"] = str(tmp_path)
    package_dir = tmp_path / "episode"
    package_dir.mkdir()
    try:
        main._set_live_upload_manifest_entry(
            "BreakoutNoFrameskip-v4",
            "episode1",
            state="pending",
            package_dir=str(package_dir),
            storage_format=main.STORAGE_FORMAT_LOSSLESS_VIDEO,
            frames=3,
        )
        entries = list(main._pending_live_upload_entries("BreakoutNoFrameskip-v4"))
        assert entries[0][0] == "episode1"
        assert entries[0][1]["state"] == "pending"

        main._set_live_upload_manifest_entry(
            "BreakoutNoFrameskip-v4",
            "episode1",
            state="failed",
            package_dir=str(package_dir),
            storage_format=main.STORAGE_FORMAT_LOSSLESS_VIDEO,
            frames=3,
            error="network",
        )
        manifest = main._load_live_upload_manifest("BreakoutNoFrameskip-v4")
        assert manifest["episodes"]["episode1"]["state"] == "failed"
        assert manifest["episodes"]["episode1"]["error"] == "network"

        main._set_live_upload_manifest_entry(
            "BreakoutNoFrameskip-v4",
            "episode1",
            state="uploaded",
            package_dir=str(package_dir),
            storage_format=main.STORAGE_FORMAT_LOSSLESS_VIDEO,
            frames=3,
        )
        assert list(main._pending_live_upload_entries("BreakoutNoFrameskip-v4")) == []
    finally:
        main.CONFIG["storage"]["local_dir"] = old_local_dir


def test_build_recorded_dataset_includes_video_episode_metadata():
    main._lazy_init()
    recorder = object.__new__(main.DatasetRecorderWrapper)
    episode_id = uuid.uuid4()
    session_id = uuid.uuid4()
    recorder.episode_ids = [episode_id.bytes, episode_id.bytes]
    recorder.seeds = [123, 123]
    recorder.frames = [None, None]
    recorder.actions = [[0], []]
    recorder.rewards = [1.0, None]
    recorder.terminations = [True, None]
    recorder.truncations = [False, None]
    recorder.infos = ["{}", None]
    recorder.session_ids = [session_id.bytes, session_id.bytes]
    recorder.collector = "random"
    recorder._gymrec_version = "0.1.0+test"
    recorder.storage_format = main.STORAGE_FORMAT_LOSSLESS_VIDEO
    recorder.video_paths = ["videos/e.rgb.mkv.bin", "videos/e.rgb.mkv.bin"]
    recorder.frame_indices = [0, 1]
    recorder.frame_sha256s = ["a", "b"]
    recorder.frame_widths = [2, 2]
    recorder.frame_heights = [1, 1]
    recorder.episode_num_observations = [2, 2]

    dataset = recorder._build_recorded_dataset()

    assert len(dataset) == 2
    assert dataset["video_path"] == ["videos/e.rgb.mkv.bin", "videos/e.rgb.mkv.bin"]
    assert dataset["frame_index"] == [0, 1]
    assert dataset["episode_num_observations"] == [2, 2]
    assert dataset["storage_format"] == [main.STORAGE_FORMAT_LOSSLESS_VIDEO] * 2


def test_live_episode_manager_packages_shard_without_previews(tmp_path, monkeypatch):
    if main.CONFIG is None:
        main.CONFIG = main._load_config()
    main._lazy_init()
    old_local_dir = main.CONFIG["storage"]["local_dir"]
    main.CONFIG["storage"]["local_dir"] = str(tmp_path)
    episode_uuid = uuid.uuid4()
    episode_id = episode_uuid.hex
    video_relpath = f"videos/{episode_id}.rgb.mkv.bin"
    dataset = main.Dataset.from_dict(
        {
            "episode_id": [episode_uuid.bytes],
            "seed": [123],
            "actions": [[]],
            "rewards": [None],
            "terminations": [None],
            "truncations": [None],
            "infos": [None],
            "session_id": [uuid.uuid4().bytes],
            "collector": ["random"],
            "gymrec_version": ["0.1.0+test"],
            "storage_format": [main.STORAGE_FORMAT_LOSSLESS_VIDEO],
            "video_path": [video_relpath],
            "frame_index": [0],
            "frame_sha256": ["hash"],
            "frame_width": [2],
            "frame_height": [2],
            "episode_num_observations": [1],
        }
    )
    dataset = dataset.cast_column("episode_id", main.Value("binary"))
    dataset = dataset.cast_column("session_id", main.Value("binary"))
    calls = []

    def fake_upload(env_id, shard, **kwargs):
        calls.append((env_id, shard, kwargs))
        assert kwargs["include_previews"] is False
        assert kwargs["episode_ids"] == {episode_id}
        assert package.parquet_path and package.parquet_path.endswith("episode.parquet")
        assert package_path.exists()
        return True

    monkeypatch.setattr(main, "_upload_dataset_shard_to_hub", fake_upload)
    try:
        manager = main.LiveEpisodeUploadManager(
            "BreakoutNoFrameskip-v4", main.STORAGE_FORMAT_LOSSLESS_VIDEO
        )
        package = manager.begin_episode(episode_uuid)
        package_path = tmp_path / "BreakoutNoFrameskip_dash_v4_live_pending" / episode_id
        video_path = package_path / video_relpath
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(b"video")

        assert manager.upload_episode(package, dataset) is True
        assert calls
        assert not package_path.exists()
        manifest = main._load_live_upload_manifest("BreakoutNoFrameskip-v4")
        assert manifest["episodes"][episode_id]["state"] == "uploaded"
    finally:
        main.CONFIG["storage"]["local_dir"] = old_local_dir


def test_upload_uses_remote_episode_ids_when_local_marker_is_stale(tmp_path, monkeypatch):
    if main.CONFIG is None:
        main.CONFIG = main._load_config()
    old_local_dir = main.CONFIG["storage"]["local_dir"]
    main.CONFIG["storage"]["local_dir"] = str(tmp_path)
    episode_uuid = uuid.uuid4()
    dataset = make_minimal_dataset(episode_uuid=episode_uuid)
    calls = []

    def fake_upload(env_id, shard, **kwargs):
        calls.append((env_id, list(shard["episode_id"]), kwargs))
        return True

    monkeypatch.setattr(main, "ensure_hf_login", lambda: True)
    monkeypatch.setattr(main, "drain_live_upload_queue", lambda *args, **kwargs: True)
    monkeypatch.setattr(main, "_remote_dataset_episode_ids", lambda env_id: set())
    monkeypatch.setattr(main, "_upload_dataset_shard_to_hub", fake_upload)
    try:
        dataset.save_to_disk(main.get_local_dataset_path("BreakoutNoFrameskip-v4"))
        main._save_uploaded_episode_ids("BreakoutNoFrameskip-v4", {episode_uuid.hex})

        assert main.upload_local_dataset("BreakoutNoFrameskip-v4") is True
        assert len(calls) == 1
        assert calls[0][2]["episode_ids"] == {episode_uuid.hex}
    finally:
        main.CONFIG["storage"]["local_dir"] = old_local_dir


def test_upload_local_dataset_fails_without_local_dataset_or_pending_queue(tmp_path, monkeypatch):
    if main.CONFIG is None:
        main.CONFIG = main._load_config()
    old_local_dir = main.CONFIG["storage"]["local_dir"]
    main.CONFIG["storage"]["local_dir"] = str(tmp_path)
    monkeypatch.setattr(main, "ensure_hf_login", lambda: True)
    try:
        assert main.upload_local_dataset("BreakoutNoFrameskip-v4") is False
    finally:
        main.CONFIG["storage"]["local_dir"] = old_local_dir


def test_save_dataset_metadata_recordings_count_only_new_batch(tmp_path):
    if main.CONFIG is None:
        main.CONFIG = main._load_config()
    old_local_dir = main.CONFIG["storage"]["local_dir"]
    main.CONFIG["storage"]["local_dir"] = str(tmp_path)
    try:
        main.save_dataset_locally(
            make_minimal_dataset(collector="random"),
            "BreakoutNoFrameskip-v4",
            metadata={"env_id": "BreakoutNoFrameskip-v4"},
        )
        main.save_dataset_locally(
            make_minimal_dataset(collector="human"),
            "BreakoutNoFrameskip-v4",
            metadata={"env_id": "BreakoutNoFrameskip-v4"},
        )

        with open(main._get_metadata_path("BreakoutNoFrameskip-v4")) as f:
            metadata = json.load(f)

        assert [entry["episodes"] for entry in metadata["recordings"]] == [1, 1]
        assert [entry["frames"] for entry in metadata["recordings"]] == [2, 2]
        assert metadata["recordings"][1]["collectors"] == ["human"]
    finally:
        main.CONFIG["storage"]["local_dir"] = old_local_dir


def test_recover_live_image_journal_adds_truncated_terminal_row(tmp_path):
    main._lazy_init()
    episode_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    frame_path = tmp_path / "frames" / "frame_00000.webp"
    frame_path.parent.mkdir()
    terminal_path = tmp_path / "terminal_candidate.webp"
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    terminal = np.full((2, 2, 3), 64, dtype=np.uint8)
    main.PILImage.fromarray(frame).save(frame_path, format="WEBP", lossless=True)
    main.PILImage.fromarray(terminal).save(terminal_path, format="WEBP", lossless=True)
    records = [
        {
            "row_index": 0,
            "episode_id": episode_id,
            "seed": 123,
            "action": [0],
            "reward": 1.0,
            "termination": False,
            "truncation": False,
            "info": "{}",
            "session_id": session_id,
            "collector": "random",
            "gymrec_version": "0.1.0+test",
            "storage_format": main.STORAGE_FORMAT_IMAGES,
            "observation_path": "frames/frame_00000.webp",
            "terminal_candidate_path": "terminal_candidate.webp",
            "terminal_candidate_sha256": main._sha256_rgb(terminal),
            "terminal_candidate_width": 2,
            "terminal_candidate_height": 2,
        }
    ]

    dataset = main._dataset_from_recovered_live_records(str(tmp_path), episode_id, records)

    assert len(dataset) == 2
    assert dataset["actions"] == [[0], []]
    assert dataset["truncations"] == [True, None]


def test_recover_live_video_journal_appends_terminal_candidate(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    main._lazy_init()
    episode_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    video_relpath = f"videos/{episode_id}.rgb.mkv.bin"
    video_path = tmp_path / video_relpath
    terminal_path = tmp_path / "terminal_candidate.webp"
    step_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    terminal_frame = np.full((2, 2, 3), 128, dtype=np.uint8)
    main._encode_lossless_rgb_video([step_frame], str(video_path), fps=30)
    main.PILImage.fromarray(terminal_frame).save(
        terminal_path, format="WEBP", lossless=True
    )
    records = [
        {
            "row_index": 0,
            "episode_id": episode_id,
            "seed": 123,
            "action": [0],
            "reward": 1.0,
            "termination": False,
            "truncation": False,
            "info": "{}",
            "session_id": session_id,
            "collector": "random",
            "gymrec_version": "0.1.0+test",
            "storage_format": main.STORAGE_FORMAT_LOSSLESS_VIDEO,
            "fps": 30,
            "video_path": video_relpath,
            "frame_index": 0,
            "frame_sha256": main._sha256_rgb(step_frame),
            "frame_width": 2,
            "frame_height": 2,
            "terminal_candidate_path": "terminal_candidate.webp",
            "terminal_candidate_sha256": main._sha256_rgb(terminal_frame),
            "terminal_candidate_width": 2,
            "terminal_candidate_height": 2,
        }
    ]

    dataset = main._dataset_from_recovered_live_records(str(tmp_path), episode_id, records)

    assert len(dataset) == 2
    assert dataset["frame_index"] == [0, 1]
    assert dataset["episode_num_observations"] == [2, 2]
    assert dataset["truncations"] == [True, None]
    main._verify_lossless_rgb_video_stream(
        str(video_path),
        2,
        2,
        [main._sha256_rgb(step_frame), main._sha256_rgb(terminal_frame)],
    )


def test_streaming_video_verifier_accepts_hashes_and_rejects_corruption(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    main._lazy_init()
    frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 255, dtype=np.uint8),
    ]
    video_path = tmp_path / "tiny.rgb.mkv.bin"
    main._encode_lossless_rgb_video(frames, str(video_path), fps=30)
    hashes = [main._sha256_rgb(frame) for frame in frames]

    main._verify_lossless_rgb_video_stream(str(video_path), 2, 2, hashes)
    with pytest.raises(RuntimeError, match="verification failed"):
        main._verify_lossless_rgb_video_stream(str(video_path), 2, 2, ["bad", hashes[1]])
