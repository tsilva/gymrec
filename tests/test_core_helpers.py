import argparse
import asyncio
import hashlib
import json
import shutil
import uuid
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from jinja2 import UndefinedError

import main

TEST_ENVIRONMENT_CONTRACT_ID = "e" * 64


@pytest.fixture(autouse=True)
def isolate_environment_artifacts(monkeypatch):
    """Legacy storage tests focus on storage mechanics; provider artifacts are tested separately."""
    monkeypatch.setattr(main, "_validate_environment_artifacts", lambda dataset, root, **kwargs: dataset)
    monkeypatch.setattr(main, "_contract_upload_operations", lambda *args, **kwargs: [])


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


class RecordingHubApi:
    def __init__(self):
        self.preuploaded = []
        self.commits = []

    def preupload_lfs_files(self, *, additions, **kwargs):
        self.preuploaded = [operation.path_in_repo for operation in additions]

    def create_commit(self, *, operations, commit_message, parent_commit, **kwargs):
        contents = {}
        for operation in operations:
            if operation.__class__.__name__ != "CommitOperationAdd":
                continue
            source = operation.path_or_fileobj
            if hasattr(source, "read"):
                contents[operation.path_in_repo] = source.read()
            else:
                with open(source, "rb") as file_obj:
                    contents[operation.path_in_repo] = file_obj.read()
        self.commits.append(
            {
                "operations": list(operations),
                "message": commit_message,
                "parent_commit": parent_commit,
                "contents": contents,
            }
        )


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


class FakeProviderSession:
    provider_id = "stable-retro-turbo"
    environment_id = "Game-Nes-v0"
    effective_config = {}
    provenance = {"distribution": provider_id, "version": "test", "assets": {}}
    control_profile = "stable_retro.Nes"
    fps = 60

    def __init__(self, env):
        self.env = env

    def recording_observation(self, observation):
        return observation

    def policy_observation(self, observation):
        return observation

    def adapt_policy_action(self, action):
        return action

    def validate_policy(self, policy):
        return None

    def action_from_labels(self, labels):
        return 0


def fake_provider_runtime(env):
    session = FakeProviderSession(env)
    contract = main.EnvironmentContract.parse(
        {
            "contract_version": 1,
            "provider_id": session.provider_id,
            "environment_id": session.environment_id,
            "config": {},
        }
    )
    return session, main._environment_artifact(contract, session)


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
    _gymrec_state_name = "Level1-1"
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
            "episode_id": [str(episode_uuid), str(episode_uuid)],
            "step_index": [0, 1],
            "seed": [123, 123],
            "actions": [0, None],
            "policy_actions": [None, None],
            "rewards": [1.0, None],
            "terminations": [True, None],
            "truncations": [False, None],
            "infos": ["{}", None],
            "session_id": [str(session_uuid), str(session_uuid)],
            "dataset_format_version": [main.DATASET_FORMAT_VERSION] * 2,
            "collector": [collector, collector],
            "gymrec_version": ["0.1.0+test", "0.1.0+test"],
            "storage_format": [main.STORAGE_FORMAT_IMAGES, main.STORAGE_FORMAT_IMAGES],
            "provider_id": ["stable-retro-turbo"] * 2,
            "env_id": ["Game-Nes-v0"] * 2,
            "environment_contract_id": [TEST_ENVIRONMENT_CONTRACT_ID] * 2,
            "collector_contract_id": [None] * 2,
            "policy_mode": [None] * 2,
            "policy_seed": [None] * 2,
            "collector_terminated": [False, False],
            "observations": [
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.ones((1, 1, 3), dtype=np.uint8),
            ],
        }
    )
    return main._recording_dataset_from_dict(dataset.to_dict(), main.STORAGE_FORMAT_IMAGES)


def make_minimal_video_dataset(root, episode_uuid=None):
    main._lazy_init()
    episode_uuid = episode_uuid or uuid.uuid4()
    image_dataset = make_minimal_dataset(episode_uuid=episode_uuid)
    frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 255, dtype=np.uint8),
    ]
    video_relpath = f"videos/{episode_uuid.hex}{main.CANONICAL_VIDEO_SUFFIX}"
    video_path = root / video_relpath
    main._encode_lossless_rgb_video(frames, str(video_path), fps=30)
    data = {field.name: image_dataset[field.name] for field in main.COMMON_DATASET_FIELDS}
    data["storage_format"] = [main.STORAGE_FORMAT_LOSSLESS_VIDEO] * len(frames)
    data.update(
        {
            "video_path": [video_relpath] * len(frames),
            "frame_sha256": [main._sha256_rgb(frame) for frame in frames],
            "frame_width": [2] * len(frames),
            "frame_height": [2] * len(frames),
        }
    )
    return main._recording_dataset_from_dict(data, main.STORAGE_FORMAT_LOSSLESS_VIDEO)


def test_canonical_schema_requires_null_terminal_transition_fields():
    data = make_minimal_dataset().to_dict()
    data["policy_actions"][-1] = 0
    dataset = main._recording_dataset_from_dict(data, main.STORAGE_FORMAT_IMAGES)

    with pytest.raises(ValueError, match="non-null transition fields: policy_actions"):
        main._validate_canonical_dataset_schema(dataset)


def test_canonical_schema_requires_complete_transition_rows():
    data = make_minimal_dataset().to_dict()
    data["infos"][0] = None
    dataset = main._recording_dataset_from_dict(data, main.STORAGE_FORMAT_IMAGES)

    with pytest.raises(ValueError, match="null transition fields: infos"):
        main._validate_canonical_dataset_schema(dataset)


def test_canonical_schema_accepts_collector_terminated_trajectory():
    data = make_minimal_dataset().to_dict()
    data["terminations"][0] = False
    data["collector_terminated"][-1] = True
    dataset = main._recording_dataset_from_dict(data, main.STORAGE_FORMAT_IMAGES)

    assert main._validate_canonical_dataset_schema(dataset) is dataset


def test_canonical_schema_requires_an_explicit_trajectory_boundary():
    data = make_minimal_dataset().to_dict()
    data["terminations"][0] = False
    dataset = main._recording_dataset_from_dict(data, main.STORAGE_FORMAT_IMAGES)

    with pytest.raises(ValueError, match="must be true when the provider did not end"):
        main._validate_canonical_dataset_schema(dataset)


def test_canonical_schema_rejects_collector_flag_after_provider_boundary():
    data = make_minimal_dataset().to_dict()
    data["collector_terminated"][-1] = True
    dataset = main._recording_dataset_from_dict(data, main.STORAGE_FORMAT_IMAGES)

    with pytest.raises(ValueError, match="must be false after a provider"):
        main._validate_canonical_dataset_schema(dataset)


def configure_mock_dataset_upload(monkeypatch, remote_files):
    main._lazy_init()
    api = RecordingHubApi()
    monkeypatch.setattr(main, "HfApi", lambda: api)
    monkeypatch.setattr(main, "_identity_hf_repo_id", lambda identity: "owner/repo")
    monkeypatch.setattr(main, "_current_hf_username", lambda: "owner")
    monkeypatch.setattr(
        main,
        "_hf_repo_state",
        lambda api_arg, repo_id, create=False: (True, "rev", list(remote_files)),
    )
    monkeypatch.setattr(main, "_remote_parquet_columns", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "_next_hf_shard_index", lambda *args, **kwargs: 1)
    monkeypatch.setattr(main, "load_local_metadata", lambda identity: {})
    return api


@pytest.fixture
def temp_storage_dir(tmp_path):
    main._lazy_init()
    previous_local_dir = main.CONFIG["storage"]["local_dir"]
    main.CONFIG["storage"]["local_dir"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        main.CONFIG["storage"]["local_dir"] = previous_local_dir


def test_env_id_encoding_round_trips_special_characters():
    env_id = "ALE/Breakout-v5_custom"
    encoded = main._encode_env_id_for_hf(env_id)

    assert encoded == "ALE_slash_Breakout_dash_v5_underscore_custom"
    assert main._decode_hf_repo_name(encoded) == env_id


@pytest.mark.parametrize("env_id", ["Game_slash_Test-v0", "Game_dash_Test-v0"])
def test_env_id_encoding_does_not_recursively_decode_literal_tokens(env_id):
    assert main._decode_hf_repo_name(main._encode_env_id_for_hf(env_id)) == env_id


def test_config_merge_rejects_unknown_keys_and_wrong_types():
    with pytest.raises(ValueError, match=r"Unknown configuration key in \[storage\]"):
        main._merge_config(main.DEFAULT_CONFIG, {"storage": {"unknown": True}})
    with pytest.raises(ValueError, match="storage.format must be str"):
        main._merge_config(main.DEFAULT_CONFIG, {"storage": {"format": 1}})


def test_dotenv_files_layer_without_overriding_process_environment(tmp_path):
    project = tmp_path / "project.env"
    cwd = tmp_path / "cwd.env"
    project.write_text("PROJECT_ONLY=project\nSHARED=project\n")
    cwd.write_text("CWD_ONLY=cwd\nSHARED=cwd\n")
    environ = {"SHARED": "shell"}

    main._load_environment_files(str(project), str(cwd), environ)

    assert environ == {"PROJECT_ONLY": "project", "CWD_ONLY": "cwd", "SHARED": "shell"}


def test_storage_default_is_resolved_only_by_config(monkeypatch):
    config = {**main.DEFAULT_CONFIG, "storage": dict(main.DEFAULT_CONFIG["storage"])}
    config["storage"]["format"] = main.STORAGE_FORMAT_LOSSLESS_VIDEO
    monkeypatch.setattr(main, "CONFIG", config)

    assert main._configured_storage_format() == main.STORAGE_FORMAT_LOSSLESS_VIDEO
    with pytest.raises(ValueError, match="Storage format is required"):
        main._normalize_storage_format(None)


def test_policy_recording_identity_defaults_to_source_repo_and_accepts_override():
    source = argparse.Namespace(
        env_id="SuperMarioBros-Nes-v0",
        repo_id="tsilva/SuperMarioBros-NES_Level1",
    )

    default_identity = main._policy_recording_identity(source)
    override_identity = main._policy_recording_identity(
        source, "hf://tsilva/mario-world-model-data"
    )

    assert default_identity.env_id == "SuperMarioBros-Nes-v0"
    assert default_identity.dataset_repo_id == "tsilva/SuperMarioBros-NES_Level1"
    assert default_identity.display_ref == "hf://tsilva/SuperMarioBros-NES_Level1"
    assert override_identity.dataset_repo_id == "tsilva/mario-world-model-data"


def test_repo_recording_identities_isolate_same_environment_local_state(temp_storage_dir):
    level1 = main.RecordingIdentity(
        env_id="SuperMarioBros-Nes-v0",
        dataset_repo_id="tsilva/SuperMarioBros-NES_Level1",
    )
    level4 = main.RecordingIdentity(
        env_id="SuperMarioBros-Nes-v0",
        dataset_repo_id="tsilva/SuperMarioBros-Nes-v0_Level4-1",
    )

    assert main.get_local_dataset_path(level1) == str(
        temp_storage_dir / "repos" / "tsilva" / "SuperMarioBros-NES_Level1" / "dataset"
    )
    assert main.get_local_dataset_path(level1) != main.get_local_dataset_path(level4)
    assert main._get_metadata_path(level1).endswith(
        "repos/tsilva/SuperMarioBros-NES_Level1/metadata.json"
    )
    assert main._get_uploaded_episodes_path(level1).endswith(
        "repos/tsilva/SuperMarioBros-NES_Level1/uploaded.json"
    )
    assert main._get_live_upload_queue_dir(level1).endswith(
        "repos/tsilva/SuperMarioBros-NES_Level1/live_pending"
    )
    paths = main._recording_paths(level1)
    assert paths.dataset == main.get_local_dataset_path(level1)
    assert paths.metadata == main._get_metadata_path(level1)
    assert paths.uploaded == main._get_uploaded_episodes_path(level1)
    assert paths.live_queue == main._get_live_upload_queue_dir(level1)


def test_repo_recording_save_persists_identity_and_is_discoverable(temp_storage_dir):
    identity = main.RecordingIdentity(
        env_id="SuperMarioBros-Nes-v0",
        dataset_repo_id="tsilva/SuperMarioBros-NES_Level1",
    )

    main.save_dataset_locally(
        make_minimal_dataset(collector="hf://tsilva/SuperMarioBros-NES_Level1"),
        identity,
        artifact_root=str(temp_storage_dir),
        metadata={"env_id": identity.env_id},
    )

    assert "hf://tsilva/SuperMarioBros-NES_Level1" in (
        main._get_available_recording_refs_from_local()
    )
    assert main.load_local_metadata(identity)["dataset_repo_id"] == identity.dataset_repo_id


def test_load_recorded_dataset_repo_ref_uses_exact_dataset_repo(monkeypatch):
    hub_dataset = make_minimal_dataset()
    calls = []
    monkeypatch.setattr(main, "load_local_dataset", lambda identity: None)
    monkeypatch.setattr(
        main,
        "load_dataset",
        lambda repo_id, split: calls.append((repo_id, split)) or hub_dataset,
        raising=False,
    )
    monkeypatch.setattr(
        main,
        "_attach_video_runtime_source",
        lambda dataset, hf_repo_id: dataset,
    )

    assert main.load_recorded_dataset("hf://tsilva/SuperMarioBros-NES_Level1") == (
        hub_dataset,
        "hub",
    )
    assert calls == [("tsilva/SuperMarioBros-NES_Level1", "train")]






















def test_episode_selection_rejects_multiple_selectors():
    with pytest.raises(ValueError, match="Use only one"):
        main._select_episode_numbers(10, episode_range="1-2", first=1)


def test_episode_selection_handles_range_first_last_and_all():
    assert main._select_episode_numbers(5) == [1, 2, 3, 4, 5]
    assert main._select_episode_numbers(5, episode_range="2-4") == [2, 3, 4]
    assert main._select_episode_numbers(5, first=2) == [1, 2]
    assert main._select_episode_numbers(5, last=2) == [4, 5]


def test_dataset_storage_format_requires_current_explicit_metadata():
    assert (
        main._dataset_storage_format(DatasetLike({"storage_format": ["lossless-video"]}))
        == main.STORAGE_FORMAT_LOSSLESS_VIDEO
    )
    with pytest.raises(ValueError, match="required storage_format"):
        main._dataset_storage_format(DatasetLike({"video_path": ["videos/e.rgb.mkv.bin"]}))
    with pytest.raises(ValueError, match="required storage_format"):
        main._dataset_storage_format(DatasetLike({"observations": ["frame.webp"]}))
    with pytest.raises(ValueError, match="every row"):
        main._dataset_storage_format(DatasetLike({"storage_format": [None]}))
    with pytest.raises(ValueError, match="mixed storage_format"):
        main._dataset_storage_format(DatasetLike({"storage_format": ["images", "lossless-video"]}))


def test_get_gymrec_version_uses_installed_metadata_outside_checkout(monkeypatch):
    monkeypatch.setattr(main, "_installed_package_version", lambda name: "1.2.3")
    monkeypatch.setattr(main.os.path, "isfile", lambda path: False)

    assert main._get_gymrec_version() == "1.2.3"


def test_get_gymrec_version_adds_git_hash_in_checkout(monkeypatch):
    monkeypatch.setattr(main, "_installed_package_version", lambda name: "1.2.3")
    monkeypatch.setattr(main.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=0, stdout="abc1234\n"),
    )

    assert main._get_gymrec_version() == "1.2.3+abc1234"


def test_missing_subcommand_uses_record_parser_defaults_and_preserves_global_args():
    parser = argparse.ArgumentParser()
    main._add_provider_arg(parser)
    subparsers = parser.add_subparsers(dest="command")
    record_parser = subparsers.add_parser("record")
    main._add_provider_arg(record_parser, default=argparse.SUPPRESS)
    main._add_env_id_arg(record_parser)
    record_parser.add_argument("--agent", default="human")

    args = main._parse_cli_args(parser, ["--provider", "stable-retro-turbo"])

    assert args.command == "record"
    assert args.env_id is None
    assert args.agent == "human"
    assert args.provider == "stable-retro-turbo"


def test_load_recorded_dataset_prefers_local_without_hub_lookup(monkeypatch):
    local_dataset = DatasetLike({"actions": [[0]]})
    monkeypatch.setattr(main, "load_local_dataset", lambda env_id: local_dataset)

    def fail_hub_lookup(*args, **kwargs):
        raise AssertionError("Hub lookup should not run when local data exists")

    monkeypatch.setattr(main, "load_dataset", fail_hub_lookup, raising=False)

    assert main.load_recorded_dataset("BreakoutNoFrameskip-v4") == (
        local_dataset,
        "local",
    )


def test_load_recorded_dataset_uses_single_hub_load(monkeypatch):
    hub_dataset = make_minimal_dataset()
    calls = []
    monkeypatch.setattr(main, "load_local_dataset", lambda env_id: None)
    monkeypatch.setattr(main, "env_id_to_hf_repo_id", lambda env_id: "user/gymrec__env")
    monkeypatch.setattr(
        main,
        "load_dataset",
        lambda repo_id, split: calls.append((repo_id, split)) or hub_dataset,
        raising=False,
    )
    monkeypatch.setattr(
        main,
        "_attach_video_runtime_source",
        lambda dataset, hf_repo_id: dataset,
    )

    assert main.load_recorded_dataset("BreakoutNoFrameskip-v4") == (
        hub_dataset,
        "hub",
    )
    assert calls == [("user/gymrec__env", "train")]


def test_dataset_playback_episodes_verify_items_use_next_observation_row():
    dataset = DatasetLike(
        {
            "episode_id": ["ep1", "ep1", "ep1"],
            "observations": ["obs0", "obs1", "terminal_obs"],
            "actions": [1, 2, None],
            "rewards": [0.0, 1.0, None],
            "terminations": [False, True, None],
            "truncations": [False, False, None],
            "environment_contract_id": [TEST_ENVIRONMENT_CONTRACT_ID] * 3,
        }
    )

    episodes, total_steps = main._dataset_playback_episodes(dataset, verify=True)

    assert total_steps == 2
    assert [list(episode["items"]) for episode in episodes] == [
        [
            (1, "obs1", 0.0, False, False),
            (2, "terminal_obs", 1.0, True, False),
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


def test_remote_storage_conflict_message_rejects_unsupported_remote_layout():
    message = main._remote_storage_conflict_message(
        "BreakoutNoFrameskip-v4",
        "user/gymrec__BreakoutNoFrameskip_dash_v4",
        main.STORAGE_FORMAT_LOSSLESS_VIDEO,
        main.REMOTE_STORAGE_FORMAT_UNSUPPORTED,
    )

    assert "unsupported" in message
    assert "--replace" in message


def test_record_plan_rejects_human_headless():
    plan, error = main._make_record_plan(
        argparse.Namespace(agent="human", headless=True, episodes=None, max_steps=None)
    )

    assert plan is None
    assert "--headless" in error


def test_record_plan_defaults_agent_to_one_episode():
    plan, error = main._make_record_plan(
        argparse.Namespace(agent="random", headless=False, episodes=None)
    )

    assert error is None
    assert plan.agent == "random"
    assert plan.max_episodes == 1


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


def test_record_plan_uses_the_requested_seed_everywhere():
    plan, error = main._make_record_plan(
        argparse.Namespace(
            agent="random",
            headless=False,
            episodes=2,
            max_steps=None,
            upload_live=False,
            dry_run=True,
            seed=123,
        )
    )

    assert error is None
    assert plan.seed == 123


def test_record_plan_defaults_to_reproducible_zero_seed():
    plan, error = main._make_record_plan(
        argparse.Namespace(
            agent="hf",
            headless=False,
            episodes=3,
            max_steps=None,
            upload_live=False,
            dry_run=True,
            seed=None,
        )
    )

    assert error is None
    assert plan.seed == main.DEFAULT_RECORD_SEED == 0
    np.random.seed(plan.seed + plan.max_episodes - 1)


def test_record_plan_rejects_seed_that_overflows_across_episodes():
    plan, error = main._make_record_plan(
        argparse.Namespace(
            agent="hf",
            headless=False,
            episodes=2,
            max_steps=None,
            upload_live=False,
            dry_run=True,
            seed=main.MAX_COMPATIBLE_SEED,
        )
    )

    assert plan is None
    assert "--seed must be <=" in error


def test_playback_hint_is_printed_only_after_recording_upload_finishes(monkeypatch):
    events = []
    identity = main.RecordingIdentity(env_id="BreakoutNoFrameskip-v4")

    def confirm_upload(*_args, **_kwargs):
        events.append("prompt")
        return True

    def upload(_identity):
        events.append("upload")
        return True

    monkeypatch.setattr(main.Confirm, "ask", confirm_upload)
    monkeypatch.setattr(main, "upload_local_dataset", upload)
    monkeypatch.setattr(main.console, "print", lambda message: events.append(str(message)))

    main._finish_recording_publication(identity, dry_run=False)

    assert events[:2] == ["prompt", "upload"]
    assert events[-1].startswith("To play back:")


def test_agent_input_source_wraps_policy_only():
    source = main.AgentInputSource(lambda observation: "action")

    assert not hasattr(source, "headless")
    assert source.get_action(None) == "action"


def test_huggingface_model_ref_parses_hf_scheme_and_urls():
    assert main.is_huggingface_model_ref("hf://tsilva/SuperMarioBros-Nes-v0_Level1-1")
    assert main.parse_huggingface_model_ref(
        "hf://tsilva/SuperMarioBros-Nes-v0_Level1-1/model.zip"
    ) == ("tsilva/SuperMarioBros-Nes-v0_Level1-1", "model.zip", None)
    assert main.parse_huggingface_model_ref(
        "hf://tsilva/SuperMarioBros-Nes-v0_Level1-1@" + "a" * 40
    ) == ("tsilva/SuperMarioBros-Nes-v0_Level1-1", None, "a" * 40)
    assert main.parse_huggingface_model_ref(
        "https://huggingface.co/tsilva/SuperMarioBros-Nes-v0_Level1-1"
    ) == ("tsilva/SuperMarioBros-Nes-v0_Level1-1", None, None)
    assert main.parse_huggingface_model_ref(
        "https://huggingface.co/tsilva/SuperMarioBros-Nes-v0_Level1-1/blob/main/model.zip"
    ) == ("tsilva/SuperMarioBros-Nes-v0_Level1-1", "model.zip", "main")


def test_huggingface_policy_source_defaults_to_stochastic_actions():
    environment = {
        "contract_version": 1,
        "provider_id": "supermariobrosnes-turbo",
        "environment_id": "SuperMarioBros-Nes-v0",
        "config": {
            "state": "Level1-1",
            "frame_skip": 4,
            "obs_resize": [84, 84],
            "frame_stack": 4,
            "obs_grayscale": True,
            "obs_layout": "chw",
            "task": {"id": "arbitrary-provider-owned-task"},
        },
    }
    source = main.HFPolicySource(
        repo_id="tsilva/SuperMarioBros-Nes-v0_Level1-1",
        revision="main",
        checkpoint_filename="model.zip",
        model_path="/tmp/model.zip",
        model_json_path="/tmp/model.json",
        recipe_json_path="/tmp/recipe.json",
        release_manifest_path=None,
        model_document={
            "checkpoint": {"sha256": "checkpoint"},
        },
        environment=environment,
    )

    assert source.deterministic is False
    assert source.provider == "supermariobrosnes-turbo"
    assert source.env_id == "SuperMarioBros-Nes-v0"
    assert source.environment_contract.config == environment["config"]


def _write_policy_bundle(tmp_path, *, recipe_format_version=1, checkpoint_sha256=None):
    checkpoint_path = tmp_path / "model.zip"
    checkpoint_path.write_bytes(b"policy checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    recipe = {
        "document_type": "rlab.recipe",
        "format_version": recipe_format_version,
        "provenance": {},
        "recipe": {
            "schema_version": 2,
            "eval": {
                "action_sampling": "stochastic",
                "episodes": 100,
                "max_steps": 4500,
                "n_envs": 20,
                "seed": 10000,
                "seed_protocol": "vector-lane-v1",
                "environment": {
                    "contract_version": 1,
                    "provider_id": "supermariobrosnes-turbo",
                    "environment_id": "SuperMarioBros-Nes-v0",
                    "config": {
                        "state": "Level1-1",
                        "frame_skip": 4,
                        "maxpool_last_two": False,
                        "obs_resize": [84, 84],
                        "obs_crop": [32, 0, 0, 0],
                        "obs_crop_mode": "remove",
                        "obs_crop_fill": 0,
                        "obs_resize_algorithm": "area",
                        "sticky_action_prob": 0.0,
                        "frame_stack": 4,
                        "obs_copy": "safe_view",
                        "obs_grayscale": True,
                        "obs_layout": "chw",
                        "use_restricted_actions": "filtered",
                        "task": {
                            "id": "arbitrary-provider-owned-task",
                            "action": {"set": "simple"},
                        },
                    },
                },
            },
        },
    }
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))
    recipe_hash = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    model = {
        "document_type": "rlab.model",
        "format_version": 1,
        "checkpoint": {
            "filename": "model.zip",
            "sha256": checkpoint_sha256 or checkpoint_hash,
            "size_bytes": checkpoint_path.stat().st_size,
            "algorithm_id": "ppo",
            "model_class": "stable_baselines3.ppo.ppo.PPO",
        },
        "recipe": {
            "filename": "recipe.json",
            "document_type": "rlab.recipe",
            "format_version": recipe_format_version,
            "sha256": recipe_hash,
            "size_bytes": recipe_path.stat().st_size,
        },
        "policy": {
            "algorithm_id": "ppo",
            "model_class": "stable_baselines3.ppo.ppo.PPO",
        },
        "provenance": {},
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model, sort_keys=True))
    return {
        "model.json": str(model_path),
        "recipe.json": str(recipe_path),
        "model.zip": str(checkpoint_path),
    }


def _mock_policy_repo(monkeypatch, files):
    main._lazy_init()
    monkeypatch.setattr(
        main,
        "_resolve_huggingface_model_commit",
        lambda _repo_id, _revision: ("a" * 40, set(files)),
    )
    monkeypatch.setattr(
        main,
        "hf_hub_download",
        lambda *, filename, **_kwargs: files[filename],
    )


def _add_release_manifest(tmp_path, files, *, recipe_sha256=None):
    artifacts = {}
    for filename, path in files.items():
        payload = open(path, "rb").read()
        artifacts[filename] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    if recipe_sha256 is not None:
        artifacts["recipe.json"]["sha256"] = recipe_sha256
    manifest = {
        "document_type": "rlab.release_manifest",
        "format_version": 1,
        "repository": {"repo_id": "tsilva/level1-1"},
        "artifacts": artifacts,
    }
    path = tmp_path / "release_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    files["release_manifest.json"] = str(path)


def test_resolve_huggingface_policy_consumes_versioned_recipe_bundle(tmp_path, monkeypatch):
    files = _write_policy_bundle(tmp_path)
    _mock_policy_repo(monkeypatch, files)

    source = main.resolve_huggingface_policy_source("hf://tsilva/level1-1")

    assert source.provider == "supermariobrosnes-turbo"
    assert source.environment_contract.config["state"] == "Level1-1"
    assert source.deterministic is False
    assert source.revision == "a" * 40
    assert source.collector == f"hf://tsilva/level1-1@{'a' * 40}"
    assert source.environment == json.loads(open(files["recipe.json"]).read())["recipe"]["eval"]["environment"]


def test_resolve_huggingface_policy_rejects_unknown_recipe_version(tmp_path, monkeypatch):
    files = _write_policy_bundle(tmp_path, recipe_format_version=99)
    _mock_policy_repo(monkeypatch, files)

    with pytest.raises(SystemExit, match="Unsupported recipe.json format_version 99"):
        main.resolve_huggingface_policy_source("hf://tsilva/level1-1")


def test_resolve_huggingface_policy_rejects_checkpoint_hash_mismatch(tmp_path, monkeypatch):
    files = _write_policy_bundle(tmp_path, checkpoint_sha256="0" * 64)
    _mock_policy_repo(monkeypatch, files)

    with pytest.raises(SystemExit, match="model.zip SHA-256 mismatch"):
        main.resolve_huggingface_policy_source("hf://tsilva/level1-1")


def test_resolve_huggingface_policy_validates_optional_release_manifest(tmp_path, monkeypatch):
    files = _write_policy_bundle(tmp_path)
    _add_release_manifest(tmp_path, files)
    _mock_policy_repo(monkeypatch, files)

    source = main.resolve_huggingface_policy_source("hf://tsilva/level1-1")

    assert source.release_manifest_path == files["release_manifest.json"]


def test_resolve_huggingface_policy_rejects_inconsistent_release_manifest(tmp_path, monkeypatch):
    files = _write_policy_bundle(tmp_path)
    _add_release_manifest(tmp_path, files, recipe_sha256="0" * 64)
    _mock_policy_repo(monkeypatch, files)

    with pytest.raises(SystemExit, match="release artifact recipe.json SHA-256 mismatch"):
        main.resolve_huggingface_policy_source("hf://tsilva/level1-1")




def test_collector_contract_id_is_stable_seed_independent_and_materialized_once(
    tmp_path, monkeypatch
):
    files = _write_policy_bundle(tmp_path)
    _mock_policy_repo(monkeypatch, files)
    source = main.resolve_huggingface_policy_source("hf://tsilva/level1-1")
    provider_session = argparse.Namespace(
        env=argparse.Namespace(
            observation_space=gym.spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8),
            action_space=gym.spaces.Discrete(7),
        )
    )

    def build(*, device="cpu", source_value=source, environment_id="e" * 64):
        return main.build_collector_contract(
            source_value,
            provider_session,
            environment_contract_id=environment_id,
            inference_device=device,
        )

    first = build()
    second = build()
    changed_device = build(device="mps")
    changed_mode = build(source_value=replace(source, deterministic=True))
    changed_environment = build(environment_id="f" * 64)
    main._materialize_collector_contract(first, tmp_path / "dataset")
    main._materialize_collector_contract(first, tmp_path / "dataset")

    assert first.contract_id == second.contract_id
    assert first.contract_id != changed_device.contract_id
    assert first.contract_id != changed_mode.contract_id
    assert first.contract_id != changed_environment.contract_id
    assert first.collection_document["execution"]["environment_contract_id"] == "e" * 64
    assert first.collection_document["execution"]["recorded_actions"] == "exact-env-step-input"
    collector_dir = tmp_path / "dataset" / "collectors" / first.contract_id
    assert sorted(path.name for path in collector_dir.iterdir()) == [
        "collection.json",
        "model.json",
        "recipe.json",
    ]






def test_dataset_playback_episodes_preserve_episode_seeds_and_actions():
    episode_1 = uuid.uuid4().bytes
    episode_2 = uuid.uuid4().bytes
    dataset = DatasetLike(
        {
            "episode_id": [episode_1, episode_1, episode_2, episode_2],
            "environment_contract_id": ["e" * 64] * 4,
            "seed": [111, 111, 222, 222],
            "actions": [0, None, 0, None],
        }
    )

    episodes, total_steps = main._dataset_playback_episodes(dataset)

    assert total_steps == 2
    assert [episode["seed"] for episode in episodes] == [111, 222]
    assert [episode["step_count"] for episode in episodes] == [1, 1]
    assert [list(episode["items"]) for episode in episodes] == [[0], [0]]


def test_dataset_playback_episode_counts_do_not_consume_lazy_items():
    dataset = DatasetLike(
        {
            "episode_id": ["ep1", "ep1", "ep1"],
            "environment_contract_id": ["e" * 64] * 3,
            "seed": [123, 123, 123],
            "actions": [1, 2, None],
        }
    )

    episodes, total_steps = main._dataset_playback_episodes(dataset)
    group_total = sum(episode["step_count"] for episode in episodes)

    assert total_steps == 2
    assert group_total == 2
    assert [list(episode["items"]) for episode in episodes] == [[1, 2]]


def test_agent_recording_uses_sequential_recipe_seed_base():
    env = OneStepTerminalEnv()
    provider_session, environment_artifact = fake_provider_runtime(env)
    policy_resets = []

    class SeededPolicy:
        last_policy_action = None

        def reset(self, *, seed=None, **_kwargs):
            policy_resets.append(seed)

        def __call__(self, _observation):
            self.last_policy_action = 4
            return 0

    recorder = main.DatasetRecorderWrapper(
        input_source=main.AgentInputSource(SeededPolicy()),
        headless=True,
        storage_format=main.STORAGE_FORMAT_IMAGES,
        initial_seed=10000,
        provider_session=provider_session,
        environment_artifact=environment_artifact,
    )
    recorder.collector_contract = argparse.Namespace(contract_id="a" * 64, policy_mode="stochastic")

    dataset = asyncio.run(recorder.record(fps=1000, max_episodes=2))

    assert env.reset_seeds == [10000, 10001]
    assert sorted(set(dataset["seed"])) == [10000, 10001]
    assert policy_resets == [10000, 10001]
    assert set(dataset["collector_contract_id"]) == {"a" * 64}
    assert dataset["step_index"] == [0, 1, 0, 1]
    assert dataset["actions"] == [0, None, 0, None]
    assert dataset["policy_actions"] == [4, None, 4, None]
    assert dataset["policy_seed"] == [10000, 10000, 10001, 10001]


def test_replay_resets_between_dataset_episodes():
    async def run_replay():
        env = OneStepTerminalEnv()
        provider_session, environment_artifact = fake_provider_runtime(env)
        recorder = main.DatasetRecorderWrapper(
            headless=True,
            storage_format=main.STORAGE_FORMAT_IMAGES,
            provider_session=provider_session,
            environment_artifact=environment_artifact,
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


def test_live_upload_manifest_tracks_pending_failed_and_uploaded(temp_storage_dir):
    package_dir = temp_storage_dir / "episode"
    package_dir.mkdir()
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


def test_materialize_dataset_replay_streams_image_rows(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    output_path = tmp_path / main.DATASET_REPLAY_FILENAME

    main._materialize_dataset_replay(
        make_minimal_dataset(),
        str(tmp_path),
        str(output_path),
        fps=30,
    )

    assert main._verify_browser_preview_video(output_path)["frames"] == 2


def test_materialize_dataset_replay_accepts_collector_terminated_trajectory(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    data = make_minimal_dataset().to_dict()
    data["terminations"][0] = False
    data["collector_terminated"][-1] = True
    dataset = main._recording_dataset_from_dict(data, main.STORAGE_FORMAT_IMAGES)
    output_path = tmp_path / main.DATASET_REPLAY_FILENAME

    main._materialize_dataset_replay(dataset, str(tmp_path), str(output_path), fps=30)

    assert main._verify_browser_preview_video(output_path)["frames"] == 2


def test_materialize_dataset_replay_skips_incomplete_first_episode(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    incomplete = make_minimal_dataset().select([0])
    complete = make_minimal_dataset()
    combined = main._recording_dataset_from_dict(
        {
            column: list(incomplete[column]) + list(complete[column])
            for column in incomplete.column_names
        },
        main.STORAGE_FORMAT_IMAGES,
    )
    output_path = tmp_path / main.DATASET_REPLAY_FILENAME

    main._materialize_dataset_replay(combined, str(tmp_path), str(output_path), fps=30)

    assert main._verify_browser_preview_video(output_path)["frames"] == 2


def test_materialize_dataset_replay_reuses_episode_preview(tmp_path, monkeypatch):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    episode_uuid = uuid.uuid4()
    dataset = make_minimal_video_dataset(tmp_path, episode_uuid=episode_uuid)
    preview_path = tmp_path / main._preview_video_relpath(episode_uuid.hex)
    preview_frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), 127, dtype=np.uint8),
    ]
    main._encode_browser_preview_video(preview_frames, str(preview_path), fps=30)
    output_path = tmp_path / main.DATASET_REPLAY_FILENAME
    monkeypatch.setattr(
        main,
        "_transcode_browser_preview_video",
        lambda *args, **kwargs: pytest.fail("existing preview should be reused"),
    )

    main._materialize_dataset_replay(dataset, str(tmp_path), str(output_path), fps=30)

    assert output_path.read_bytes() == preview_path.read_bytes()


def test_materialize_dataset_replay_transcodes_live_canonical_stream(tmp_path, monkeypatch):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")
    dataset = make_minimal_video_dataset(tmp_path)
    output_path = tmp_path / main.DATASET_REPLAY_FILENAME
    original_transcode = main._transcode_browser_preview_video
    calls = []

    def tracked_transcode(input_path, replay_path, ffmpeg_path=None):
        calls.append(input_path)
        return original_transcode(input_path, replay_path, ffmpeg_path=ffmpeg_path)

    monkeypatch.setattr(main, "_transcode_browser_preview_video", tracked_transcode)

    main._materialize_dataset_replay(dataset, str(tmp_path), str(output_path), fps=30)

    assert calls == [str(tmp_path / dataset[0]["video_path"])]
    assert main._verify_browser_preview_video(output_path)["frames"] == 2


@pytest.mark.parametrize(
    ("probe", "message"),
    [
        ({"streams": [], "format": {"duration": "1.0"}}, "video stream"),
        (
            {
                "streams": [
                    {
                        "codec_name": "vp9",
                        "codec_tag_string": "avc1",
                        "pix_fmt": "yuv420p",
                        "nb_read_frames": "2",
                    }
                ],
                "format": {"duration": "1.0"},
            },
            "codec_name",
        ),
        (
            {
                "streams": [
                    {
                        "codec_name": "h264",
                        "codec_tag_string": "avc1",
                        "pix_fmt": "yuv444p",
                        "nb_read_frames": "2",
                    }
                ],
                "format": {"duration": "1.0"},
            },
            "pix_fmt",
        ),
        (
            {
                "streams": [
                    {
                        "codec_name": "h264",
                        "codec_tag_string": "avc1",
                        "pix_fmt": "yuv420p",
                        "nb_read_frames": "0",
                    }
                ],
                "format": {"duration": "0"},
            },
            "positive duration",
        ),
    ],
)
def test_verify_browser_preview_video_rejects_invalid_probe(tmp_path, monkeypatch, probe, message):
    video_path = tmp_path / "replay.mp4"
    video_path.write_bytes(b"xxxxmoovxxxxmdat")
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(stdout=json.dumps(probe)),
    )

    with pytest.raises(ValueError, match=message):
        main._verify_browser_preview_video(video_path, ffprobe_path="ffprobe")


def test_verify_browser_preview_video_requires_faststart(tmp_path, monkeypatch):
    video_path = tmp_path / "replay.mp4"
    video_path.write_bytes(b"xxxxmdatxxxxmoov")
    probe = {
        "streams": [
            {
                "codec_name": "h264",
                "codec_tag_string": "avc1",
                "pix_fmt": "yuv420p",
                "nb_read_frames": "2",
            }
        ],
        "format": {"duration": "1.0"},
    }
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(stdout=json.dumps(probe)),
    )

    with pytest.raises(ValueError, match="moov before mdat"):
        main._verify_browser_preview_video(video_path, ffprobe_path="ffprobe")


def test_build_recorded_dataset_includes_video_episode_metadata():
    main._lazy_init()
    recorder = object.__new__(main.DatasetRecorderWrapper)
    episode_id = uuid.uuid4()
    session_id = uuid.uuid4()
    recorder.storage_format = main.STORAGE_FORMAT_LOSSLESS_VIDEO
    shared = {
        "episode_id": str(episode_id),
        "step_index": 0,
        "seed": 123,
        "policy_actions": None,
        "session_id": str(session_id),
        "dataset_format_version": main.DATASET_FORMAT_VERSION,
        "collector": "random",
        "gymrec_version": "0.1.0+test",
        "storage_format": main.STORAGE_FORMAT_LOSSLESS_VIDEO,
        "env_id": None,
        "capture_backend": None,
        "state_name": None,
        "rom_sha256": None,
        "action_encoding": None,
        "action_labels": None,
        "frame_skip": None,
        "sticky_action_prob": None,
        "collector_contract_id": None,
        "policy_mode": None,
        "policy_seed": None,
        "video_path": "videos/e.rgb.mkv.bin",
        "frame_width": 2,
        "frame_height": 1,
    }
    recorder._recording_rows = [
        {
            **shared,
            "actions": 0,
            "rewards": 1.0,
            "terminations": True,
            "truncations": False,
            "infos": "{}",
            "frame_sha256": "a",
        },
        {
            **shared,
            "step_index": 1,
            "actions": None,
            "rewards": None,
            "terminations": None,
            "truncations": None,
            "infos": None,
            "frame_sha256": "b",
        },
    ]

    dataset = recorder._build_recorded_dataset()

    assert len(dataset) == 2
    assert dataset["video_path"] == ["videos/e.rgb.mkv.bin", "videos/e.rgb.mkv.bin"]
    assert dataset["storage_format"] == [main.STORAGE_FORMAT_LOSSLESS_VIDEO] * 2


@pytest.mark.parametrize("upload_success", [True, False])
def test_live_episode_manager_packages_shard_without_previews(
    temp_storage_dir, monkeypatch, upload_success
):
    main._lazy_init()
    episode_uuid = uuid.uuid4()
    episode_id = str(episode_uuid)
    video_relpath = f"videos/{episode_uuid.hex}.rgb.mkv.bin"
    dataset = main.Dataset.from_dict(
        {
            "episode_id": [str(episode_uuid)],
            "step_index": [0],
            "seed": [123],
            "actions": [None],
            "policy_actions": [None],
            "rewards": [None],
            "terminations": [None],
            "truncations": [None],
            "infos": [None],
            "session_id": [str(uuid.uuid4())],
            "dataset_format_version": [main.DATASET_FORMAT_VERSION],
            "collector": ["random"],
            "gymrec_version": ["0.1.0+test"],
            "storage_format": [main.STORAGE_FORMAT_LOSSLESS_VIDEO],
            "video_path": [video_relpath],
            "frame_sha256": ["hash"],
            "frame_width": [2],
            "frame_height": [2],
        }
    )
    calls = []

    def fake_upload(env_id, shard, **kwargs):
        calls.append((env_id, shard, kwargs))
        assert kwargs["include_previews"] is False
        assert kwargs["preview_fps"] == 47
        assert kwargs["episode_ids"] == {episode_id}
        assert (package_path / "episode.parquet").exists()
        return upload_success

    monkeypatch.setattr(main, "_upload_dataset_shard_to_hub", fake_upload)
    _session, environment_artifact = fake_provider_runtime(OneStepTerminalEnv())
    manager = main.LiveEpisodeUploadManager(
        "BreakoutNoFrameskip-v4",
        main.STORAGE_FORMAT_LOSSLESS_VIDEO,
        environment_artifact=environment_artifact,
        fps=47,
    )
    package = manager.begin_episode(episode_uuid)
    package_path = temp_storage_dir / "BreakoutNoFrameskip_dash_v4_live_pending" / episode_id
    video_path = package_path / video_relpath
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")

    assert manager.upload_episode(package, dataset) is upload_success
    assert calls
    manifest = main._load_live_upload_manifest("BreakoutNoFrameskip-v4")
    expected_state = "uploaded" if upload_success else "failed"
    assert manifest["episodes"][episode_id]["state"] == expected_state
    assert manifest["episodes"][episode_id]["fps"] == 47.0
    assert package_path.exists() is not upload_success


def test_live_upload_retry_uses_manifest_fps(temp_storage_dir, monkeypatch):
    dataset = make_minimal_dataset()
    episode_id = dataset[0]["episode_id"]
    package_dir = temp_storage_dir / "retry-package"
    package_dir.mkdir()
    dataset.to_parquet(package_dir / "episode.parquet")
    main._set_live_upload_manifest_entry(
        "BreakoutNoFrameskip-v4",
        episode_id,
        state="failed",
        package_dir=str(package_dir),
        storage_format=main.STORAGE_FORMAT_IMAGES,
        frames=len(dataset),
        fps=37,
        error="network",
    )
    calls = []

    monkeypatch.setattr(
        main,
        "_remote_dataset_state",
        lambda identity: main.RemoteDatasetState("rev", frozenset(), 0),
    )

    def upload(identity, episode, package, shard, **kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(main, "_upload_live_episode_package", upload)

    assert main.drain_live_upload_queue("BreakoutNoFrameskip-v4")
    assert calls[0]["fps"] == 37


def test_failed_live_episode_prints_retry_command_from_recording_identity(monkeypatch):
    messages = []
    manager = argparse.Namespace(
        identity=main.RecordingIdentity(env_id="BreakoutNoFrameskip-v4"),
        upload_episode=lambda _package, _dataset: False,
    )
    recorder = object.__new__(main.DatasetRecorderWrapper)
    recorder.live_upload_manager = manager
    recorder._recording_rows = [{}]
    recorder._live_episode = main.LiveEpisodePackage("episode", "/tmp/package")
    recorder._build_recorded_dataset = lambda: object()
    recorder._clear_recording_buffers = lambda: None
    monkeypatch.setattr(main.console, "print", lambda message: messages.append(message))

    recorder._finish_live_episode()

    assert "gymrec upload BreakoutNoFrameskip-v4" in messages[0]
    assert recorder._live_episode is None


def test_upload_uses_remote_episode_ids_when_local_marker_is_stale(temp_storage_dir, monkeypatch):
    episode_uuid = uuid.uuid4()
    dataset = make_minimal_dataset(episode_uuid=episode_uuid)
    calls = []

    def fake_upload(env_id, shard, **kwargs):
        calls.append((env_id, list(shard["episode_id"]), kwargs))
        return True

    monkeypatch.setattr(main, "ensure_hf_login", lambda: True)
    monkeypatch.setattr(main, "drain_live_upload_queue", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        main,
        "_remote_dataset_state",
        lambda env_id: main.RemoteDatasetState("abc123", frozenset(), 0),
    )
    monkeypatch.setattr(main, "_upload_dataset_shard_to_hub", fake_upload)
    dataset.save_to_disk(main.get_local_dataset_path("BreakoutNoFrameskip-v4"))
    main._save_uploaded_episode_ids("BreakoutNoFrameskip-v4", {episode_uuid.hex})

    assert main.upload_local_dataset("BreakoutNoFrameskip-v4") is True
    assert len(calls) == 1
    assert calls[0][2]["episode_ids"] == {str(episode_uuid)}


def test_first_upload_adds_replay_card_and_data_in_one_commit(tmp_path, monkeypatch):
    dataset = make_minimal_dataset()
    api = configure_mock_dataset_upload(monkeypatch, remote_files=[])

    def materialize(dataset_arg, local_root, output_path, fps):
        with open(output_path, "wb") as file_obj:
            file_obj.write(b"validated replay")
        return output_path

    monkeypatch.setattr(main, "_materialize_dataset_replay", materialize)

    assert main._upload_dataset_shard_to_hub(
        "BreakoutNoFrameskip-v4",
        dataset,
        storage_format=main.STORAGE_FORMAT_IMAGES,
        local_root=str(tmp_path),
        episode_ids={dataset[0]["episode_id"]},
        remote_state=main.RemoteDatasetState("rev", frozenset(), 0),
        preview_fps=60,
        max_retries=1,
    )

    assert len(api.commits) == 1
    commit = api.commits[0]
    paths = set(commit["contents"])
    assert main.DATASET_REPLAY_FILENAME in paths
    assert "README.md" in paths
    assert any(path.startswith("data/") and path.endswith(".parquet") for path in paths)
    assert set(api.preuploaded) == paths
    card = commit["contents"]["README.md"].decode()
    assert '<video controls loop muted playsinline preload="metadata"' in card
    assert "/datasets/owner/repo/resolve/main/replay.mp4" in card
    assert "pipeline_tag" not in card
    assert "| Total frames | 2 |" in card
    assert "| Trajectories | 1 |" in card


def test_append_preserves_existing_root_replay(tmp_path, monkeypatch):
    dataset = make_minimal_dataset()
    remote_files = [
        main.DATASET_REPLAY_FILENAME,
        "README.md",
        "data/train-00000-of-00001.parquet",
    ]
    api = configure_mock_dataset_upload(monkeypatch, remote_files=remote_files)
    monkeypatch.setattr(
        main,
        "_materialize_dataset_replay",
        lambda *args, **kwargs: pytest.fail("append must preserve the published replay"),
    )

    assert main._upload_dataset_shard_to_hub(
        "BreakoutNoFrameskip-v4",
        dataset,
        storage_format=main.STORAGE_FORMAT_IMAGES,
        local_root=str(tmp_path),
        episode_ids={dataset[0]["episode_id"]},
        remote_state=main.RemoteDatasetState(
            "rev", frozenset({str(uuid.uuid4()), str(uuid.uuid4())}), 10
        ),
        preview_fps=60,
        max_retries=1,
    )

    commit = api.commits[0]
    assert main.DATASET_REPLAY_FILENAME not in commit["contents"]
    card = commit["contents"]["README.md"].decode()
    assert "| Total frames | 12 |" in card
    assert "| Trajectories | 3 |" in card


def test_replace_regenerates_replay_and_uses_replacement_counts(tmp_path, monkeypatch):
    dataset = make_minimal_dataset()
    remote_files = [
        main.DATASET_REPLAY_FILENAME,
        "README.md",
        "data/train-00000-of-00001.parquet",
    ]
    api = configure_mock_dataset_upload(monkeypatch, remote_files=remote_files)

    def materialize(dataset_arg, local_root, output_path, fps):
        with open(output_path, "wb") as file_obj:
            file_obj.write(b"replacement replay")
        return output_path

    monkeypatch.setattr(main, "_materialize_dataset_replay", materialize)

    assert main._upload_dataset_shard_to_hub(
        "BreakoutNoFrameskip-v4",
        dataset,
        storage_format=main.STORAGE_FORMAT_IMAGES,
        local_root=str(tmp_path),
        episode_ids={dataset[0]["episode_id"]},
        replace=True,
        remote_state=main.RemoteDatasetState(
            "rev", frozenset({str(uuid.uuid4()), str(uuid.uuid4())}), 10
        ),
        preview_fps=60,
        max_retries=1,
    )

    commit = api.commits[0]
    replay_operations = [
        operation
        for operation in commit["operations"]
        if operation.path_in_repo == main.DATASET_REPLAY_FILENAME
    ]
    assert {operation.__class__.__name__ for operation in replay_operations} == {
        "CommitOperationAdd",
        "CommitOperationDelete",
    }
    card = commit["contents"]["README.md"].decode()
    assert "| Total frames | 2 |" in card
    assert "| Trajectories | 1 |" in card
    assert "| Total frames | 12 |" not in card


def test_replay_failure_happens_before_remote_commit(tmp_path, monkeypatch):
    dataset = make_minimal_dataset()
    api = configure_mock_dataset_upload(monkeypatch, remote_files=[])
    repo_state_calls = []

    def repo_state(api_arg, repo_id, create=False):
        repo_state_calls.append(create)
        return False, None, []

    monkeypatch.setattr(main, "_hf_repo_state", repo_state)
    monkeypatch.setattr(
        main,
        "_materialize_dataset_replay",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid replay")),
    )

    assert not main._upload_dataset_shard_to_hub(
        "BreakoutNoFrameskip-v4",
        dataset,
        storage_format=main.STORAGE_FORMAT_IMAGES,
        local_root=str(tmp_path),
        episode_ids={dataset[0]["episode_id"]},
        remote_state=main.RemoteDatasetState("rev", frozenset(), 0),
        preview_fps=60,
        max_retries=1,
    )
    assert api.commits == []
    assert repo_state_calls == [False]


def test_no_new_rows_repairs_missing_publication_artifacts(temp_storage_dir, monkeypatch):
    dataset = make_minimal_dataset()
    episode_id = dataset[0]["episode_id"]
    dataset.save_to_disk(main.get_local_dataset_path("BreakoutNoFrameskip-v4"))
    captured = {}

    monkeypatch.setattr(main, "ensure_hf_login", lambda: True)
    monkeypatch.setattr(main, "drain_live_upload_queue", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        main,
        "_remote_dataset_state",
        lambda identity: main.RemoteDatasetState("rev", frozenset({episode_id}), len(dataset)),
    )
    monkeypatch.setattr(main, "HfApi", lambda: object())
    monkeypatch.setattr(
        main,
        "_hf_repo_state",
        lambda *args, **kwargs: (
            True,
            "rev",
            ["README.md", "data/train-00000-of-00001.parquet"],
        ),
    )

    def upload(identity, shard, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(main, "_upload_dataset_shard_to_hub", upload)

    assert main.upload_local_dataset("BreakoutNoFrameskip-v4")
    assert captured["publish_data"] is False
    assert captured["episode_ids"] == set()
    assert captured["remote_state"].frames == len(dataset)


def test_remote_dataset_state_counts_rows_and_episodes_at_pinned_revision(monkeypatch):
    first_episode = uuid.uuid4()
    second_episode = uuid.uuid4()
    rows = [
        {"episode_id": first_episode.bytes},
        {"episode_id": first_episode.bytes},
        {"episode_id": second_episode.bytes},
    ]
    load_calls = []

    monkeypatch.setattr(
        main,
        "_hf_repo_state",
        lambda api, repo_id, create=False: (
            True,
            "pinned-sha",
            ["data/train-00000-of-00001.parquet"],
        ),
    )

    def fake_load_dataset(repo_id, **kwargs):
        load_calls.append((repo_id, kwargs))
        return rows

    monkeypatch.setattr(main, "load_dataset", fake_load_dataset)

    state = main._remote_dataset_state("BreakoutNoFrameskip-v4")

    assert state == main.RemoteDatasetState(
        "pinned-sha",
        frozenset({str(first_episode), str(second_episode)}),
        3,
    )
    assert load_calls[0][1]["revision"] == "pinned-sha"


def test_dataset_card_uses_row_derived_existing_counts(monkeypatch):
    captured = {}
    monkeypatch.setattr(main, "load_local_metadata", lambda identity: {})
    monkeypatch.setattr(main, "_collector_contract_summaries", lambda *args, **kwargs: [])

    def fake_render(env_id, repo_id, **kwargs):
        captured.update(kwargs)
        return "card"

    monkeypatch.setattr(main, "render_dataset_card_content", fake_render)

    assert (
        main._build_dataset_card_content(
            "BreakoutNoFrameskip-v4",
            "BreakoutNoFrameskip-v4",
            "owner/repo",
            new_frames=4,
            new_episodes=1,
            existing_frames=10,
            existing_episodes=2,
            dataset=make_minimal_dataset(collector="row-collector"),
        )
        == "card"
    )
    assert captured["frames"] == 14
    assert captured["episodes"] == 3
    assert captured["collectors"] == ["row-collector"]
    assert captured["gymrec_versions"] == ["0.1.0+test"]
    assert captured["metadata"]["provider_id"] == "stable-retro-turbo"
    assert captured["metadata"]["environment_contract_id"] == TEST_ENVIRONMENT_CONTRACT_ID






def test_dataset_card_template_rejects_missing_context():
    with pytest.raises(UndefinedError):
        main._render_dataset_card_template({})


def test_upload_local_dataset_fails_without_local_dataset_or_pending_queue(
    temp_storage_dir, monkeypatch
):
    monkeypatch.setattr(main, "ensure_hf_login", lambda: True)
    assert main.upload_local_dataset("BreakoutNoFrameskip-v4") is False


def test_save_dataset_metadata_recordings_count_only_new_batch(temp_storage_dir):
    main.save_dataset_locally(
        make_minimal_dataset(collector="random"),
        "BreakoutNoFrameskip-v4",
        artifact_root=str(temp_storage_dir),
        metadata={"env_id": "BreakoutNoFrameskip-v4"},
    )
    main.save_dataset_locally(
        make_minimal_dataset(collector="human"),
        "BreakoutNoFrameskip-v4",
        artifact_root=str(temp_storage_dir),
        metadata={"env_id": "BreakoutNoFrameskip-v4"},
    )

    with open(main._get_metadata_path("BreakoutNoFrameskip-v4")) as f:
        metadata = json.load(f)

    assert [entry["episodes"] for entry in metadata["recordings"]] == [1, 1]
    assert [entry["frames"] for entry in metadata["recordings"]] == [2, 2]
    assert metadata["recordings"][1] == {
        "timestamp": metadata["recordings"][1]["timestamp"],
        "episodes": 1,
        "frames": 2,
        "storage_format": main.STORAGE_FORMAT_IMAGES,
    }


def test_recover_live_image_journal_adds_collector_terminal_row(tmp_path):
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
            "step_index": 0,
            "seed": 123,
            "action": 0,
            "policy_action": None,
            "reward": 1.0,
            "termination": False,
            "truncation": False,
            "info": "{}",
            "session_id": session_id,
            "dataset_format_version": main.DATASET_FORMAT_VERSION,
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
    assert dataset["actions"] == [0, None]
    assert dataset["step_index"] == [0, 1]
    assert dataset["truncations"] == [False, None]
    assert dataset["collector_terminated"] == [False, True]


def test_recover_live_video_journal_appends_collector_terminal_candidate(tmp_path):
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
    main.PILImage.fromarray(terminal_frame).save(terminal_path, format="WEBP", lossless=True)
    row = main._canonical_dataset_row(
        episode_id=episode_id,
        step_index=0,
        seed=123,
        actions=0,
        policy_actions=None,
        rewards=1.0,
        terminations=False,
        truncations=False,
        infos="{}",
        session_id=session_id,
        dataset_format_version=main.DATASET_FORMAT_VERSION,
        collector="random",
        gymrec_version="0.1.0+test",
        storage_format=main.STORAGE_FORMAT_LOSSLESS_VIDEO,
    )
    row.update(
        video_path=video_relpath,
        frame_sha256=main._sha256_rgb(step_frame),
        frame_width=2,
        frame_height=2,
    )
    records = [
        {
            "row_index": 0,
            "row": row,
            "fps": 30,
            "terminal_candidate_path": "terminal_candidate.webp",
            "terminal_candidate_sha256": main._sha256_rgb(terminal_frame),
            "terminal_candidate_width": 2,
            "terminal_candidate_height": 2,
        }
    ]

    dataset = main._dataset_from_recovered_live_records(str(tmp_path), episode_id, records)

    assert len(dataset) == 2
    assert dataset["step_index"] == [0, 1]
    assert dataset["truncations"] == [False, None]
    assert dataset["collector_terminated"] == [False, True]
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
