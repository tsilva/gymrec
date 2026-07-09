import argparse
import asyncio
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


def test_env_id_encoding_round_trips_special_characters():
    env_id = "ALE/Breakout-v5_custom"
    encoded = main._encode_env_id_for_hf(env_id)

    assert encoded == "ALE_slash_Breakout_dash_v5_underscore_custom"
    assert main._decode_hf_repo_name(encoded) == env_id


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
