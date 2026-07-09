import argparse

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
        return self._columns[key]


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
