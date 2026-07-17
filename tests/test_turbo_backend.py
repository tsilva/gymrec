import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest

import main
import provider_contract


class OneStepEnv(gym.Env):
    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8)

    def __init__(self):
        self.received_actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((2, 2, 3), dtype=np.uint8), {"seed": seed}

    def step(self, action):
        self.received_actions.append(action)
        return np.ones((2, 2, 3), dtype=np.uint8), 7.5, True, False, {"provider": True}


class ContinuingEnv(OneStepEnv):
    def step(self, action):
        self.received_actions.append(action)
        return np.ones((2, 2, 3), dtype=np.uint8), 7.5, False, False, {"provider": True}


class FakeSession:
    provider_id = "stable-retro-turbo"
    environment_id = "Game-Nes-v0"
    effective_config = {"opaque": {"task": "provider-owned"}}
    provenance = {
        "distribution": "stable-retro-turbo",
        "version": "test",
        "assets": {"rom_sha256": "a" * 64},
    }
    control_profile = "stable_retro.Nes"
    fps = 60

    def __init__(self):
        self.env = OneStepEnv()
        self.adapted = []

    def recording_observation(self, observation):
        return observation

    def policy_observation(self, observation):
        return observation

    def adapt_policy_action(self, action):
        self.adapted.append(action)
        return int(np.asarray(action).reshape(-1)[0])

    def validate_policy(self, policy):
        return None

    def action_from_labels(self, labels):
        return 2 if "RIGHT" in labels else 0


class FakeProvider:
    provider_id = "stable-retro-turbo"
    contract_version = 1

    def __init__(self):
        self.calls = []

    def create(self, *, environment_id, config, render_mode):
        self.calls.append((environment_id, config, render_mode))
        session = FakeSession()
        session.environment_id = environment_id
        session.effective_config = config
        return session

    def catalog(self):
        return ("Game-Nes-v0",)


@dataclass
class FakeEntryPoint:
    name: str
    value: object

    def load(self):
        return self.value


def contract(config=None):
    return provider_contract.EnvironmentContract.parse(
        {
            "contract_version": 1,
            "provider_id": "stable-retro-turbo",
            "environment_id": "Game-Nes-v0",
            "config": config or {},
        }
    )


def artifact_for(session, environment_contract=None):
    environment_contract = environment_contract or contract(session.effective_config)
    artifact_id, document = provider_contract.build_environment_document(
        environment_contract, session
    )
    return main.EnvironmentArtifact(artifact_id, document)


def test_provider_discovery_loads_only_the_two_allowlisted_ids():
    stable = FakeProvider()
    ignored = FakeProvider()
    ignored.provider_id = "third-party-provider"
    discovered = provider_contract.discover_providers(
        entry_points=[
            FakeEntryPoint("stable-retro-turbo", stable),
            FakeEntryPoint("third-party-provider", ignored),
        ]
    )

    assert discovered == {"stable-retro-turbo": stable}
    with pytest.raises(ValueError, match="Unsupported environment provider"):
        provider_contract.load_provider("third-party-provider", entry_points=[])


def test_provider_contract_rejects_legacy_and_extra_fields():
    with pytest.raises(ValueError, match="invalid envelope"):
        provider_contract.EnvironmentContract.parse(
            {"env_provider": "stable-retro-turbo", "game": "Game-Nes-v0"}
        )
    with pytest.raises(ValueError, match="unsupported extra"):
        provider_contract.EnvironmentContract.parse(
            {**contract().as_dict(), "extra": True}
        )


def test_session_creation_forwards_opaque_config_without_interpreting_it():
    provider = FakeProvider()
    config = {"task": {"id": "anything"}, "nested": [1, 2, 3]}
    session = provider_contract.create_session(
        contract(config),
        render_mode="rgb_array",
        entry_points=[FakeEntryPoint("stable-retro-turbo", provider)],
    )

    assert provider.calls == [("Game-Nes-v0", config, "rgb_array")]
    assert session.effective_config == config


def test_environment_document_is_content_addressed_and_materialized(tmp_path):
    session = FakeSession()
    artifact = artifact_for(session)

    main._materialize_environment_artifact(artifact, tmp_path)
    path = tmp_path / "environments" / artifact.contract_id / "environment.json"
    document = json.loads(path.read_text())

    assert document["provider_id"] == "stable-retro-turbo"
    assert document["effective_config"] == session.effective_config
    assert document["provenance"] == session.provenance
    provider_contract.validate_environment_document(document, artifact.contract_id)


def test_environment_document_rejects_schema_drift_and_nonfinite_config():
    session = FakeSession()
    artifact = artifact_for(session)
    altered = {**artifact.document, "legacy_backend": "retro"}
    with pytest.raises(ValueError, match="invalid schema"):
        provider_contract.validate_environment_document(altered)

    with pytest.raises(ValueError, match="JSON"):
        provider_contract.canonical_json_bytes({"bad": float("nan")})


def test_playback_requires_the_exact_recorded_provider_version(monkeypatch):
    artifact = artifact_for(FakeSession())
    monkeypatch.setattr(main, "_installed_package_version", lambda _name: "different")

    with pytest.raises(ValueError, match="requires stable-retro-turbo==test"):
        main._session_from_environment_document(artifact.document, render_mode="rgb_array")


def test_human_input_maps_keys_to_labels_and_provider_owns_action(monkeypatch):
    main._lazy_init()
    session = FakeSession()
    key = next(
        key
        for key, label in main.CONTROL_PROFILES[session.control_profile].items()
        if label == "RIGHT"
    )
    source = main.HumanInputSource(session, main.threading.Lock(), {key})

    assert source.get_action(None) == 2


def test_human_input_rejects_an_unadvertised_control_profile():
    session = FakeSession()
    session.control_profile = "provider.missing"
    with pytest.raises(ValueError, match="No keyboard mapping is installed"):
        main.HumanInputSource(session, main.threading.Lock(), set())


def test_recording_stores_exact_provider_transition(tmp_path):
    main._lazy_init()
    session = FakeSession()
    environment_contract = contract(session.effective_config)
    artifact = artifact_for(session, environment_contract)
    recorder = main.DatasetRecorderWrapper(
        session.env,
        input_source=main.AgentInputSource(lambda _observation: 2),
        headless=True,
        storage_format=main.STORAGE_FORMAT_IMAGES,
        initial_seed=10,
        provider_session=session,
        environment_artifact=artifact,
    )

    dataset = asyncio.run(recorder.record(fps=1000, max_episodes=1))

    assert dataset[0]["actions"] == 2
    assert dataset[0]["rewards"] == 7.5
    assert dataset[0]["terminations"] is True
    assert dataset[0]["truncations"] is False
    assert json.loads(dataset[0]["infos"]) == {"provider": True}
    assert dataset[0]["provider_id"] == "stable-retro-turbo"
    assert dataset[0]["environment_contract_id"] == artifact.contract_id


def test_user_exit_discards_partial_episode_without_fabricating_truncation():
    session = FakeSession()
    session.env = ContinuingEnv()
    artifact = artifact_for(session)
    recorder = main.DatasetRecorderWrapper(
        session.env,
        input_source=main.AgentInputSource(lambda _observation: 2),
        headless=False,
        storage_format=main.STORAGE_FORMAT_IMAGES,
        initial_seed=10,
        provider_session=session,
        environment_artifact=artifact,
    )
    inputs = iter((True, False))
    recorder._input_loop = lambda: next(inputs)
    recorder._ensure_screen = lambda _frame: None
    recorder._render_frame = lambda _frame: None

    dataset = asyncio.run(recorder.record(fps=1000, max_episodes=None))

    assert dataset is None
    assert session.env.received_actions == [2]
    assert recorder._recording_rows == []


def test_random_policy_seed_is_reproducible():
    first = main.RandomPolicy(gym.spaces.Discrete(100))
    second = main.RandomPolicy(gym.spaces.Discrete(100))
    first.reset(seed=123)
    second.reset(seed=123)

    assert [first(None) for _ in range(20)] == [second(None) for _ in range(20)]


def test_removed_environment_specific_surfaces_do_not_return():
    source = open(main.__file__).read().lower()
    banned = (
        "mariorecipetaskenv",
        "mariorightjumppolicy",
        "breakoutcatcherpolicy",
        "nes_simple_action_masks",
        "_create_env__vizdoom",
        "_create_env__alepy",
        "resolve_runtime_backend",
        "import_roms",
        "reindex_games",
    )
    assert [token for token in banned if token in source] == []
    assert set(main.AGENT_POLICY_FACTORIES) == {"random"}


@pytest.mark.parametrize(
    "arguments",
    [
        ["record", "Game-Nes-v0", "--backend", "stable-retro"],
        ["record", "Game-Nes-v0", "--roms-path", "./roms"],
        ["record", "Game-Nes-v0", "--agent", "mario"],
        ["record", "Game-Nes-v0", "--max-steps", "10"],
        ["import_roms", "./roms"],
        ["reindex_games"],
    ],
)
def test_removed_cli_surfaces_are_rejected(arguments):
    result = subprocess.run(
        [sys.executable, main.__file__, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr or "unrecognized arguments" in result.stderr
