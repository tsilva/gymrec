import asyncio
import copy
import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

import main
import provider_contract
import providers


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


class FakeVectorEnv:
    num_envs = 1
    single_action_space = gym.spaces.MultiBinary(3)
    single_observation_space = gym.spaces.Box(
        0, 255, shape=(2, 2, 3), dtype=np.uint8
    )

    def __init__(self):
        self.received_actions = []

    def reset(self, *, seed=None, options=None):
        return np.zeros((1, 2, 2, 3), dtype=np.uint8), {
            "seed": np.asarray([seed]),
            "hidden": np.asarray([1]),
            "_hidden": np.asarray([False]),
        }

    def step(self, actions):
        self.received_actions.append(actions.copy())
        return (
            np.ones((1, 2, 2, 3), dtype=np.uint8),
            np.asarray([7.5], dtype=np.float32),
            np.asarray([True]),
            np.asarray([False]),
            {"native": np.asarray([3]), "_native": np.asarray([True])},
        )

    def render(self):
        return np.full((2, 2, 3), 9, dtype=np.uint8)

    def close(self):
        return None


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


def test_provider_discovery_uses_only_gymrecs_two_internal_adapters():
    discovered = provider_contract.discover_providers()

    assert set(discovered) == {
        "stable-retro-turbo",
        "supermariobrosnes-turbo",
    }
    with pytest.raises(ValueError, match="Unsupported environment provider"):
        provider_contract.load_provider("third-party-provider")


def test_provider_catalog_reports_progress_around_each_provider(monkeypatch):
    class CatalogProvider:
        def __init__(self, *environment_ids):
            self.environment_ids = environment_ids

        def catalog(self):
            return self.environment_ids

    class FakeProgress:
        def __init__(self):
            self.descriptions = []
            self.advances = []

        def update(self, task_id, *, description, refresh):
            self.descriptions.append((task_id, description, refresh))

        def advance(self, task_id):
            self.advances.append(task_id)

    monkeypatch.setattr(
        main,
        "discover_providers",
        lambda: {
            "provider-a": CatalogProvider("GameA-v0", "GameB-v0"),
            "provider-b": CatalogProvider("GameC-v0"),
        },
    )
    monkeypatch.setattr(
        main, "SUPPORTED_PROVIDER_IDS", frozenset({"provider-a", "provider-b"})
    )
    progress = FakeProgress()

    rows = main._provider_catalog(progress=progress, task_id=7)

    assert rows == [
        ("provider-a", "GameA-v0"),
        ("provider-a", "GameB-v0"),
        ("provider-b", "GameC-v0"),
    ]
    assert [description for _, description, _ in progress.descriptions] == [
        "[bold]Scanning provider-a environments[/]",
        "[bold]Scanning provider-b environments[/]",
    ]
    assert progress.advances == [7, 7]


def test_provider_catalog_progress_starts_before_lazy_initialization(monkeypatch):
    events = []

    class FakeProgress:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args):
            events.append("exit")

        def add_task(self, description, *, total):
            events.append(("add", description, total))
            return 3

        def refresh(self):
            events.append("refresh")

        def update(self, task_id, **kwargs):
            events.append(("update", task_id, kwargs))

    progress = FakeProgress()
    monkeypatch.setattr(main, "_episode_progress", lambda transient: progress)
    monkeypatch.setattr(main, "_lazy_init", lambda: events.append("initialize"))

    def fake_catalog(*, progress, task_id):
        events.append(("catalog", progress, task_id))
        return [("provider-a", "GameA-v0")]

    monkeypatch.setattr(main, "_provider_catalog", fake_catalog)

    assert main._load_provider_catalog_with_progress() == [
        ("provider-a", "GameA-v0")
    ]
    assert events[:4] == [
        "enter",
        ("add", "[bold]Initializing Gymrec[/]", 3),
        "refresh",
        "initialize",
    ]
    assert events[5] == ("catalog", progress, 3)


def test_environment_choices_keep_duplicate_ids_provider_qualified():
    choices = main._environment_selection_choices(
        [
            ("stable-retro-turbo", "SuperMarioBros-Nes-v0"),
            ("supermariobrosnes-turbo", "SuperMarioBros-Nes-v0"),
        ]
    )

    assert [choice.key for choice in choices] == ["environment:0", "environment:1"]
    assert [choice.category for choice in choices] == [
        "stable-retro-turbo",
        "SuperMarioBros-Nes-turbo",
    ]
    assert [choice.exact_value for choice in choices] == [
        "stable-retro-turbo:SuperMarioBros-Nes-v0",
        "supermariobrosnes-turbo:SuperMarioBros-Nes-v0",
    ]
    assert [choice.value for choice in choices] == [
        ("stable-retro-turbo", "SuperMarioBros-Nes-v0"),
        ("supermariobrosnes-turbo", "SuperMarioBros-Nes-v0"),
    ]


def test_recording_choices_merge_local_and_hub_origins():
    choices = main._recording_selection_choices(
        ["SuperMarioBros-Nes-v0", "hf://owner/policy-data"],
        ["SuperMarioBros-Nes-v0", "hf://owner/remote-only"],
    )

    by_label = {choice.label: choice for choice in choices}
    assert by_label["SuperMarioBros-Nes-v0"].category == "Local + Hub"
    assert by_label["hf://owner/policy-data"].category == "Local"
    assert by_label["hf://owner/remote-only"].category == "Hub"


def test_text_fallback_requires_provider_for_ambiguous_exact_id(monkeypatch):
    choices = main._environment_selection_choices(
        [
            ("stable-retro-turbo", "SuperMarioBros-Nes-v0"),
            ("supermariobrosnes-turbo", "SuperMarioBros-Nes-v0"),
        ]
    )
    answers = iter(["SuperMarioBros-Nes-v0", "2"])
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(main.Prompt, "ask", lambda *_args, **_kwargs: next(answers))

    selected = main._select_choice_text_fallback(
        choices,
        title="Select an environment",
        argument_name="the environment ID and --provider",
    )

    assert selected == ("supermariobrosnes-turbo", "SuperMarioBros-Nes-v0")


def test_text_fallback_accepts_provider_qualified_id(monkeypatch):
    choices = main._environment_selection_choices(
        [("stable-retro-turbo", "SonicTheHedgehog-Genesis-v0")]
    )
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        main.Prompt,
        "ask",
        lambda *_args, **_kwargs: "stable-retro-turbo:SonicTheHedgehog-Genesis-v0",
    )

    selected = main._select_choice_text_fallback(
        choices,
        title="Select an environment",
        argument_name="the environment ID and --provider",
    )

    assert selected == ("stable-retro-turbo", "SonicTheHedgehog-Genesis-v0")


def test_noninteractive_selector_requires_explicit_id(monkeypatch):
    choices = main._environment_selection_choices(
        [("stable-retro-turbo", "SonicTheHedgehog-Genesis-v0")]
    )
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: False)

    with pytest.raises(ValueError, match="pass the environment ID and --provider explicitly"):
        main._select_choice_text_fallback(
            choices,
            title="Select an environment",
            argument_name="the environment ID and --provider",
        )


def test_text_fallback_turns_eof_into_an_explicit_id_error(monkeypatch):
    choices = main._environment_selection_choices(
        [("stable-retro-turbo", "SonicTheHedgehog-Genesis-v0")]
    )
    monkeypatch.setattr(main.sys.stdin, "isatty", lambda: True)

    def end_input(*_args, **_kwargs):
        raise EOFError

    monkeypatch.setattr(main.Prompt, "ask", end_input)

    with pytest.raises(ValueError, match="selection ended.*pass the environment ID"):
        main._select_choice_text_fallback(
            choices,
            title="Select an environment",
            argument_name="the environment ID and --provider",
        )


def test_text_menu_environment_variable_forces_fallback(monkeypatch):
    monkeypatch.setenv("GYMREC_TEXT_MENU", "1")
    monkeypatch.setenv("TERM", "xterm-256color")

    assert not main._terminal_tui_supported()


def test_async_selector_maps_opaque_tui_key_to_domain_value(monkeypatch):
    import gymrec_tui

    choices = main._environment_selection_choices(
        [("stable-retro-turbo", "SonicTheHedgehog-Genesis-v0")]
    )
    monkeypatch.setattr(main, "_terminal_tui_supported", lambda: True)

    async def choose(items, **_kwargs):
        assert items == choices
        return "environment:0"

    monkeypatch.setattr(gymrec_tui, "select_item", choose)

    selected = asyncio.run(
        main._select_choice(
            choices,
            title="Select an environment",
            placeholder="Search",
            argument_name="the environment ID and --provider",
        )
    )

    assert selected == ("stable-retro-turbo", "SonicTheHedgehog-Genesis-v0")


def test_main_returns_normally_when_interactive_selector_is_cancelled(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["gymrec"])

    async def cancel(_available_recordings_only=False):
        return None

    monkeypatch.setattr(main, "select_environment_interactive", cancel)
    monkeypatch.setattr(
        main,
        "_lazy_init",
        lambda: pytest.fail("cancellation must not continue initialization"),
    )

    asyncio.run(main.main())


def test_provider_contract_rejects_legacy_and_extra_fields():
    with pytest.raises(ValueError, match="invalid envelope"):
        provider_contract.EnvironmentContract.parse(
            {"env_provider": "stable-retro-turbo", "game": "Game-Nes-v0"}
        )
    with pytest.raises(ValueError, match="unsupported extra"):
        provider_contract.EnvironmentContract.parse(
            {**contract().as_dict(), "extra": True}
        )


def test_shared_single_lane_adapter_preserves_native_transition_and_reset_contract():
    vector_env = FakeVectorEnv()
    env = providers.SingleLaneEnv(
        vector_env, system="Nes", buttons=("B", "A", "RIGHT")
    )

    observation, info = env.reset(seed=7)
    assert observation.shape == (2, 2, 3)
    assert info == {"seed": 7}

    action = np.asarray([0, 0, 1], dtype=np.int8)
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation.tolist() == np.ones((2, 2, 3), dtype=np.uint8).tolist()
    assert reward == 7.5
    assert terminated is True
    assert truncated is False
    assert info == {"native": 3}
    assert vector_env.received_actions[0].tolist() == [[0, 0, 1]]
    with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
        env.step(action)


def test_legacy_task_uses_only_action_conversion_and_warns_about_semantics():
    config = {
        "state": "Level1-1",
        "task": {
            "id": "mario",
            "action": {"set": "simple"},
            "reward": {"reward_mode": "score"},
            "termination": {"success": ["level_change"]},
        },
    }

    with pytest.warns(UserWarning, match="uses only config.task.action"):
        declared, kwargs, policy_actions, effective = providers._prepare_config(
            config
        )

    assert declared == config
    assert kwargs == {"state": "Level1-1"}
    assert policy_actions == providers.BUILTIN_ACTION_SETS["simple"]
    assert effective == {
        "state": "Level1-1",
        "task": {"action": {"set": "simple"}},
    }


def test_shared_session_adapts_discrete_policy_actions_to_named_native_controls():
    vector_env = FakeVectorEnv()
    session = providers.ProviderSession(
        provider_id="stable-retro-turbo",
        environment_id="Game-Nes-v0",
        declared_config={},
        effective_config={},
        vector_env=vector_env,
        system="Nes",
        buttons=("B", "A", "RIGHT"),
        policy_actions=(("RIGHT",), ("RIGHT", "B")),
        fps=60,
        assets={},
    )

    assert session.adapt_policy_action(0).tolist() == [0, 0, 1]
    assert session.adapt_policy_action(1).tolist() == [1, 0, 1]
    assert session.recording_observation(None).tolist() == np.full(
        (2, 2, 3), 9, dtype=np.uint8
    ).tolist()


def test_session_creation_uses_the_gymrec_owned_provider_registry(monkeypatch):
    provider = FakeProvider()
    config = {"task": {"id": "anything"}, "nested": [1, 2, 3]}
    monkeypatch.setitem(
        provider_contract.PROVIDERS, "stable-retro-turbo", provider
    )
    session = provider_contract.create_session(
        contract(config),
        render_mode="rgb_array",
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


@pytest.mark.parametrize(
    ("field", "change", "message"),
    [
        (
            "effective_config",
            lambda value: {**value, "drift": True},
            "effective config does not match",
        ),
        (
            "provenance",
            lambda value: {
                **value,
                "assets": {**value["assets"], "rom_sha256": "changed"},
            },
            "assets do not match",
        ),
        (
            "action_space",
            lambda value: {**value, "n": value["n"] + 1},
            "action space does not match",
        ),
        (
            "observation_space",
            lambda value: {**value, "shape": [1, 2, 3]},
            "observation space does not match",
        ),
    ],
)
def test_playback_rejects_recorded_environment_drift(
    monkeypatch, field, change, message
):
    artifact = artifact_for(FakeSession())
    document = copy.deepcopy(artifact.document)
    document[field] = change(document[field])
    monkeypatch.setattr(main, "_installed_package_version", lambda _name: "test")
    monkeypatch.setattr(
        main, "create_session", lambda _contract, render_mode: FakeSession()
    )

    with pytest.raises(ValueError, match=message):
        main._session_from_environment_document(document, render_mode="rgb_array")


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
    assert dataset[0]["collector_terminated"] is False
    assert dataset[1]["collector_terminated"] is False
    assert json.loads(dataset[0]["infos"]) == {"provider": True}
    assert dataset[0]["provider_id"] == "stable-retro-turbo"
    assert dataset[0]["environment_contract_id"] == artifact.contract_id


def test_user_exit_preserves_partial_episode_without_fabricating_truncation():
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

    assert session.env.received_actions == [2]
    assert len(dataset) == 2
    assert dataset[0]["actions"] == 2
    assert dataset[0]["terminations"] is False
    assert dataset[0]["truncations"] is False
    assert dataset[0]["collector_terminated"] is False
    assert dataset[1]["actions"] is None
    assert dataset[1]["collector_terminated"] is True


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
