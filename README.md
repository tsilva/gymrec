# gymrec

Record and replay gameplay from native Gymnasium environments as Hugging Face datasets.

gymrec intentionally supports exactly two environment providers:

- `stable-retro-turbo`
- `supermariobrosnes-turbo`

Gymrec owns both provider adapters. The runtime packages expose only their normal
Gymnasium APIs and retain their native preprocessing, reward, termination, and
info behavior. Gymrec creates one native vector lane, adapts it to the recorder,
converts policy or human controls, and records the exact action, reward,
termination, truncation, and info returned by that environment.

## Install

Requires Python 3.14.

```bash
uv sync
```

The two runtime packages are ordinary dependencies. They do not register or
implement anything specifically for Gymrec.

## Commands

```bash
uv run gymrec list_environments

# Human recording. --env-config configures the selected native runtime.
uv run gymrec record SuperMarioBros-Nes-v0 \
  --provider supermariobrosnes-turbo \
  --env-config '{"state":"Level1-1"}' \
  --dry-run

# Reproducible generic collection.
uv run gymrec record SuperMarioBros-Nes-v0 \
  --provider stable-retro-turbo \
  --env-config '{"state":"Level1-1"}' \
  --agent random --headless --episodes 10 --seed 10000 --dry-run

# A policy bundle selects its own provider and complete evaluation contract.
uv run gymrec record hf://owner/model-repo --episodes 10 --headless --dry-run

uv run gymrec playback SuperMarioBros-Nes-v0
uv run gymrec playback SuperMarioBros-Nes-v0 --verify
uv run gymrec video SuperMarioBros-Nes-v0 --range 3-7
uv run gymrec upload SuperMarioBros-Nes-v0
uv run gymrec upload SuperMarioBros-Nes-v0 --replace
uv run gymrec minari-export SuperMarioBros-Nes-v0
```

Run `gymrec <command> --help` for the complete option surface. ROM importing and environment indexing belong to the provider packages; gymrec has no `--roms-path`, `import_roms`, `reindex_games`, or backend override.

## Interactive selection

Run `gymrec` or omit the environment from `gymrec record` to open the searchable
environment selector. Omitting the recording from `gymrec playback` or
`gymrec video` opens the same selector for locally available and Hugging Face
recordings.

- Start typing to fuzzy-search environment names, providers, or recording refs.
- Press `/` to return focus to search.
- Use Arrow keys, Page Up/Down, Home/End, or Tab to navigate.
- Press Enter or click a result to select it; press Escape to cancel.

Provider badges remain part of the selection identity. When multiple providers
expose the same environment ID, such as `SuperMarioBros-Nes-v0`, they remain
separate choices. Passing an explicit environment and `--provider` bypasses the
selector entirely.

Limited terminals use a compact searchable text menu that shows at most 25
matches. Set `GYMREC_TEXT_MENU=1` to force this fallback.

## Policy environment contract

Policy recording requires an explicit `recipe.eval.environment` object:

```json
{
  "contract_version": 1,
  "provider_id": "supermariobrosnes-turbo",
  "environment_id": "SuperMarioBros-Nes-v0",
  "config": {
    "state": "Level1-1",
    "frame_skip": 4,
    "obs_resize": [84, 84],
    "task": {"action": {"set": "simple"}}
  }
}
```

Gymrec validates the envelope and forwards native runtime options from `config`.
For existing recipes it consumes only `config.task.action` as a mechanical
policy-to-native action mapping. Named `simple` and `right` sets and explicit
ordered control combinations are supported. Other `task` sections are ignored
with a warning and are omitted from the effective configuration: Gymrec never
shapes reward or adds success, failure, stalling, or episode-limit semantics.

## Recording contract

Each row contains generic provenance:

- `provider_id`
- `env_id`
- `environment_contract_id`
- `collector_contract_id` for policy recordings

The immutable environment document is stored at `environments/<environment_contract_id>/environment.json`. It binds the runtime package version, declared and effective configuration, Gymrec-computed asset provenance, action and observation spaces, control profile, and FPS. Playback requires the exact recorded runtime version and asset provenance.

Dataset format version 2 is a clean break. Legacy schemas and legacy policy environment shapes are rejected rather than migrated or aligned.

## Human controls

gymrec owns keyboard event handling and the named control profiles in `keymappings.toml`. Its internal provider advertises a profile and translates the active labels into the runtime's native action. Human mode fails clearly if either side does not provide a compatible profile.

## Development

```bash
uv run pytest
uv run ruff check .
```

ROM-dependent provider integration tests require the corresponding runtime assets.
Core contract tests use fake vector environments and the internal provider registry.
