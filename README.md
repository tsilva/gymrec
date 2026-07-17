# gymrec

Record and replay gameplay from provider-owned Gymnasium environments as Hugging Face datasets.

gymrec intentionally supports exactly two environment providers:

- `stable-retro-turbo`
- `supermariobrosnes-turbo`

The provider packages own environment construction, ROM and state discovery, preprocessing, action adaptation, reward shaping, success and failure rules, stalling, and episode termination. gymrec records the final action passed to `env.step()`, the provider-selected recording frame, and the exact reward, termination, truncation, and info returned by the provider.

## Install

```bash
uv sync
```

The installed provider releases must register their implementation under the `gymrec.environment_providers` entry-point group. Older releases without provider contract version 1 are rejected.

## Commands

```bash
uv run gymrec list_environments

# Human recording. --env-config is passed opaquely to the provider.
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
    "task": {
      "id": "mario",
      "action": {"set": "simple"},
      "signals": {
        "x": ["xscrollHi", "xscrollLo"],
        "score": "score",
        "lives": "lives",
        "level": ["levelHi", "levelLo"]
      },
      "events": {
        "life_loss": {"signal": "lives", "operation": "decrease"},
        "level_change": {"signal": "level", "operation": "change"}
      },
      "termination": {
        "failure": ["life_loss"],
        "success": ["level_change"],
        "max_episode_steps": 4500
      },
      "reward": {"reward_mode": "score"}
    }
  }
}
```

gymrec validates only the envelope and passes `config` unchanged. It does not infer an evaluation environment from training metadata or interpret the task contents.

## Recording contract

Each row contains generic provenance:

- `provider_id`
- `env_id`
- `environment_contract_id`
- `collector_contract_id` for policy recordings

The immutable environment document is stored at `environments/<environment_contract_id>/environment.json`. It binds the provider version, declared and effective configuration, provider-reported asset provenance, action and observation spaces, control profile, and FPS. Playback requires the exact recorded provider version and asset provenance.

Dataset format version 2 is a clean break. Legacy schemas and legacy policy environment shapes are rejected rather than migrated or aligned.

## Human controls

gymrec owns keyboard event handling and the named control profiles in `keymappings.toml`. A provider advertises a profile and translates the active labels into its native action. Human mode fails clearly if either side does not provide a compatible profile.

## Development

```bash
uv run pytest
uv run ruff check .
```

ROM-dependent provider integration tests require the corresponding provider assets. Core contract tests use fake entry points and environments.
