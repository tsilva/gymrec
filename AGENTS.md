# gymrec Project Guide

This file provides repository-specific guidance for coding agents.

## Project Overview

gymrec records and replays the final single-environment Gymnasium contract exposed by exactly two native runtime packages: `stable-retro-turbo` and `supermariobrosnes-turbo`. Gymrec owns both internal provider adapters, single-lane adaptation, policy and human action conversion, ROM/state provenance, and the recording contract. The runtime packages know nothing about Gymrec and retain their native Gymnasium preprocessing, reward, termination, and info behavior.

## Development Setup

```bash
uv sync
cp .env.example .env
```

Add `HF_TOKEN` only for token-based Hub uploads. Dependencies are defined in `pyproject.toml` and include Python 3.14, Gymnasium, pygame, datasets, and the two allowlisted turbo providers. Do not add ALE, VizDoom, or generic environment constructors.

The native packages are ordinary dependencies and must not register Gymrec entry points or ship Gymrec-specific modules. Keep both packages exempt from the global seven-day `exclude-newer` policy so their current wheels remain resolvable.

## Commands

CLI help (`gymrec --help` and `gymrec <command> --help`) is the authoritative option reference. README.md provides curated recipes. For repository development, run commands as `uv run gymrec ...`.

## Architecture

### Provider Sessions

- `provider_contract.py` validates the generic environment envelope and resolves only Gymrec's two internal providers.
- `providers.py` owns the static registry, shared single-lane session behavior, native factories, action adaptation, named-control conversion, FPS, and asset/version provenance.
- Runtime-specific branching is allowed only inside the internal providers where native constructors, catalogs, action spaces, or asset discovery differ.
- Gymrec may consume legacy `config.task.action` only for mechanical policy-action conversion. It must warn and ignore all other task sections and must never shape native rewards or add success, failure, stalling, or episode-limit semantics.

### Input Sources

- `HumanInputSource` maps pygame events to named controls and asks the provider session for a native action.
- `RandomPolicy` is seeded and samples the provider environment's action space.
- External policies use a strict evaluation environment envelope and Gymrec-owned observation/action adaptation.
- Do not add built-in game-specific policies.

### DatasetRecorderWrapper

- Records exactly the action passed to `env.step()` and exactly the reward, termination, truncation, and info returned by the provider.
- Records observations as lossless WebP image rows or per-episode lossless RGB video artifacts.
- Uses provider callbacks to obtain policy observations, recording frames, and native actions.
- Supports headless non-human collection and pygame rendering for human collection/playback.

### Dataset Contract

- Naming convention: `{username}/gymrec__{encoded_env_id}`, with reversible `_slash_`, `_dash_`, and `_underscore_` encoding.
- Generic fields include `episode_id`, `step_index`, `seed`, `actions`, `policy_actions`, `rewards`, `terminations`, `truncations`, `infos`, `session_id`, `dataset_format_version`, `collector`, `gymrec_version`, `storage_format`, `provider_id`, `env_id`, `environment_contract_id`, `collector_contract_id`, `policy_mode`, `policy_seed`, and `collector_terminated`.
- Image storage adds `observations`; video storage adds `video_path`, `frame_sha256`, `frame_width`, and `frame_height`.
- Immutable environment documents live at `environments/<environment_contract_id>/environment.json`; standardized policy documents live at `collectors/<collector_contract_id>/`.
- The schema is a clean break. Reject legacy datasets, legacy policy recipes, missing provider versions, altered assets, and unsupported providers instead of adding compatibility fallbacks.

## Human Controls

- Space starts recording. Escape exits and preserves recorded steps as a collector-terminated trajectory segment without changing provider termination or truncation values.
- `keymappings.toml` maps keyboard events to named labels.
- The provider advertises a compatible profile and converts active labels to its native action.
- Keep named-control construction shared; isolate unavoidable native action-space differences inside `providers.py`.

## Key Constraints

- Fake-vector and internal-provider contract behavior is covered in `tests/`; native emulator, rendering, Hub, and ffmpeg behavior may still require integration verification.
- Pygame display creation remains lazy after the first observation.
- The main loop is asynchronous and uses `asyncio.sleep()` for pacing.
- Hub uploads require `gymrec login` or `HF_TOKEN`; local recording and `--dry-run` do not.
- Human and seeded random are the only built-in collectors.
- README.md must be kept current with significant changes.

## Configuration Files

- `config.toml` — Recorder display, storage, dataset, and overlay settings.
- `keymappings.toml` — Keyboard keys mapped to named provider control profiles.
