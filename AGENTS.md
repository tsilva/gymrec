# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gymrec is a Python tool for recording and replaying gameplay from Gymnasium environments (Atari, Stable-Retro, VizDoom). It captures frames and actions, stores them as Hugging Face datasets, and can replay recordings to verify environment determinism.

## Development Setup

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

Dependencies are defined in `pyproject.toml` and include Python 3.12+, gymnasium, pygame, datasets, and platform-specific game engines (ale-py, vizdoom, stable-retro-turbo).

### Stable-Retro Runtime

Use the PyPI `stable-retro-turbo` package for Stable-Retro support. It publishes native macOS arm64 wheels and keeps the upstream-compatible import name:

```python
import stable_retro as retro
```

`pyproject.toml` pins `stable-retro-turbo` and exempts only that package from the global `exclude-newer = "7 days"` policy because the Python 3.12 macOS arm64 wheels are from the current PyPI release line. Do not reintroduce the old committed-wheel or `stable-retro-apple-silicon` setup unless explicitly requested.

## Core Commands

### Recording gameplay
```bash
uv run python main.py record <env_id>
uv run python main.py record BreakoutNoFrameskip-v4 --fps 30
```

### Recording with agent (automated data collection)
```bash
# Random agent, headless, collect 100 episodes
uv run python main.py record SuperMarioBros-Nes --agent random --headless --episodes 100

# Random agent with display (for monitoring)
uv run python main.py record SuperMarioBros-Nes --agent random --episodes 10
```

### Replaying recorded datasets
```bash
uv run python main.py playback <env_id>
uv run python main.py playback BreakoutNoFrameskip-v4 --fps 30
```

### Listing available environments
```bash
uv run python main.py list_environments
```

## Architecture

### Single-File Design
All code is in `main.py`. The project prioritizes simplicity over modularization.

### Key Components

**InputSource Abstraction**
- `InputSource`: Abstract base class for input sources
- `HumanInputSource`: Keyboard input via pygame
- `AgentInputSource`: Policy-based input for automated data collection
- `BasePolicy` / `RandomPolicy`: Policy interface with random sampling implementation

**DatasetRecorderWrapper**
- Gymnasium wrapper that handles recording and playback
- Accepts `InputSource` for flexible input (human or agent)
- Supports headless mode for fast automated data collection
- Manages pygame rendering (2x scaled display, skipped in headless mode)
- Records frames to temporary WebP files, then converts to HF Dataset
- Handles three environment types with different action spaces:
  - Atari (ALE-py): Discrete actions
  - VizDoom: MultiBinary actions (or Dict with binary/continuous)
  - Stable-Retro: MultiBinary actions, platform-specific mappings

**Environment Creation**
- `create_env()` detects environment type by pattern matching env_id
- Adds `_env_id`, `_vizdoom`, or `_stable_retro` attributes to distinguish backends
- Each backend has different initialization requirements and action spaces

**Action Mapping System**
- Platform-specific key bindings: `ATARI_KEY_BINDINGS`, `VIZDOOM_KEY_BINDINGS`, `STABLE_RETRO_KEY_BINDINGS`
- VizDoom requires button index mapping
- Stable-Retro has per-console mappings (Nes, Snes, Genesis, etc.)
- Action conversion for dataset storage normalizes numpy/dict/int to serializable format

**Dataset Management**
- Naming convention: `{username}/gymrec__{encoded_env_id}`, where env IDs use the reversible `_slash_`, `_dash_`, and `_underscore_` encoding.
- Concatenates new recordings with existing datasets on Hub
- Auto-generates dataset cards with episode/frame statistics
- Fields: `episode_id`, `seed`, `observations`, `actions`, `rewards`, `terminations`, `truncations`, `infos`, `session_id`, `collector`, `gymrec_version`
- **Provenance columns** (added per-row, constant per session):
  - `session_id` (`binary(16)`): UUID grouping all episodes from one `gymrec record` run
  - `collector` (`string`): Who collected the data (`"human"`, `"random"`, `"mario"`, `"breakout"`, or future agent names)
  - `gymrec_version` (`string`): Version string like `"0.1.0+abc1234"` from `_get_gymrec_version()`
- Backward-compatible concatenation: old datasets missing provenance columns get sentinel values (`"unknown"` / `b'\x00'*16`)

**FPS Handling**
- Attempts to read from environment metadata first
- Falls back to defaults: Atari=90fps, VizDoom=45fps, Retro=90fps (see config.toml)
- Pattern matches env_id when metadata unavailable

## Important Patterns

### Action Space Handling
The wrapper must handle three distinct action space types:
1. **Discrete** (Atari): Single integer action
2. **MultiBinary** (VizDoom, Retro): Array of 0s and 1s
3. **Dict** (Some VizDoom configs): `{"binary": ..., "continuous": ...}`

Action conversion happens in two places:
- Recording: Convert env actions to serializable format
- Playback: Convert stored actions back to env format

### Environment Detection
Use the `_vizdoom` and `_stable_retro` attributes added by `create_env()` to branch behavior:
```python
if hasattr(self.env, '_vizdoom') and self.env._vizdoom:
    # VizDoom-specific logic
elif hasattr(self.env, '_stable_retro') and self.env._stable_retro:
    # Stable-Retro-specific logic
else:
    # Atari/ALE-py logic
```

### Frame Extraction
VizDoom environments may return dict observations. Extract the actual image:
```python
if isinstance(frame, dict):
    for k in ["obs", "image", "screen"]:
        if k in frame:
            frame = frame[k]
            break
```

### User Controls (Human Mode)
- Space: Start recording
- ESC: Exit
- Platform-specific game controls (arrow keys, Z/X buttons, etc.)

### Agent Mode
- `--agent {human,random,mario,breakout}`: Choose input source (default: human)
- `--headless`: Run without display (agent mode only, runs at max speed)
- `--episodes N`: Collect N episodes then stop
  - Human mode: defaults to unlimited (run until ESC)
  - Agent mode: defaults to 1 episode

## Key Constraints

- **No testing infrastructure**: The project has no tests. Changes must be manually verified.
- **Pygame dependency**: All rendering and input uses pygame. The screen is created lazily after first observation.
- **Async design**: Main loop uses `asyncio` with `await asyncio.sleep()` for frame pacing.
- **Environment variables**: Requires `HF_TOKEN` in `.env` for dataset uploads.
- **Headless mode**: Only available with `--agent` flag (not human mode).
- README.md must be kept up to date with any significant project changes using the readme-generator skill.

## Creating Custom Policies

To create a custom policy, subclass `BasePolicy`:

```python
class MyPolicy(BasePolicy):
    def __call__(self, observation):
        # observation is the RGB array from the environment
        # Return an action compatible with the action space
        return self.action_space.sample()  # or your logic here

# Use it in the record command
policy = MyPolicy(env.action_space)
input_source = AgentInputSource(policy, headless=True)
recorder = DatasetRecorderWrapper(env, input_source=input_source, headless=True)
```

Policies receive the full observation (RGB array) and must return actions compatible with the environment's action space:
- **Discrete**: Return an integer (0 to n-1)
- **MultiBinary**: Return a numpy array of 0s and 1s
- **Dict**: Return a dict with appropriate keys

## Configuration Files

- `config.toml` — Override display scale, FPS defaults, dataset settings, overlay. All keys commented out by default.
- `keymappings.toml` — Customize key bindings per platform (Atari, VizDoom, Stable-Retro).

## CLI Shorthand

The `gymrec` entry point is installed in the venv:
```bash
uv run gymrec record BreakoutNoFrameskip-v4
uv run gymrec playback BreakoutNoFrameskip-v4
```
