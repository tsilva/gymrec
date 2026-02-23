<div align="center">
  <img src="logo.png" alt="gymrec" width="512"/>

  # gymrec

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
  [![Hugging Face](https://img.shields.io/badge/HuggingFace-Datasets-yellow.svg)](https://huggingface.co/docs/datasets)

  **🎮 Record and replay gameplay from Gymnasium environments as Hugging Face datasets 📊**

  [Features](#features) · [Quick Start](#quick-start) · [Usage](#usage) · [Supported Environments](#supported-environments)
</div>

---

## 🚀 Features

[![CI](https://github.com/tsilva/gymrec/actions/workflows/release.yml/badge.svg)](https://github.com/tsilva/gymrec/actions/workflows/release.yml)

- **🎯 Multi-platform support** — Works with Atari (ALE-py), Stable-Retro, and VizDoom environments
- **💾 Dataset-first design** — Captures frames and actions directly as Hugging Face datasets
- **🎮 Automatic key bindings** — Platform-specific controls preconfigured for each environment type
- **🔄 Playback verification** — Replay recordings to confirm environment determinism
- **☁️ Hub integration** — Push datasets directly to Hugging Face Hub with auto-generated dataset cards

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/tsilva/gymrec.git
cd gymrec

# Install dependencies
uv sync

# Configure Hugging Face token
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

## 📖 Usage

### 🎬 Record gameplay

```bash
uv run python main.py record BreakoutNoFrameskip-v4
uv run python main.py record VizdoomBasic-v0 --fps 35
uv run python main.py record Airstriker-Genesis --fps 60
```

Press **Space** to start recording. Use **ESC** to stop and exit.

### 🤖 Automated collection (random agent)

```bash
# Collect 100 episodes headlessly at max speed
uv run gymrec record SuperMarioBros-Nes --agent random --headless --episodes 100

# Same but with display (for monitoring)
uv run gymrec record SuperMarioBros-Nes --agent random --episodes 10

# Collect 100 episodes across 5 parallel worker processes
uv run gymrec record BreakoutNoFrameskip-v4 --agent random --headless --episodes 100 --workers 5
```

`--headless` skips rendering for maximum collection speed. `--episodes` defaults to 1 if omitted. `--workers` spawns multiple parallel processes to distribute episode collection — requires `--agent` (not human mode).

### 🔄 Replay a dataset

```bash
uv run python main.py playback BreakoutNoFrameskip-v4
```

Replays the recorded actions from your Hugging Face Hub dataset.

### 📋 List available environments

```bash
uv run python main.py list_environments
```

Shows all available Atari, Stable-Retro, and VizDoom environments.

## 🎯 Supported Environments

| Platform | Examples | Default FPS |
|----------|----------|-------------|
| 🕹️ Atari (ALE-py) | `BreakoutNoFrameskip-v4`, `PongNoFrameskip-v4` | 90 |
| 🔫 VizDoom | `VizdoomBasic-v0`, `VizdoomCorridor-v0` | 45 |
| 🎲 Stable-Retro | `Airstriker-Genesis`, `SuperMarioBros-Nes` | 90 |

### 🎮 Controls

| Platform | Controls |
|----------|----------|
| 🕹️ Atari | Arrow keys for movement |
| 🔫 VizDoom | Arrows (move/turn), Ctrl (attack), Space (use), 1-7 (weapons) |
| 🎲 Stable-Retro | Arrows, Z/X (A/B buttons), Tab/Enter (Select/Start) |

## 📋 Requirements

- 🐍 Python 3.12+
- ⚡ [uv](https://docs.astral.sh/uv/getting-started/installation/) (for dependency management)
- 🤗 Hugging Face account and token (for dataset uploads)

### 🍎 macOS Apple Silicon Note

The `stable-retro` package on PyPI ships a broken wheel for Apple Silicon (x86_64 binary mislabeled as arm64). This repo includes a pre-built native ARM64 wheel in `wheels/` that is installed automatically by `uv sync` — no extra steps needed.

<details>
<summary>🔧 Rebuilding the wheel from source (only needed for new Python versions or stable-retro updates)</summary>

```bash
# 1. Clone with all submodules
git clone --recursive https://github.com/Farama-Foundation/stable-retro.git /tmp/stable-retro-build
cd /tmp/stable-retro-build

# 2. Disable the pce core (broken zlib on macOS)
#    In CMakeLists.txt, change:
#      add_core(pce mednafen_pce_fast)
#    To:
#      if(NOT APPLE)
#        add_core(pce mednafen_pce_fast)
#      endif()

# 3. Configure with Apple Clang (NOT Homebrew gcc)
cmake . -G "Unix Makefiles" \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DPYEXT_SUFFIX=.cpython-312-darwin.so

# 4. Build
make -j8 stable_retro

# 5. Create wheel
pip wheel . --no-build-isolation -w /path/to/gymrec/wheels/

# 6. Update pyproject.toml [tool.uv.sources] with the new wheel filename
```

**Why Clang?** Homebrew's GCC passes `--exclude-libs` to the linker, which Apple's `ld` doesn't support. Using `/usr/bin/cc` (Apple Clang) avoids this.

**Why skip pce?** The PC Engine core bundles an old zlib that redefines `fdopen` as `NULL`, which conflicts with the macOS SDK headers.

</details>

## 🔧 How It Works

1. **🎬 Recording** — The `DatasetRecorderWrapper` captures each frame as a JPEG and logs the corresponding action
2. **💾 Storage** — Frames and actions are assembled into a Hugging Face Dataset with columns: `episode_id`, `timestamp`, `image`, `step`, `action`
3. **☁️ Upload** — Datasets are pushed to Hub with naming convention `{username}/GymnasiumRecording__{env_id}`
4. **🔄 Playback** — Recorded actions are fed back to the environment to verify deterministic replay

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
