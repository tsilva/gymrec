<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/gymrec/main/logo.png" alt="gymrec" width="512"/>

  **🎮 Record and replay gameplay from Gymnasium environments as Hugging Face datasets 📊**
</div>

gymrec is a Python CLI for collecting gameplay from Gymnasium environments and saving it as replayable Hugging Face datasets. It supports human keyboard capture, built-in agent policies, local dataset storage, Hub uploads, playback verification, and MP4 exports.

It works across Atari through ALE-py, Stable-Retro console environments, the native `supermariobrosnes-turbo` runtime, and VizDoom. Recordings store observations, actions, rewards, episode metadata, collector provenance, and the gymrec version used to collect the run.

## Install

```bash
git clone https://github.com/tsilva/gymrec.git
cd gymrec
./install.sh
cp .env.example .env
```

`install.sh` uses `uv tool install . -e` from the repo root. On a fresh checkout it installs the `gymrec` command as an editable uv tool; when the tool already exists, rerunning it upgrades dependency versions.

Add your Hugging Face token to `.env` when you want to upload datasets:

```bash
HF_TOKEN=your-api-token
```

Set `ROMS_PATH` in the `.env` file for the directory where you run `gymrec` to make gymrec use that source for games:

```bash
ROMS_PATH=/path/to/roms
```

`ROMS_PATH` is passed to ALE-py as `ALE_ROMS_DIR` for Atari games; if `ROMS_PATH` points at a single file, ALE-py receives its parent directory. For Stable-Retro, gymrec scans and imports matching ROMs from `ROMS_PATH` before listing or launching games. For `supermariobrosnes-turbo`, it may point to the ROM file itself or a directory: gymrec selects the canonical `SuperMarioBros-Nes-v0` ROM by SHA-256 even when several `.nes` files are present. You can also pass it directly on the command line:

The editable installation loads the repository `.env` as its default configuration even when `gymrec` is launched from another directory. A `.env` in the invocation directory overrides those defaults.

```bash
gymrec list_environments --roms-path ./roms
ROMS_PATH=./roms gymrec
```

Run the CLI with the installed `gymrec` command. For local development without the tool install, use `uv run gymrec ...` or `uv run python main.py ...` from the repo root.

## Commands

```bash
gymrec login                                      # authenticate with Hugging Face Hub
gymrec list_environments                         # list Atari, Stable-Retro, and VizDoom envs with ROM status
gymrec reindex_games                             # re-scan ROMS_PATH and refresh the game cache

gymrec record BreakoutNoFrameskip-v4             # record human gameplay
gymrec record BreakoutNoFrameskip-v4 --dry-run   # save locally without upload prompt
gymrec record BreakoutNoFrameskip-v4 --storage lossless-video --dry-run
gymrec record BreakoutNoFrameskip-v4 --upload-live
gymrec record SuperMarioBros-Nes-v0 --agent random --headless --episodes 100
gymrec record SuperMarioBros-Nes-v0 --backend supermariobrosnes-turbo --dry-run
gymrec record BreakoutNoFrameskip-v4 --agent breakout --headless --episodes 50
gymrec record hf://tsilva/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo --headless --episodes 10 --dry-run
gymrec record https://huggingface.co/tsilva/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo --headless --episodes 10 --dry-run

gymrec upload BreakoutNoFrameskip-v4             # upload new local episodes to Hub
gymrec upload BreakoutNoFrameskip-v4 --replace   # replace remote files with local dataset
gymrec playback BreakoutNoFrameskip-v4           # replay recorded actions
gymrec playback BreakoutNoFrameskip-v4 --verify  # compare replay frames against recorded frames
gymrec playback SuperMarioBros-Nes-v0 --backend stable-retro
gymrec playback SuperMarioBros-Nes-v0 --backend supermariobrosnes-turbo

gymrec video BreakoutNoFrameskip-v4              # export all episodes to MP4
gymrec video BreakoutNoFrameskip-v4 --range 3-7  # export a 1-based episode range
gymrec video BreakoutNoFrameskip-v4 --first 5
gymrec video BreakoutNoFrameskip-v4 --last 5

gymrec import_roms ./roms                        # import Stable-Retro ROMs
gymrec minari-export BreakoutNoFrameskip-v4      # export local data to Minari format
```

## Usage

Human recording opens a pygame window. Press `Space` to start recording, use the environment-specific controls printed in the terminal, press `Tab` to toggle the overlay, use `+`/`-` to adjust FPS, and press `Esc` to stop. For backends that expose deterministic control knobs, human recording forces one environment step per emulator frame and disables sticky actions (`frameskip=1` / sticky probability `0.0`); the resolved kwargs are saved in local metadata and reused for local playback.

Agent recording supports `human`, `random`, `mario`, and `breakout`. `--headless` is for agent mode only and requires `--episodes`.

`record` also accepts Hugging Face rlab SB3 policy bundles. Each model repo must contain versioned `model.json` and `recipe.json` documents plus the checkpoint declared by `model.json`. gymrec validates the document versions, file sizes, SHA-256 bindings, algorithm/model identity, resolved evaluation environment, provider, task, and action sampling mode before execution. Unknown versions and unsupported providers fail explicitly.

For Mario bundles, gymrec creates the exact native vector provider declared by `recipe.eval.environment` (`supermariobrosnes-turbo` or `stable-retro-turbo`). Crop, grayscale, resize algorithm, layout, frame stacking, max pooling, frame skip, sticky actions, and provider flags are executed by that provider; gymrec does not recreate the preprocessing in Python. The policy receives the native policy observation while the dataset stores raw RGB renders and backend-neutral nine-button NES action vectors. The recipe's Mario reward/termination contract and evaluation seed are also applied. Action sampling follows `recipe.json` by default; use `--deterministic` or `--stochastic` only to override it. `--device cpu`, `--device mps`, or `--device cuda` overrides SB3 device selection. `--hf-file` is only an optional assertion that the requested checkpoint matches `model.json`.

Policy-recorded rows include the source repo/revision, recipe and checkpoint SHA-256 values, and both bundle format versions. These fields make the policy/environment contract recoverable from the trajectory dataset without relying on a mutable model-repo branch.

Observation storage defaults to `lossless-video`, which stores canonical observations as per-episode lossless RGB streams under `videos/<episode>.rgb.mkv.bin` plus table rows containing `video_path`, `frame_index`, and `frame_sha256`; each canonical stream is decoded and hash-verified before the recording is accepted. Browser-friendly `videos/<episode>.preview.mp4` files are preview-only and are not used for replay/training. Lossless video storage requires `ffmpeg` and `ffprobe` on `PATH`. Use `--storage images` to store one lossless WebP-backed HF `Image` row per observation instead.

Use `--upload-live` to preflight Hugging Face access before gameplay and upload each completed episode as its own shard while recording. In live mode, `lossless-video` streams frames directly into ffmpeg instead of buffering the whole episode in memory; completed steps are journaled with a recoverable terminal-frame candidate, verified episodes are uploaded immediately, failed or interrupted uploads are left in a resumable local queue, and `gymrec upload <env_id>` recovers/retries that queue. Live upload is incompatible with `--dry-run` and skips preview MP4 generation.

Hub uploads append new parquet shards only when the existing remote dataset has the same storage layout as the local dataset. If a remote dataset already uses a different current storage format, `gymrec upload` refuses to append so the Dataset Viewer does not show a mixed schema; use `gymrec upload <env_id> --replace` to intentionally rewrite the remote repo from the current local dataset.

The interactive recording menu only shows Atari and Stable-Retro environments whose ROMs are installed in the active Python environment. It uses a full-screen terminal menu by default and falls back to a plain text search prompt when the terminal cannot support it. Set `GYMREC_TEXT_MENU=1` to force the text selector. `list_environments` also shows missing ROMs for backends that register games separately from installed game files.

Playback uses the local dataset first, then falls back to the Hugging Face Hub dataset repo. Video export requires `ffmpeg` and writes MP4 files from local data or downloaded Hub data.

`SuperMarioBros-Nes-v0` is the logical environment ID for both Mario runtimes. Use `--backend stable-retro` or `--backend supermariobrosnes-turbo` on `record` and `playback` to select the runtime without changing the dataset name. The exact nine-button NES `MultiBinary` action vector is stored unchanged, so one captured trajectory can be replayed through either backend for parity checks and later video capture. Playback uses an explicit `--backend` override when supplied; otherwise it uses recorded capture-backend metadata when available and defaults older Mario datasets to Stable-Retro.

## Notes

- Requires Python `>=3.12,<3.13` and `uv`.
- Hugging Face uploads use dataset repos named `{username}/gymrec__{encoded_env_id}` by default.
- Local datasets are stored under `~/.gymrec/datasets` by default.
- `config.toml` controls display scale, FPS defaults, local storage path/format, dataset metadata, and overlay defaults.
- `keymappings.toml` controls Atari, VizDoom, and Stable-Retro keyboard bindings.
- Stable-Retro support uses the latest resolvable `stable-retro-turbo`, which keeps the `stable_retro` import name and provides PyPI wheels for macOS arm64.
- Native Mario playback uses the PyPI `supermariobrosnes-turbo` package with one lane, one thread, RGB HWC observations, `frame_stack=1`, `frame_skip=1`, sticky actions disabled, and autoreset disabled.
- `ffmpeg` must be available on `PATH` for `video` exports; `ffmpeg` and `ffprobe` are required for `--storage lossless-video`.
- `minari-export` requires Minari. For the installed tool, run `./install.sh --with 'minari>=0.5.0'`; for repository development, run `uv sync --extra minari`.

## Architecture

![gymrec architecture diagram](./architecture.png)

## License

[MIT](LICENSE)
