<div align="center">
  <img src="https://raw.githubusercontent.com/tsilva/gymrec/main/logo.png" alt="gymrec" width="512"/>

  **🎮 Record and replay gameplay from Gymnasium environments as Hugging Face datasets 📊**
</div>

gymrec is a Python CLI for collecting gameplay from Gymnasium environments and saving it as replayable Hugging Face datasets. It supports human keyboard capture, built-in agent policies, local dataset storage, Hub uploads, playback verification, and MP4 exports.

It works across Atari through ALE-py, Stable-Retro console environments, and VizDoom. Recordings store observations, actions, rewards, episode metadata, collector provenance, and the gymrec version used to collect the run.

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

`ROMS_PATH` is passed to ALE-py as `ALE_ROMS_DIR` for Atari games; if `ROMS_PATH` points at a single file, ALE-py receives its parent directory. For Stable-Retro, gymrec scans and imports matching ROMs from `ROMS_PATH` before listing or launching games. You can also pass it directly on the command line:

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
gymrec record SuperMarioBros-Nes --agent random --headless --episodes 100
gymrec record BreakoutNoFrameskip-v4 --agent breakout --headless --episodes 50
gymrec record BreakoutNoFrameskip-v4 --agent random --headless --episodes 100 --workers 5

gymrec upload BreakoutNoFrameskip-v4             # upload new local episodes to Hub
gymrec playback BreakoutNoFrameskip-v4           # replay recorded actions
gymrec playback BreakoutNoFrameskip-v4 --verify  # compare replay frames against recorded frames

gymrec video BreakoutNoFrameskip-v4              # export all episodes to MP4
gymrec video BreakoutNoFrameskip-v4 --range 3-7  # export a 1-based episode range
gymrec video BreakoutNoFrameskip-v4 --first 5
gymrec video BreakoutNoFrameskip-v4 --last 5

gymrec import_roms ./roms                        # import Stable-Retro ROMs
gymrec minari-export BreakoutNoFrameskip-v4      # export local data to Minari format
```

## Usage

Human recording opens a pygame window. Press `Space` to start recording, use the environment-specific controls printed in the terminal, press `Tab` to toggle the overlay, use `+`/`-` to adjust FPS, and press `Esc` to stop.

Agent recording supports `human`, `random`, `mario`, and `breakout`. `--headless` is for agent mode only and requires `--episodes`; `--workers` runs parallel headless collection and cannot exceed the requested episode count.

The interactive recording menu only shows Atari and Stable-Retro environments whose ROMs are installed in the active Python environment. It uses a full-screen terminal menu by default and falls back to a plain text search prompt when the terminal cannot support it. Set `GYMREC_TEXT_MENU=1` to force the text selector. `list_environments` also shows missing ROMs for backends that register games separately from installed game files.

Playback uses the local dataset first, then falls back to the Hugging Face Hub dataset repo. Video export requires `ffmpeg` and writes MP4 files from local data or downloaded Hub data.

## Notes

- Requires Python `>=3.12,<3.13` and `uv`.
- Hugging Face uploads use dataset repos named `{username}/gymrec__{encoded_env_id}` by default.
- Local datasets are stored under `~/.gymrec/datasets` by default.
- `config.toml` controls display scale, FPS defaults, local storage, dataset metadata, and overlay defaults.
- `keymappings.toml` controls Atari, VizDoom, and Stable-Retro keyboard bindings.
- Stable-Retro support uses the latest resolvable `stable-retro-turbo`, which keeps the `stable_retro` import name and provides PyPI wheels for macOS arm64.
- `ffmpeg` must be available on `PATH` for `video` exports.
- `minari-export` requires Minari; install it with `uv sync --extra minari` or `uv pip install 'minari>=0.5.0'`.

## Architecture

![gymrec architecture diagram](./architecture.png)

## License

[MIT](LICENSE)
