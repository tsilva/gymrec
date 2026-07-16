import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import gymnasium as gym
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

STYLE_KEY = "bold cyan"
STYLE_ACTION = "bold white"
STYLE_ENV = "bold green"
STYLE_PATH = "dim"
STYLE_CMD = "bold yellow"
STYLE_SUCCESS = "bold green"
STYLE_FAIL = "bold red"
STYLE_INFO = "cyan"

_project_dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_project_dotenv, override=False)
load_dotenv(find_dotenv(usecwd=True), override=True)

_initialized = False
_stableretro_roms_path_imported = set()

HUMAN_RECORD_FRAMESKIP = 1
HUMAN_RECORD_STICKY_ACTION_PROB = 0.0


def _get_gymrec_version():
    """Return version string like '0.1.0+abc1234' (or just '0.1.0' if git unavailable)."""
    pyproject_path = os.path.join(os.path.dirname(__file__), "pyproject.toml")
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        version = data.get("project", {}).get("version", "unknown")
    except Exception:
        version = "unknown"

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(__file__),
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
            return f"{version}+{git_hash}"
    except Exception:
        pass

    return version


def _json_default(obj):
    """JSON serializer for numpy types found in info dicts."""
    import numpy as _np

    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    return str(obj)


def _build_key_name_map(pygame):
    """Build a mapping from human-readable key names to pygame key constants."""
    key_map = {
        "up": pygame.K_UP,
        "down": pygame.K_DOWN,
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "space": pygame.K_SPACE,
        "tab": pygame.K_TAB,
        "return": pygame.K_RETURN,
        "lshift": pygame.K_LSHIFT,
        "rshift": pygame.K_RSHIFT,
        "lctrl": pygame.K_LCTRL,
        "rctrl": pygame.K_RCTRL,
    }
    for c in "abcdefghijklmnopqrstuvwxyz":
        key_map[c] = getattr(pygame, f"K_{c}")
    for d in "0123456789":
        key_map[d] = getattr(pygame, f"K_{d}")
    return key_map


def _resolve_key(name, key_map):
    """Resolve a human-readable key name to a pygame constant."""
    name_lower = name.lower()
    if name_lower not in key_map:
        raise ValueError(
            f"Unknown key name '{name}' in keymappings.toml. "
            f"Valid keys: {', '.join(sorted(key_map.keys()))}"
        )
    return key_map[name_lower]


def _load_keymappings(pygame):
    """Load keymappings from the bundled keymappings.toml file."""
    key_map = _build_key_name_map(pygame)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keymappings.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing bundled keymappings.toml at {config_path}. "
            "Reinstall gymrec or restore the repository config file."
        )

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    for section in ("atari", "vizdoom", "stable_retro"):
        if section not in config:
            raise ValueError(f"keymappings.toml is missing required [{section}] section")

    def parse_key_bindings(section):
        return {
            _resolve_key(key_name, key_map): action
            for key_name, action in config.get(section, {}).items()
        }

    start_key = _resolve_key(config.get("general", {}).get("start_key", "space"), key_map)
    atari = parse_key_bindings("atari")
    vizdoom = parse_key_bindings("vizdoom")
    retro = {
        console: {_resolve_key(key_name, key_map): action for key_name, action in bindings.items()}
        for console, bindings in config["stable_retro"].items()
    }

    return start_key, atari, vizdoom, retro


DEFAULT_CONFIG = {
    "display": {"scale_factor": 2},
    "fps_defaults": {"atari": 60, "vizdoom": 45, "retro": 90},
    "dataset": {
        "repo_prefix": "gymrec__",
        "license": "mit",
        "task_categories": ["reinforcement-learning"],
    },
    "storage": {
        "local_dir": os.path.join(os.path.expanduser("~"), ".gymrec", "datasets"),
        "format": "lossless-video",
    },
    "overlay": {
        "font_size": 48,
        "text_color": [255, 255, 255],
        "background_color": [0, 0, 0, 180],
        "fps": 30,
        "text": "Press SPACE to start",
    },
}

CONFIG = None

RETRO_PLATFORMS = frozenset(
    {
        "Nes",
        "GameBoy",
        "Snes",
        "GbAdvance",
        "GbColor",
        "Genesis",
        "PCEngine",
        "Saturn",
        "32x",
        "Sms",
        "GameGear",
        "SCD",
        "Atari2600",
    }
)

BACKEND_LABELS = {
    "atari": "Atari (ALE-py)",
    "vizdoom": "VizDoom",
    "stable-retro": "Stable-Retro",
    "stable-retro-turbo": "Stable-Retro TurboVecEnv",
    "supermariobrosnes-turbo": "SuperMarioBros-Nes-turbo",
}

SELECTABLE_RUNTIME_BACKENDS = (
    "stable-retro",
    "stable-retro-turbo",
    "supermariobrosnes-turbo",
)
SUPERMARIOBROS_NES_ENV_ID = "SuperMarioBros-Nes-v0"
SUPERMARIOBROS_NES_ROM_SHA256 = "f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de"
NES_BUTTON_ORDER = ("B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A")
NES_ACTION_ENCODING = "stable-retro-multibinary-9"

HUGGINGFACE_MODEL_SCHEME = "hf://"
HUGGINGFACE_MODEL_URL_HOST = "huggingface.co"
RLAB_RECIPE_FILENAME = "recipe.json"
RLAB_MODEL_FILENAME = "model.json"
RLAB_RECIPE_DOCUMENT_TYPE = "rlab.recipe"
RLAB_MODEL_DOCUMENT_TYPE = "rlab.model"
RLAB_RECIPE_FORMAT_VERSION = 1
RLAB_MODEL_FORMAT_VERSION = 1
RLAB_RECIPE_SCHEMA_VERSION = 2
OBSERVATION_IMAGE_KEYS = ("obs", "image", "screen")
STORAGE_FORMAT_IMAGES = "images"
STORAGE_FORMAT_LOSSLESS_VIDEO = "lossless-video"
STORAGE_FORMATS = (STORAGE_FORMAT_IMAGES, STORAGE_FORMAT_LOSSLESS_VIDEO)
VIDEO_ARTIFACT_DIR = "videos"
PREVIEW_VIDEO_ARTIFACT_DIR = "videos"
CANONICAL_VIDEO_SUFFIX = ".rgb.mkv.bin"
PREVIEW_VIDEO_SUFFIX = ".preview.mp4"
RUNTIME_VIDEO_BASE_COLUMN = "_gymrec_video_base_path"
RUNTIME_HF_REPO_COLUMN = "_gymrec_hf_repo_id"
ENVIRONMENT_METADATA_FILENAME = "gymrec-metadata.json"
_VIDEO_DECODE_CACHE = {}

NES_SIMPLE_ACTION_MASKS = {
    "noop": (),
    "right": (7,),
    "right_b": (7, 0),
    "right_a": (7, 8),
    "right_a_b": (7, 8, 0),
    "a": (8,),
    "left": (6,),
}

STABLE_RETRO_DISCRETE_ACTION_SETS = {
    "SuperMarioBros-Nes-v0": {
        "simple": ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left"),
        "right": ("right", "right_b", "right_a", "right_a_b"),
    }
}


def _is_nes_controller_env(env):
    """Return whether an environment exposes the shared nine-button NES controller."""
    if bool(getattr(env, "_gymrec_nes_controller", False)):
        return True
    if not bool(getattr(env, "_stable_retro", False)):
        return False
    return getattr(getattr(env, "unwrapped", None), "system", None) == "Nes"


def _extract_observation_image(observation):
    """Extract the image array from an observation, including dict observations."""
    if isinstance(observation, dict):
        for key in OBSERVATION_IMAGE_KEYS:
            if key in observation:
                return observation[key]
    return observation


def _normalize_storage_format(value):
    """Normalize and validate a configured storage format."""
    value = str(value or STORAGE_FORMAT_IMAGES).strip().lower()
    if value not in STORAGE_FORMATS:
        raise ValueError(
            f"Unknown storage format '{value}'. Expected one of: {', '.join(STORAGE_FORMATS)}"
        )
    return value


def _observation_to_rgb_array(observation):
    """Normalize an environment observation to contiguous HxWx3 uint8 RGB bytes."""
    frame = _extract_observation_image(observation)
    frame_array = np.array(frame, dtype=np.uint8)

    if frame_array.ndim == 2:
        frame_array = np.repeat(frame_array[:, :, None], 3, axis=2)
    elif frame_array.ndim == 3 and frame_array.shape[2] == 1:
        frame_array = np.repeat(frame_array, 3, axis=2)
    elif frame_array.ndim == 3 and frame_array.shape[2] >= 3:
        frame_array = frame_array[:, :, :3]
    else:
        raise ValueError(f"Unsupported frame shape: {frame_array.shape}")

    return np.ascontiguousarray(frame_array)


def _recording_observation(env, observation):
    """Return raw RGB gameplay when an env exposes preprocessed policy observations."""
    if bool(getattr(env, "_gymrec_policy_observations", False)):
        frame = env.render()
        if frame is None:
            raise RuntimeError(
                "The recipe-selected environment returned no raw RGB render for recording"
            )
        return frame
    return observation


def _sha256_rgb(frame_array):
    """Hash canonical RGB frame bytes."""
    return hashlib.sha256(np.ascontiguousarray(frame_array).tobytes()).hexdigest()


def _require_lossless_video_tools():
    """Return ffmpeg/ffprobe paths or raise before starting a video-backed recording."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    missing = []
    if ffmpeg_path is None:
        missing.append("ffmpeg")
    if ffprobe_path is None:
        missing.append("ffprobe")
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} required for --storage {STORAGE_FORMAT_LOSSLESS_VIDEO}"
        )
    return ffmpeg_path, ffprobe_path


def _encode_lossless_rgb_video(frames, output_path, fps, ffmpeg_path=None):
    """Encode a sequence of RGB frames to a canonical lossless RGB Matroska stream."""
    writer = _StreamingLosslessVideoWriter(output_path, fps, ffmpeg_path=ffmpeg_path)
    try:
        for frame in frames:
            writer.write(frame)
        writer.close()
    except Exception:
        writer.abort()
        raise


class _StreamingLosslessVideoWriter:
    """Write canonical RGB frames to ffmpeg without buffering an episode in memory."""

    def __init__(self, output_path, fps, ffmpeg_path=None):
        self.output_path = output_path
        self.fps = fps
        self.ffmpeg_path = ffmpeg_path or _require_lossless_video_tools()[0]
        self.process = None
        self.shape = None
        self.frames_written = 0

    def _start(self, frame):
        self.shape = tuple(frame.shape)
        height, width = frame.shape[:2]
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(max(int(round(float(self.fps))), 1)),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            "-pix_fmt",
            "rgb24",
            "-f",
            "matroska",
            self.output_path,
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        frame = np.ascontiguousarray(frame)
        if self.process is None:
            self._start(frame)
        elif tuple(frame.shape) != self.shape:
            raise ValueError(f"All video frames must have shape {self.shape}; got {frame.shape}")
        self.process.stdin.write(frame.tobytes())
        self.frames_written += 1

    def close(self):
        if self.process is None:
            raise ValueError("Cannot encode an empty video")
        try:
            self.process.stdin.close()
            stderr = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = self.process.wait()
            if return_code != 0:
                raise RuntimeError(stderr or f"ffmpeg exited with status {return_code}")
        finally:
            if self.process.stderr is not None:
                self.process.stderr.close()

    def abort(self):
        if self.process is None:
            return
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.wait()
        if self.process.stderr is not None:
            self.process.stderr.close()


def _encode_browser_preview_video(frames, output_path, fps, ffmpeg_path=None):
    """Encode a lossy browser-safe preview MP4. This is never canonical data."""
    if not frames:
        raise ValueError("Cannot encode an empty preview video")
    ffmpeg_path = ffmpeg_path or _require_lossless_video_tools()[0]

    first = frames[0]
    height, width = first.shape[:2]
    for frame in frames:
        if frame.shape != first.shape:
            raise ValueError(f"All preview frames must have shape {first.shape}; got {frame.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(int(round(float(fps))), 1)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(stderr or f"ffmpeg exited with status {return_code}")
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        raise
    finally:
        if process.stderr is not None:
            process.stderr.close()


def _decode_lossless_rgb_video(video_path, width, height, cache=True):
    """Decode a lossless RGB video into an NxHxWx3 uint8 array."""
    abs_path = os.path.abspath(video_path)
    stat = os.stat(abs_path)
    cache_key = (abs_path, int(width), int(height), stat.st_mtime_ns, stat.st_size)
    if cache and cache_key in _VIDEO_DECODE_CACHE:
        return _VIDEO_DECODE_CACHE[cache_key]

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to decode video-backed observations")
    cmd = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        abs_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"ffmpeg exited with status {result.returncode}")

    frame_size = int(width) * int(height) * 3
    if frame_size <= 0 or len(result.stdout) % frame_size != 0:
        raise RuntimeError(f"Decoded video byte count is invalid for {video_path}")
    frame_count = len(result.stdout) // frame_size
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        (frame_count, int(height), int(width), 3)
    )
    if cache:
        _VIDEO_DECODE_CACHE[cache_key] = frames
    return frames


def _verify_lossless_rgb_video_stream(video_path, width, height, expected_hashes, ffmpeg_path=None):
    """Verify a canonical video by streaming decoded RGB frames and comparing hashes."""
    ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to verify video-backed observations")

    frame_size = int(width) * int(height) * 3
    if frame_size <= 0:
        raise RuntimeError(f"Decoded video dimensions are invalid for {video_path}")

    cmd = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        os.path.abspath(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame_index, expected_hash in enumerate(expected_hashes):
            chunk = process.stdout.read(frame_size)
            if len(chunk) != frame_size:
                raise RuntimeError(
                    f"Decoded video ended at frame {frame_index}; expected {len(expected_hashes)} frames"
                )
            decoded_frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                (int(height), int(width), 3)
            )
            actual_hash = _sha256_rgb(decoded_frame)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    "Lossless video verification failed for "
                    f"{video_path} frame {frame_index}: {actual_hash} != {expected_hash}"
                )

        extra = process.stdout.read(1)
        if extra:
            raise RuntimeError(
                f"Decoded video has more than {len(expected_hashes)} frames: {video_path}"
            )
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(stderr or f"ffmpeg exited with status {return_code}")
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if process.poll() is None:
            process.wait()


def _episode_progress(transient=False):
    """Create the standard episode/frame progress display."""
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )


def _env_id_has_retro_platform(env_id):
    """Return True when an env id contains a known Stable-Retro platform token."""
    return any(f"-{platform}" in env_id for platform in RETRO_PLATFORMS)


def classify_env_id(env_id, stable_retro_envs=None):
    """Classify an environment id as atari, vizdoom, or stable-retro."""
    if env_id.startswith("ALE/"):
        return "atari"
    if env_id.startswith(("Vizdoom", "vizdoom")):
        return "vizdoom"
    if (
        stable_retro_envs is not None and env_id in stable_retro_envs
    ) or _env_id_has_retro_platform(env_id):
        return "stable-retro"
    return "atari"


def _capture_backend_from_dataset(dataset):
    """Return the most recently appended runtime backend carried by a dataset."""
    if dataset is None or "capture_backend" not in (getattr(dataset, "column_names", None) or []):
        return None
    for value in reversed(dataset["capture_backend"]):
        if value:
            return str(value)
    return None


def _first_dataset_value(dataset, column):
    if dataset is None or column not in (getattr(dataset, "column_names", None) or []):
        return None
    for value in reversed(dataset[column]):
        if value is not None and value != "":
            return value
    return None


def _environment_metadata_from_dataset(dataset):
    """Recover environment provenance from current row columns or a Hub sidecar."""
    metadata = {}
    for column, metadata_key in (
        ("logical_env_id", "env_id"),
        ("capture_backend", "capture_backend"),
        ("state_name", "state_name"),
        ("rom_sha256", "rom_sha256"),
        ("action_encoding", "action_encoding"),
        ("frame_skip", "frameskip"),
        ("sticky_action_prob", "sticky_actions"),
        ("policy_source_repo", "policy_source_repo"),
        ("policy_source_revision", "policy_source_revision"),
        ("policy_recipe_sha256", "policy_recipe_sha256"),
        ("policy_checkpoint_sha256", "policy_checkpoint_sha256"),
        ("policy_recipe_format_version", "policy_recipe_format_version"),
        ("policy_model_format_version", "policy_model_format_version"),
    ):
        value = _first_dataset_value(dataset, column)
        if value is not None:
            metadata[metadata_key] = value
    button_order = _first_dataset_value(dataset, "nes_button_order")
    if button_order is not None:
        try:
            metadata["nes_button_order"] = json.loads(button_order)
        except (TypeError, json.JSONDecodeError):
            metadata["nes_button_order"] = button_order
    if "frameskip" in metadata or "sticky_actions" in metadata:
        metadata["env_make_kwargs"] = {}
        atari = metadata.get("capture_backend") == "atari"
        if "frameskip" in metadata:
            metadata["env_make_kwargs"]["frameskip" if atari else "frame_skip"] = int(
                metadata["frameskip"]
            )
        if "sticky_actions" in metadata:
            metadata["env_make_kwargs"][
                "repeat_action_probability" if atari else "sticky_action_prob"
            ] = float(metadata["sticky_actions"])
    return metadata


def _attach_hub_environment_metadata(dataset, hf_repo_id):
    """Attach sidecar provenance to legacy Hub shards as runtime-only columns."""
    try:
        metadata_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=ENVIRONMENT_METADATA_FILENAME,
            repo_type="dataset",
        )
        with open(metadata_path, "r") as file_obj:
            metadata = json.load(file_obj)
    except Exception:
        return dataset

    row_values = {
        "logical_env_id": metadata.get("env_id"),
        "capture_backend": metadata.get("capture_backend"),
        "state_name": metadata.get("state_name"),
        "rom_sha256": metadata.get("rom_sha256"),
        "nes_button_order": (
            json.dumps(metadata.get("nes_button_order"))
            if metadata.get("nes_button_order") is not None
            else None
        ),
        "action_encoding": metadata.get("action_encoding"),
        "frame_skip": metadata.get("frameskip"),
        "sticky_action_prob": metadata.get("sticky_actions"),
        "policy_source_repo": metadata.get("policy_source_repo"),
        "policy_source_revision": metadata.get("policy_source_revision"),
        "policy_recipe_sha256": metadata.get("policy_recipe_sha256"),
        "policy_checkpoint_sha256": metadata.get("policy_checkpoint_sha256"),
        "policy_recipe_format_version": metadata.get("policy_recipe_format_version"),
        "policy_model_format_version": metadata.get("policy_model_format_version"),
    }
    for column, value in row_values.items():
        if column not in dataset.column_names and value is not None:
            dataset = dataset.add_column(column, [value] * len(dataset))
    return dataset


def resolve_runtime_backend(env_id, requested=None, metadata=None, recorded_backend=None):
    """Resolve logical environment identity separately from its runtime provider."""
    logical_backend = classify_env_id(env_id)
    if logical_backend == "atari":
        logical_backend = classify_env_id(env_id, stable_retro_envs=set(_get_stableretro_envs()))
    candidate = requested
    if candidate is None and metadata:
        candidate = metadata.get("capture_backend")
    if candidate is None:
        candidate = recorded_backend
    if candidate is None:
        candidate = logical_backend

    if candidate == "supermariobrosnes-turbo":
        if env_id != SUPERMARIOBROS_NES_ENV_ID:
            raise ValueError(
                "The supermariobrosnes-turbo backend only supports "
                f"{SUPERMARIOBROS_NES_ENV_ID}; got {env_id}."
            )
        return candidate
    if candidate in {"stable-retro", "stable-retro-turbo"}:
        if logical_backend != "stable-retro":
            raise ValueError(f"The {candidate} backend does not support {env_id}.")
        return candidate
    if candidate in {"atari", "vizdoom"} and candidate == logical_backend:
        return candidate
    if requested is not None or recorded_backend is not None:
        raise ValueError(f"Unknown runtime backend: {candidate}")
    return logical_backend


def _load_config():
    """Load configuration from config.toml, falling back to defaults."""
    import copy

    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)
        for section in config:
            if section in user_config:
                config[section].update(user_config[section])
    config["storage"]["local_dir"] = os.path.abspath(
        os.path.expanduser(config["storage"]["local_dir"])
    )
    return config


def _gymrec_cmd(*parts):
    """Format a user-facing command with the installed CLI entrypoint."""
    return " ".join(("gymrec", *(str(part) for part in parts)))


def _get_roms_path() -> str | None:
    roms_path = os.environ.get("ROMS_PATH")
    if not roms_path:
        return None
    return os.path.abspath(os.path.expanduser(roms_path))


def _configure_atari_roms_path():
    """Use ROMS_PATH as ALE-py's ROM directory unless ALE_ROMS_DIR is explicit."""
    roms_path = _get_roms_path()
    if roms_path and not os.environ.get("ALE_ROMS_DIR"):
        os.environ["ALE_ROMS_DIR"] = (
            os.path.dirname(roms_path) if os.path.isfile(roms_path) else roms_path
        )


def _roms_path_cache_file() -> str:
    return os.path.expanduser("~/.gymrec/roms_path_imports.json")


def _roms_path_fingerprint(path: str) -> dict:
    file_count = 0
    total_size = 0
    newest_mtime_ns = 0

    if os.path.isfile(path):
        stat = os.stat(path)
        return {
            "file_count": 1,
            "total_size": stat.st_size,
            "newest_mtime_ns": stat.st_mtime_ns,
        }

    for root, _dirs, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                stat = os.stat(filepath)
            except OSError:
                continue
            file_count += 1
            total_size += stat.st_size
            newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)

    return {
        "file_count": file_count,
        "total_size": total_size,
        "newest_mtime_ns": newest_mtime_ns,
    }


def _load_roms_path_import_cache() -> dict:
    cache_file = _roms_path_cache_file()
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_roms_path_import_cache(cache: dict):
    cache_file = _roms_path_cache_file()
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def _roms_path_cache_entry_current(cache_entry, fingerprint) -> bool:
    """Return True if source and Stable-Retro target imports still look current."""
    if not isinstance(cache_entry, dict):
        return False

    cached_fingerprint = cache_entry.get("fingerprint")
    stable_retro_games = cache_entry.get("stable_retro_games")
    if cached_fingerprint != fingerprint or stable_retro_games is None:
        return False

    try:
        import stable_retro as retro
    except Exception:
        return False

    for game in stable_retro_games:
        try:
            retro.data.get_romfile_path(game, retro.data.Integrations.ALL)
        except FileNotFoundError:
            return False
    return True


def _ensure_stableretro_roms_path_imported(quiet: bool = True, force: bool = False) -> int:
    """Import ROMS_PATH into Stable-Retro's integration store once per process."""
    roms_path = _get_roms_path()
    if not roms_path or (not force and roms_path in _stableretro_roms_path_imported):
        return 0

    _stableretro_roms_path_imported.add(roms_path)
    if not os.path.exists(roms_path):
        if not quiet:
            console.print(f"[{STYLE_FAIL}]ROMS_PATH does not exist: {roms_path}[/]")
        return 0

    if quiet:
        with console.status(
            f"[{STYLE_INFO}]Checking ROMS_PATH game index: [{STYLE_PATH}]{roms_path}[/]"
        ):
            fingerprint = _roms_path_fingerprint(roms_path)
    else:
        fingerprint = _roms_path_fingerprint(roms_path)
    cache = _load_roms_path_import_cache()
    if not force and _roms_path_cache_entry_current(cache.get(roms_path), fingerprint):
        return 0

    if quiet:
        console.print(f"[{STYLE_INFO}]Indexing games from ROMS_PATH: [{STYLE_PATH}]{roms_path}[/]")
        with console.status(f"[{STYLE_INFO}]Scanning and importing matching ROMs...[/]"):
            imported_games, stable_retro_games = _import_roms(
                roms_path, quiet=quiet, source_label="ROMS_PATH", return_games=True
            )
    else:
        imported_games, stable_retro_games = _import_roms(
            roms_path, quiet=quiet, source_label="ROMS_PATH", return_games=True
        )
    cache[roms_path] = {
        "fingerprint": fingerprint,
        "stable_retro_games": stable_retro_games,
    }
    _save_roms_path_import_cache(cache)
    if quiet:
        console.print(
            f"[{STYLE_SUCCESS}]Indexed {imported_games} Stable-Retro ROM(s) from ROMS_PATH[/]"
        )
    return imported_games


def _lazy_init():
    """Import heavy dependencies and initialize key bindings on first use."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    global CONFIG
    global np, pygame, PILImage
    global \
        whoami, \
        DatasetCardData, \
        HfApi, \
        login, \
        get_token, \
        CommitOperationAdd, \
        CommitOperationDelete, \
        hf_hub_download
    global Dataset, HFImage, Value, load_dataset, load_from_disk
    global START_KEY, ATARI_KEY_BINDINGS, VIZDOOM_KEY_BINDINGS, STABLE_RETRO_KEY_BINDINGS

    import numpy as np
    import pygame
    from datasets import (
        Dataset,
        Value,
        load_dataset,
        load_from_disk,
    )
    from datasets import (
        Image as HFImage,
    )
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationDelete,
        DatasetCardData,
        HfApi,
        get_token,
        hf_hub_download,
        login,
        whoami,
    )
    from PIL import Image as PILImage

    START_KEY, ATARI_KEY_BINDINGS, VIZDOOM_KEY_BINDINGS, STABLE_RETRO_KEY_BINDINGS = (
        _load_keymappings(pygame)
    )
    CONFIG = _load_config()


def ensure_hf_login(force=False) -> bool:
    """Ensure user is logged in to Hugging Face Hub. Prompts interactively if needed."""
    _lazy_init()
    token = get_token()

    if token and not force:
        return True

    if token and force:
        try:
            info = whoami(token=token)
            username = info.get("name", "unknown")
            console.print(f"[{STYLE_SUCCESS}]Already logged in as [{STYLE_ENV}]{username}[/][/]")
            if not Confirm.ask("Re-login with a different token?", default=False):
                return True
        except Exception:
            console.print(f"[{STYLE_FAIL}]Existing token is invalid.[/]")

    console.print(
        Panel(
            f"[{STYLE_ACTION}]Create a token at:[/] [{STYLE_CMD}]https://huggingface.co/settings/tokens[/]\n"
            f"Required permission: [{STYLE_INFO}]write[/]",
            title="Hugging Face Login",
            border_style="cyan",
        )
    )

    for attempt in range(1, 4):
        token_input = Prompt.ask("Paste your token", password=True)
        if not token_input.strip():
            console.print(f"[{STYLE_FAIL}]Empty token, try again.[/]")
            continue
        try:
            login(token=token_input.strip())
            info = whoami()
            username = info.get("name", "unknown")
            console.print(f"[{STYLE_SUCCESS}]Logged in as [{STYLE_ENV}]{username}[/][/]")
            return True
        except Exception as e:
            remaining = 3 - attempt
            if remaining > 0:
                console.print(
                    f"[{STYLE_FAIL}]Login failed: {e}[/] ({remaining} attempt{'s' if remaining > 1 else ''} left)"
                )
            else:
                console.print(f"[{STYLE_FAIL}]Login failed: {e}[/]")

    console.print(
        f"[{STYLE_FAIL}]Could not authenticate. Try:[/] [{STYLE_CMD}]{_gymrec_cmd('login')}[/]"
    )
    return False


# =============================================================================
# Input Sources and Policies
# =============================================================================


class InputSource(ABC):
    """Abstract base class for input sources (human or agent)."""

    @abstractmethod
    def get_action(self, observation) -> any:
        """Return action for the current observation."""
        pass


class HumanInputSource(InputSource):
    """Human input via pygame keyboard events."""

    def __init__(self, env, key_lock, current_keys):
        self.env = env
        self.key_lock = key_lock
        self.current_keys = current_keys
        self.key_to_action = None
        self._atari_key_bindings_raw = ATARI_KEY_BINDINGS
        self._atari_meaning_to_idx = None
        self._atari_key_to_meaning = {}
        self._vizdoom_buttons = None
        self._vizdoom_vector_map = None
        self.noop_action = 0

    def _resolve_atari_key_mapping(self):
        """Resolve meaning-based Atari key bindings to action indices."""
        standard_meaning_to_idx = {
            "NOOP": 0,
            "FIRE": 1,
            "UP": 2,
            "RIGHT": 3,
            "LEFT": 4,
            "DOWN": 5,
            "UPRIGHT": 6,
            "UPLEFT": 7,
            "DOWNRIGHT": 8,
            "DOWNLEFT": 9,
            "UPFIRE": 10,
            "RIGHTFIRE": 11,
            "LEFTFIRE": 12,
            "DOWNFIRE": 13,
            "UPRIGHTFIRE": 14,
            "UPLEFTFIRE": 15,
            "DOWNRIGHTFIRE": 16,
            "DOWNLEFTFIRE": 17,
        }

        meaning_to_idx = None
        try:
            meanings = self.env.unwrapped.get_action_meanings()
            if meanings:
                meaning_to_idx = {m.upper(): idx for idx, m in enumerate(meanings)}
        except (AttributeError, TypeError):
            pass

        if meaning_to_idx is None:
            meaning_to_idx = standard_meaning_to_idx

        resolved = {}
        for key, value in self._atari_key_bindings_raw.items():
            if isinstance(value, str):
                idx = meaning_to_idx.get(value.upper())
                if idx is not None:
                    resolved[key] = idx

        self.key_to_action = resolved
        self._atari_meaning_to_idx = meaning_to_idx
        self._atari_key_to_meaning = {}
        for key, value in self._atari_key_bindings_raw.items():
            if isinstance(value, str):
                self._atari_key_to_meaning[key] = value.upper()

    def _get_atari_action(self):
        """Return the Discrete action index for Atari environments."""
        if self.key_to_action is None:
            self._resolve_atari_key_mapping()

        pressed_meanings = set()
        for key in self.current_keys:
            if key in self._atari_key_to_meaning:
                pressed_meanings.add(self._atari_key_to_meaning[key])

        if not pressed_meanings:
            return self.noop_action

        composite = ""
        if "UP" in pressed_meanings:
            composite += "UP"
        elif "DOWN" in pressed_meanings:
            composite += "DOWN"
        if "RIGHT" in pressed_meanings:
            composite += "RIGHT"
        elif "LEFT" in pressed_meanings:
            composite += "LEFT"
        if "FIRE" in pressed_meanings:
            composite += "FIRE"

        if not composite:
            return self.noop_action

        if composite in self._atari_meaning_to_idx:
            return self._atari_meaning_to_idx[composite]

        for key in self.current_keys:
            if key in self.key_to_action:
                return self.key_to_action[key]
        return self.noop_action

    def _init_vizdoom_key_mapping(self):
        """Map important action names to their button indices."""
        available = [b.name for b in self.env.unwrapped.game.get_available_buttons()]
        offset = self.env.unwrapped.num_delta_buttons

        def idx(name):
            if name in available:
                return available.index(name) - offset
            return None

        mapping = {
            "ATTACK": idx("ATTACK"),
            "USE": idx("USE"),
            "MOVE_LEFT": idx("MOVE_LEFT"),
            "MOVE_RIGHT": idx("MOVE_RIGHT"),
            "MOVE_FORWARD": idx("MOVE_FORWARD"),
            "MOVE_BACKWARD": idx("MOVE_BACKWARD"),
            "TURN_LEFT": idx("TURN_LEFT"),
            "TURN_RIGHT": idx("TURN_RIGHT"),
            "SPEED": idx("SPEED"),
        }
        for i in range(1, 8):
            mapping[f"SELECT_WEAPON{i}"] = idx(f"SELECT_WEAPON{i}")

        space = self.env.action_space
        if isinstance(space, gym.spaces.Dict):
            space = space.get("binary")
        if isinstance(space, gym.spaces.Discrete):
            self._vizdoom_vector_map = {
                tuple(combo): i for i, combo in enumerate(self.env.unwrapped.button_map)
            }
        else:
            self._vizdoom_vector_map = None

        return {k: v for k, v in mapping.items() if v is not None}

    def _get_vizdoom_action(self):
        """Return the MultiBinary action vector for VizDoom environments."""
        if self._vizdoom_buttons is None:
            self._vizdoom_buttons = self._init_vizdoom_key_mapping()
        n_buttons = self.env.unwrapped.num_binary_buttons
        action = np.zeros(n_buttons, dtype=np.int32)

        pressed = self.current_keys
        alt = pygame.K_LALT in pressed or pygame.K_RALT in pressed

        def press(name):
            idx = self._vizdoom_buttons.get(name)
            if idx is not None and idx < n_buttons:
                action[idx] = 1

        for key, name in VIZDOOM_KEY_BINDINGS.items():
            if key in pressed:
                if key == pygame.K_LEFT:
                    press("MOVE_LEFT" if alt else name)
                elif key == pygame.K_RIGHT:
                    press("MOVE_RIGHT" if alt else name)
                else:
                    press(name)

        space = self.env.action_space
        if isinstance(space, gym.spaces.Dict):
            binary_space = space.get("binary")
            continuous_space = space.get("continuous")
            if isinstance(binary_space, gym.spaces.Discrete):
                if self._vizdoom_vector_map is None:
                    self._vizdoom_vector_map = {
                        tuple(c): i for i, c in enumerate(self.env.unwrapped.button_map)
                    }
                binary_action = self._vizdoom_vector_map.get(tuple(action), 0)
            else:
                binary_action = action

            if continuous_space is not None:
                cont_shape = continuous_space.shape
                continuous_action = np.zeros(cont_shape, dtype=np.float32)
                return {"binary": binary_action, "continuous": continuous_action}
            return {"binary": binary_action}

        if isinstance(space, gym.spaces.Discrete):
            if self._vizdoom_vector_map is None:
                self._vizdoom_vector_map = {
                    tuple(c): i for i, c in enumerate(self.env.unwrapped.button_map)
                }
            return self._vizdoom_vector_map.get(tuple(action), 0)

        return action

    def _get_retro_action(self):
        """Return a controller vector for Stable-Retro-compatible environments."""
        action = np.zeros(self.env.action_space.n, dtype=np.int32)
        platform = getattr(self.env.unwrapped, "system", None)
        mapping = STABLE_RETRO_KEY_BINDINGS.get(platform, {})
        for key in self.current_keys:
            idx = mapping.get(key)
            if idx is not None and idx < action.shape[0]:
                action[idx] = 1
        return action

    def get_action(self, observation):
        """Map pressed keys to actions for the current environment."""
        with self.key_lock:
            if hasattr(self.env, "_vizdoom") and self.env._vizdoom:
                return self._get_vizdoom_action()
            if bool(getattr(self.env, "_stable_retro", False)) or _is_nes_controller_env(self.env):
                return self._get_retro_action()
            return self._get_atari_action()


class AgentInputSource(InputSource):
    """Agent input via policy function."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self, **kwargs):
        """Reset any episode-local policy state."""
        if hasattr(self.policy, "reset"):
            try:
                self.policy.reset(**kwargs)
            except TypeError:
                self.policy.reset()

    def get_action(self, observation):
        """Get action from policy."""
        return self.policy(observation)

    def observe_step(self, reward, terminated, truncated, info):
        """Forward step results to the policy when it needs feedback."""
        if hasattr(self.policy, "observe_step"):
            self.policy.observe_step(reward, terminated, truncated, info)


# =============================================================================
# Policies
# =============================================================================


class BasePolicy(ABC):
    """Abstract base class for agent policies."""

    def __init__(self, action_space, env=None):
        self.action_space = action_space
        self.env = env

    def reset(self, **kwargs):
        """Reset any episode-local policy state."""
        pass

    def observe_step(self, reward, terminated, truncated, info):
        """Receive the result of the previously chosen action."""
        pass

    @abstractmethod
    def __call__(self, observation):
        """Return action for the given observation."""
        pass


class RandomPolicy(BasePolicy):
    """Random policy that samples from the action space."""

    def __call__(self, observation):
        """Sample a random action from the action space."""
        return self.action_space.sample()


class MarioRightJumpPolicy(BasePolicy):
    """Policy for Super Mario Bros that biases toward moving right, running, and jumping.

    Action probabilities for NES MultiBinary action space:
    - RIGHT (index 7): 90% chance (increased - prioritize moving forward)
    - B/Run (index 0): 70% chance (increased - hold B to run)
    - A/Jump (index 8): 30% chance (decreased - jump less often)
    - Other buttons: 10% chance each

    NES button order: ["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
    """

    def __init__(self, action_space):
        super().__init__(action_space)
        # NES button indices for stable-retro (fceumm core)
        # Button order: ["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
        self.BUTTON_B = 0  # B button (run/fire)
        # Index 1 is unused (null)
        self.BUTTON_SELECT = 2
        self.BUTTON_START = 3
        self.BUTTON_UP = 4
        self.BUTTON_DOWN = 5
        self.BUTTON_LEFT = 6
        self.BUTTON_RIGHT = 7  # Move right
        self.BUTTON_A = 8  # A button (Jump)

    def __call__(self, observation):
        """Return biased action favoring right movement, running, and occasional jumping."""
        import numpy as np

        action = np.zeros(self.action_space.n, dtype=np.int32)

        # Very high probability for RIGHT (90%) - prioritize moving forward
        if np.random.random() < 0.90:
            action[self.BUTTON_RIGHT] = 1

        # High probability for B/Run (70%) - hold B to run fast
        if np.random.random() < 0.70:
            action[self.BUTTON_B] = 1

        # Lower probability for JUMP/A (30%) - jump only occasionally
        if np.random.random() < 0.30:
            action[self.BUTTON_A] = 1

        # Low probability for other buttons (10% each)
        if np.random.random() < 0.10:
            action[self.BUTTON_LEFT] = 1
        if np.random.random() < 0.10:
            action[self.BUTTON_DOWN] = 1
        if np.random.random() < 0.10:
            action[self.BUTTON_UP] = 1

        return action


class BreakoutCatcherPolicy(BasePolicy):
    """Breakout policy that tracks the ball exactly and escapes reward stalls.

    On ALE Breakout environments this reads paddle and ball state directly from
    emulator RAM, which avoids the false detections and missed catches that come
    from RGB-only heuristics. The controller predicts the descending intercept at
    the paddle line, including wall reflections, then steers the paddle toward
    that landing point with a small deadzone to avoid oscillation. Every
    descending contact picks a small random paddle/ball offset, then widens that
    variation if rewards stall long enough to break deterministic loops that
    stop clearing bricks.

    BreakoutNoFrameskip-v4 action space (Discrete):
    - 0: NOOP
    - 1: FIRE
    - 2: RIGHT
    - 3: LEFT
    """

    def __init__(self, action_space, env=None):
        super().__init__(action_space, env=env)
        # ALE Breakout action indices (Discrete(4)):
        # 0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT
        self.NOOP = 0
        self.FIRE = 1
        self.RIGHT = 2
        self.LEFT = 3

        self._prev_ball_pos = None
        self._rng = np.random.default_rng()

        # Breakout RAM addresses (ALE).
        self._paddle_x_addr = 72
        self._ball_x_addr = 99
        self._ball_y_addr = 101

        # Tuned against local ALE runs to prioritize zero-loss tracking over
        # aggressive paddle movement that introduces oscillation near contact.
        self._base_target_offset = -5
        self._deadzone = 4
        self._contact_y = 177
        self._left_wall = 57
        self._right_wall = 200
        self._contact_window = 18

        # Reward-aware loop breaking: always vary the paddle/ball contact point
        # a little, then widen the range if the game has clearly stalled.
        self._stall_steps = 0
        self._stall_threshold = 1200
        self._descent_offset = self._base_target_offset
        self._descent_offset_committed = False
        self._reward_since_last_contact = False

        # RGB fallback for non-ALE environments or direct unit-style invocation.
        self._sprite_color = np.array([200, 72, 72], dtype=np.uint8)
        self._fallback_ball_y_min = 93
        self._fallback_ball_y_max = 189
        self._fallback_paddle_y_min = 189
        self._fallback_paddle_y_max = 205
        self._fallback_contact_y = 183
        self._fallback_left_wall = 8
        self._fallback_right_wall = 151

    def reset(self):
        self._prev_ball_pos = None
        self._stall_steps = 0
        self._descent_offset = self._base_target_offset
        self._descent_offset_committed = False
        self._reward_since_last_contact = False

    def observe_step(self, reward, terminated, truncated, info):
        if reward > 0:
            self._stall_steps = 0
            self._reward_since_last_contact = True
        else:
            self._stall_steps += 1

        if terminated or truncated:
            self.reset()

    def __call__(self, observation):
        """Return an action that keeps the paddle under the live ball."""
        state = self._read_breakout_state(observation)

        if state["ball_x"] is None or state["ball_y"] is None:
            self._prev_ball_pos = None
            self._descent_offset = self._base_target_offset
            self._descent_offset_committed = False
            return self.FIRE

        target_x = state["ball_x"]
        descending = False
        if self._prev_ball_pos is not None:
            dx = state["ball_x"] - self._prev_ball_pos[0]
            dy = state["ball_y"] - self._prev_ball_pos[1]
            descending = dy > 0

            if descending and state["ball_y"] >= state["contact_y"] - self._contact_window:
                if not self._descent_offset_committed:
                    self._descent_offset = self._choose_contact_offset()
                    self._descent_offset_committed = True
            elif dy < 0:
                self._descent_offset = self._base_target_offset
                self._descent_offset_committed = False
                self._reward_since_last_contact = False

            if dy > 0 and state["ball_y"] < state["contact_y"] and dx != 0:
                steps_to_contact = (state["contact_y"] - state["ball_y"]) / dy
                target_x = self._reflect_x(
                    round(state["ball_x"] + (dx * steps_to_contact)),
                    state["left_wall"],
                    state["right_wall"],
                )
        else:
            self._descent_offset = self._base_target_offset

        self._prev_ball_pos = (state["ball_x"], state["ball_y"])
        target_x += self._descent_offset

        if state["paddle_x"] < target_x - self._deadzone:
            return self.RIGHT
        if state["paddle_x"] > target_x + self._deadzone:
            return self.LEFT
        return self.NOOP

    def _choose_contact_offset(self):
        """Pick a safe contact offset, with wider variation after stalls."""
        if self._stall_steps < self._stall_threshold:
            candidates = [-7, -5, -3, 0]
            probabilities = [0.24, 0.46, 0.22, 0.08]
        elif self._stall_steps < self._stall_threshold * 2:
            candidates = [-9, -7, -5, -3, 0, 3]
            probabilities = [0.14, 0.22, 0.28, 0.18, 0.12, 0.06]
        elif self._stall_steps < self._stall_threshold * 4:
            candidates = [-11, -8, -5, -2, 2, 5, 8]
            probabilities = [0.10, 0.16, 0.22, 0.16, 0.14, 0.12, 0.10]
        else:
            candidates = [-12, -9, -6, -3, 0, 3, 6, 9]
            probabilities = [0.09, 0.12, 0.15, 0.16, 0.16, 0.12, 0.10, 0.10]

        # If the last descent still produced no reward, bias slightly away from
        # the previous contact point to avoid repeating the same bounce.
        if not self._reward_since_last_contact:
            candidates = [c for c in candidates if c != self._descent_offset] or candidates
            probabilities = None

        return int(self._rng.choice(candidates, p=probabilities))

    def _read_breakout_state(self, observation):
        ram_state = self._read_ram_state()
        if ram_state is not None:
            return ram_state
        return self._read_rgb_state(observation)

    def _read_ram_state(self):
        ale = getattr(getattr(self.env, "unwrapped", None), "ale", None)
        if ale is None or not hasattr(ale, "getRAM"):
            return None

        ram = ale.getRAM()
        return {
            "ball_x": int(ram[self._ball_x_addr]) or None,
            "ball_y": int(ram[self._ball_y_addr]) or None,
            "paddle_x": int(ram[self._paddle_x_addr]),
            "contact_y": self._contact_y,
            "left_wall": self._left_wall,
            "right_wall": self._right_wall,
        }

    def _read_rgb_state(self, observation):
        obs = np.asarray(observation)

        paddle_region = obs[self._fallback_paddle_y_min : self._fallback_paddle_y_max, :, :]
        paddle_mask = np.all(paddle_region == self._sprite_color, axis=2)
        _, paddle_cols = np.where(paddle_mask)
        paddle_x = float(np.mean(paddle_cols)) if len(paddle_cols) else 80.0

        ball_region = obs[self._fallback_ball_y_min : self._fallback_ball_y_max, :, :]
        ball_mask = np.all(ball_region == self._sprite_color, axis=2)
        ball_rows, ball_cols = np.where(ball_mask)
        ball_x = None
        ball_y = None
        if len(ball_cols):
            unique_rows = sorted(np.unique(ball_rows + self._fallback_ball_y_min), reverse=True)
            for row in unique_rows:
                row_cols = ball_cols[(ball_rows + self._fallback_ball_y_min) == row]
                if 1 <= len(row_cols) <= 6:
                    ball_x = float(np.mean(row_cols))
                    ball_y = float(row)
                    break

        return {
            "ball_x": ball_x,
            "ball_y": ball_y,
            "paddle_x": paddle_x,
            "contact_y": self._fallback_contact_y,
            "left_wall": self._fallback_left_wall,
            "right_wall": self._fallback_right_wall,
        }

    @staticmethod
    def _reflect_x(x, left_wall, right_wall):
        while x < left_wall or x > right_wall:
            if x > right_wall:
                x = right_wall - (x - right_wall)
            elif x < left_wall:
                x = left_wall + (left_wall - x)
        return x


@dataclass(frozen=True)
class HFPolicySource:
    ref: str
    repo_id: str
    revision: str
    checkpoint_filename: str
    model_path: str
    model_document: dict
    recipe_document: dict
    evaluation: dict
    environment: dict
    provider: str
    backend: str
    env_id: str
    state: str | None
    action_set: str
    frame_skip: int
    frame_stack: int
    observation_size: int
    obs_crop: tuple[int, int, int, int] | None
    deterministic: bool = False
    device: str = "auto"

    @property
    def collector(self) -> str:
        return f"hf://{self.repo_id}"

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.model_document["checkpoint"]["sha256"])

    @property
    def recipe_sha256(self) -> str:
        return str(self.model_document["recipe"]["sha256"])

    @property
    def env_make_kwargs(self) -> dict:
        """Compile the resolved evaluation environment into native provider kwargs."""
        environment = self.environment
        kwargs = dict(environment.get("env_args") or {})
        observation_size = int(environment["observation_size"])
        kwargs.update(
            {
                "obs_resize": (observation_size, observation_size),
                "obs_crop": tuple(int(value) for value in environment["obs_crop"]),
                "obs_crop_mode": str(environment["obs_crop_mode"]),
                "obs_crop_fill": int(environment["obs_crop_fill"]),
                "obs_resize_algorithm": str(environment["obs_resize_algorithm"]),
                "frame_skip": int(environment["frame_skip"]),
                "maxpool_last_two": bool(environment["max_pool_frames"]),
                "sticky_action_prob": float(environment["sticky_action_prob"]),
            }
        )
        return kwargs


class StableBaselines3Policy(BasePolicy):
    """Run an SB3 policy against recipe-native provider observations."""

    def __init__(self, action_space, source: HFPolicySource, observation_space=None):
        super().__init__(action_space)
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise SystemExit(
                "stable-baselines3 is required to record Hugging Face policy checkpoints. "
                "Install dependencies with `uv sync` or reinstall the gymrec tool."
            ) from exc

        checkpoint = source.model_document["checkpoint"]
        if checkpoint.get("algorithm_id") != "ppo":
            raise SystemExit(
                "gymrec currently supports only rlab SB3 PPO policy bundles; "
                f"got {checkpoint.get('algorithm_id')!r}"
            )
        self.source = source
        self.model = PPO.load(
            source.model_path,
            device=source.device,
            custom_objects={
                "learning_rate": 0.0,
                "lr_schedule": lambda _progress_remaining: 0.0,
                "clip_range": lambda _progress_remaining: 0.0,
                "clip_range_vf": None,
            },
        )
        self._action_masks = (
            _stable_retro_action_masks(source.env_id, source.action_set)
            if isinstance(action_space, gym.spaces.MultiBinary)
            else ()
        )
        model_observation_shape = getattr(
            getattr(self.model, "observation_space", None), "shape", None
        )
        provider_observation_shape = getattr(observation_space, "shape", None)
        if (
            model_observation_shape is not None
            and provider_observation_shape is not None
            and tuple(model_observation_shape) != tuple(provider_observation_shape)
        ):
            raise SystemExit(
                "Recipe-native provider observation shape does not match the policy: "
                f"provider={tuple(provider_observation_shape)}, "
                f"model={tuple(model_observation_shape)}"
            )
        if self._action_masks:
            model_action_count = getattr(getattr(self.model, "action_space", None), "n", None)
            if model_action_count != len(self._action_masks):
                raise SystemExit(
                    "Recipe action set does not match the policy action space: "
                    f"recipe={len(self._action_masks)}, model={model_action_count}"
                )

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.model.set_random_seed(int(seed))

    def observe_step(self, reward, terminated, truncated, info):
        del reward, terminated, truncated, info

    def __call__(self, observation):
        action, _ = self.model.predict(observation, deterministic=self.source.deterministic)
        return self._copy_action(self._to_env_action(action))

    def _to_env_action(self, action):
        action_array = np.asarray(action)
        action_index = int(action_array.reshape(-1)[0])

        if isinstance(self.action_space, gym.spaces.Discrete):
            return action_index

        if isinstance(self.action_space, gym.spaces.MultiBinary):
            if action_index < 0 or action_index >= len(self._action_masks):
                raise ValueError(
                    f"Policy returned action {action_index}, but action set "
                    f"{self.source.action_set!r} has {len(self._action_masks)} actions"
                )
            native_action = np.zeros(self.action_space.n, dtype=np.int32)
            for button_index in self._action_masks[action_index]:
                if button_index < self.action_space.n:
                    native_action[button_index] = 1
            return native_action

        raise ValueError(
            "Hugging Face SB3 policy recording currently supports Discrete and "
            "MultiBinary action spaces only."
        )

    @staticmethod
    def _copy_action(action):
        if isinstance(action, np.ndarray):
            return action.copy()
        return action


def is_huggingface_model_ref(value):
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        return True
    parsed = urlparse(text)
    return parsed.scheme in {"http", "https"} and parsed.netloc == HUGGINGFACE_MODEL_URL_HOST


def parse_huggingface_model_ref(value):
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        path = text.removeprefix(HUGGINGFACE_MODEL_SCHEME).strip("/")
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"expected Hugging Face model ref like hf://owner/repo, got {value!r}")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:]) or None
        return repo_id, filename, None

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != HUGGINGFACE_MODEL_URL_HOST:
        raise ValueError(f"expected Hugging Face model URL, got {value!r}")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"expected Hugging Face model URL with owner/repo, got {value!r}")
    repo_id = "/".join(parts[:2])
    filename = None
    revision = None
    if len(parts) >= 5 and parts[2] in {"blob", "raw", "resolve"}:
        revision = parts[3]
        filename = "/".join(parts[4:])
    elif len(parts) > 2:
        filename = "/".join(parts[2:])
    return repo_id, filename, revision


def _metadata_value(metadata, path):
    value = metadata
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, (list, tuple)) and isinstance(key, int) and key < len(value):
            value = value[key]
        else:
            return None
    return value


def _metadata_int(metadata, *paths, default):
    for path in paths:
        value = _metadata_value(metadata, path)
        if value is not None:
            return int(value)
    if default is None:
        return None
    return int(default)


def _metadata_float(metadata, *paths, default):
    for path in paths:
        value = _metadata_value(metadata, path)
        if value is not None:
            return float(value)
    if default is None:
        return None
    return float(default)


def _metadata_str(metadata, *paths, default=None):
    for path in paths:
        value = _metadata_value(metadata, path)
        if value:
            return str(value)
    return default


def _stable_retro_action_masks(env_id, action_set):
    action_sets = STABLE_RETRO_DISCRETE_ACTION_SETS.get(env_id, {})
    if action_set not in action_sets:
        valid = ", ".join(sorted(action_sets)) or "none"
        raise SystemExit(
            f"Unsupported action_set {action_set!r} for {env_id}; supported sets: {valid}"
        )
    return tuple(NES_SIMPLE_ACTION_MASKS[name] for name in action_sets[action_set])


def _download_huggingface_policy_file(repo_id, revision, filename):
    try:
        return hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=filename,
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not download {filename} from Hugging Face model repo {repo_id}: {exc}"
        ) from exc


def _load_json_document(path, *, label):
    try:
        with open(path) as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must contain a JSON object")
    return value


def _huggingface_snapshot_revision(path, fallback):
    """Recover the immutable Hub commit from a standard hf_hub_download cache path."""
    snapshot_dir = os.path.dirname(path)
    if os.path.basename(os.path.dirname(snapshot_dir)) == "snapshots":
        return os.path.basename(snapshot_dir)
    return fallback


def _validate_rlab_document(document, *, label, document_type, format_version):
    if document.get("document_type") != document_type:
        raise SystemExit(
            f"Unsupported {label} document_type {document.get('document_type')!r}; "
            f"expected {document_type!r}"
        )
    if document.get("format_version") != format_version:
        raise SystemExit(
            f"Unsupported {label} format_version {document.get('format_version')!r}; "
            f"gymrec supports version {format_version}"
        )


def _validate_bound_file(path, binding, *, label):
    if not isinstance(binding, dict):
        raise SystemExit(f"model.json is missing its {label} binding")
    expected_size = binding.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise SystemExit(f"model.json has an invalid {label} size_bytes")
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise SystemExit(
            f"{label} size mismatch: model.json declares {expected_size}, got {actual_size}"
        )
    expected_sha256 = binding.get("sha256")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 != actual_sha256:
        raise SystemExit(
            f"{label} SHA-256 mismatch: model.json declares {expected_sha256!r}, "
            f"got {actual_sha256!r}"
        )


def _recipe_evaluation_environment(recipe_document, *, repo_id):
    recipe = recipe_document.get("recipe")
    if not isinstance(recipe, dict):
        raise SystemExit(f"{repo_id}/recipe.json is missing recipe")
    if recipe.get("schema_version") != RLAB_RECIPE_SCHEMA_VERSION:
        raise SystemExit(
            f"Unsupported rlab recipe schema_version {recipe.get('schema_version')!r}; "
            f"gymrec supports version {RLAB_RECIPE_SCHEMA_VERSION}"
        )
    evaluation = recipe.get("eval")
    if not isinstance(evaluation, dict):
        raise SystemExit(f"{repo_id}/recipe.json is missing recipe.eval")
    environment = evaluation.get("environment")
    if not isinstance(environment, dict):
        raise SystemExit(f"{repo_id}/recipe.json is missing recipe.eval.environment")
    required = {
        "env_provider",
        "frame_skip",
        "game",
        "max_pool_frames",
        "obs_crop",
        "obs_crop_fill",
        "obs_crop_mode",
        "obs_resize_algorithm",
        "observation_size",
        "sticky_action_prob",
        "task",
    }
    missing = sorted(required - set(environment))
    if missing:
        raise SystemExit(
            f"{repo_id}/recipe.json evaluation environment is missing: {', '.join(missing)}"
        )
    if not isinstance(environment.get("env_args", {}), dict):
        raise SystemExit(f"{repo_id}/recipe.json environment.env_args must be an object")
    obs_crop = environment["obs_crop"]
    if not isinstance(obs_crop, list) or len(obs_crop) != 4:
        raise SystemExit(f"{repo_id}/recipe.json environment.obs_crop must contain four integers")
    action_sampling = evaluation.get("action_sampling")
    if action_sampling not in {"stochastic", "deterministic"}:
        raise SystemExit(
            f"Unsupported recipe.eval.action_sampling {action_sampling!r}; "
            "expected 'stochastic' or 'deterministic'"
        )
    return evaluation, environment


def resolve_huggingface_policy_source(
    ref,
    *,
    filename=None,
    revision=None,
    device="auto",
    deterministic=None,
):
    _lazy_init()
    try:
        repo_id, parsed_filename, parsed_revision = parse_huggingface_model_ref(ref)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    resolved_revision = revision or parsed_revision or "main"
    model_json_path = _download_huggingface_policy_file(
        repo_id, resolved_revision, RLAB_MODEL_FILENAME
    )
    recipe_json_path = _download_huggingface_policy_file(
        repo_id, resolved_revision, RLAB_RECIPE_FILENAME
    )
    model_document = _load_json_document(model_json_path, label=f"{repo_id}/model.json")
    recipe_document = _load_json_document(recipe_json_path, label=f"{repo_id}/recipe.json")
    _validate_rlab_document(
        model_document,
        label="model.json",
        document_type=RLAB_MODEL_DOCUMENT_TYPE,
        format_version=RLAB_MODEL_FORMAT_VERSION,
    )
    _validate_rlab_document(
        recipe_document,
        label="recipe.json",
        document_type=RLAB_RECIPE_DOCUMENT_TYPE,
        format_version=RLAB_RECIPE_FORMAT_VERSION,
    )

    recipe_binding = model_document.get("recipe")
    if not isinstance(recipe_binding, dict):
        raise SystemExit(f"{repo_id}/model.json is missing recipe binding")
    if recipe_binding.get("filename") != RLAB_RECIPE_FILENAME:
        raise SystemExit(
            f"Unsupported recipe filename {recipe_binding.get('filename')!r}; "
            f"expected {RLAB_RECIPE_FILENAME!r}"
        )
    if (
        recipe_binding.get("document_type") != RLAB_RECIPE_DOCUMENT_TYPE
        or recipe_binding.get("format_version") != RLAB_RECIPE_FORMAT_VERSION
    ):
        raise SystemExit(f"{repo_id}/model.json recipe binding does not match recipe.json")
    _validate_bound_file(recipe_json_path, recipe_binding, label="recipe.json")

    checkpoint = model_document.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"{repo_id}/model.json is missing checkpoint")
    checkpoint_filename = checkpoint.get("filename")
    if not isinstance(checkpoint_filename, str) or not checkpoint_filename:
        raise SystemExit(f"{repo_id}/model.json has an invalid checkpoint filename")
    requested_filename = filename or parsed_filename
    if requested_filename and requested_filename not in {
        RLAB_MODEL_FILENAME,
        RLAB_RECIPE_FILENAME,
        checkpoint_filename,
    }:
        raise SystemExit(
            f"Requested checkpoint {requested_filename!r} does not match "
            f"{repo_id}/model.json ({checkpoint_filename!r})"
        )
    model_path = _download_huggingface_policy_file(repo_id, resolved_revision, checkpoint_filename)
    _validate_bound_file(model_path, checkpoint, label=checkpoint_filename)

    policy = model_document.get("policy")
    if not isinstance(policy, dict):
        raise SystemExit(f"{repo_id}/model.json is missing policy")
    for key in ("algorithm_id", "model_class"):
        if policy.get(key) != checkpoint.get(key):
            raise SystemExit(f"{repo_id}/model.json policy.{key} does not match checkpoint.{key}")

    evaluation, environment = _recipe_evaluation_environment(recipe_document, repo_id=repo_id)
    provider = str(environment["env_provider"])
    backend_by_provider = {
        "supermariobrosnes-turbo": "supermariobrosnes-turbo",
        "stable-retro-turbo": "stable-retro-turbo",
    }
    if provider not in backend_by_provider:
        raise SystemExit(
            f"Unsupported rlab evaluation env_provider {provider!r}; supported providers: "
            + ", ".join(sorted(backend_by_provider))
        )
    task = environment["task"]
    if not isinstance(task, dict) or task.get("id") != "mario":
        raise SystemExit(
            f"Unsupported rlab evaluation task {getattr(task, 'get', lambda *_: None)('id')!r}; "
            "gymrec currently supports recipe task 'mario'"
        )
    action = task.get("action")
    if not isinstance(action, dict) or not action.get("set"):
        raise SystemExit(f"{repo_id}/recipe.json task.action.set is required")
    env_args = environment.get("env_args", {})
    frame_stack = int(env_args.get("frame_stack", 1))
    action_sampling = evaluation["action_sampling"]
    resolved_deterministic = (
        action_sampling == "deterministic" if deterministic is None else bool(deterministic)
    )

    return HFPolicySource(
        ref=ref,
        repo_id=repo_id,
        revision=_huggingface_snapshot_revision(model_json_path, resolved_revision),
        checkpoint_filename=checkpoint_filename,
        model_path=model_path,
        model_document=model_document,
        recipe_document=recipe_document,
        evaluation=evaluation,
        environment=environment,
        provider=provider,
        backend=backend_by_provider[provider],
        env_id=str(environment["game"]),
        state=str(environment["state"]) if environment.get("state") is not None else None,
        action_set=str(action["set"]),
        frame_skip=max(int(environment["frame_skip"]), 1),
        frame_stack=max(frame_stack, 1),
        observation_size=int(environment["observation_size"]),
        obs_crop=tuple(int(value) for value in environment["obs_crop"]),
        deterministic=resolved_deterministic,
        device=device,
    )


AGENT_POLICY_FACTORIES = {
    "random": lambda env: RandomPolicy(env.action_space),
    "mario": lambda env: MarioRightJumpPolicy(env.action_space),
    "breakout": lambda env: BreakoutCatcherPolicy(env.action_space, env=env),
}


@dataclass(frozen=True)
class LiveEpisodePackage:
    episode_id: str
    package_dir: str

    @property
    def video_root(self):
        return self.package_dir

    @property
    def frame_dir(self):
        return os.path.join(self.package_dir, "frames")

    @property
    def journal_path(self):
        return os.path.join(self.package_dir, "journal.jsonl")

    @property
    def terminal_candidate_path(self):
        return os.path.join(self.package_dir, "terminal_candidate.webp")


def _upload_live_episode_package(
    recording_identity,
    episode_id,
    package_dir,
    dataset,
    *,
    storage_format,
    max_retries,
    base_wait,
    persist_dataset,
):
    """Upload and transition one live episode package, persisting it when requested."""
    os.makedirs(package_dir, exist_ok=True)
    dataset = _strip_runtime_columns(dataset)
    if persist_dataset:
        dataset.to_parquet(os.path.join(package_dir, "episode.parquet"))
    manifest_kwargs = {
        "package_dir": package_dir,
        "storage_format": storage_format,
        "frames": len(dataset),
    }
    identity = _coerce_recording_identity(recording_identity)
    _set_live_upload_manifest_entry(identity, episode_id, state="pending", **manifest_kwargs)
    success = _upload_dataset_shard_to_hub(
        identity,
        dataset,
        storage_format=storage_format,
        local_root=package_dir,
        episode_ids={episode_id},
        replace=False,
        include_previews=False,
        max_retries=max_retries,
        base_wait=base_wait,
    )
    if success:
        _set_live_upload_manifest_entry(identity, episode_id, state="uploaded", **manifest_kwargs)
        shutil.rmtree(package_dir, ignore_errors=True)
        return True

    _set_live_upload_manifest_entry(
        identity,
        episode_id,
        state="failed",
        error="upload failed",
        **manifest_kwargs,
    )
    return False


class LiveEpisodeUploadManager:
    """Materialize, upload, and track one verified episode at a time."""

    def __init__(self, recording_identity, storage_format, max_retries=5, base_wait=1.0):
        self.identity = _coerce_recording_identity(recording_identity)
        self.env_id = self.identity.display_ref
        self.storage_format = storage_format
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.queue_dir = _get_live_upload_queue_dir(self.identity)

    def begin_episode(self, episode_uuid):
        episode_id = episode_uuid.hex
        package_dir = os.path.join(self.queue_dir, episode_id)
        if os.path.exists(package_dir):
            shutil.rmtree(package_dir)
        os.makedirs(package_dir, exist_ok=True)
        package = LiveEpisodePackage(episode_id=episode_id, package_dir=package_dir)
        _set_live_upload_manifest_entry(
            self.identity,
            package.episode_id,
            state="recording",
            package_dir=package.package_dir,
            storage_format=self.storage_format,
            frames=0,
        )
        return package

    def upload_episode(self, package, dataset):
        return _upload_live_episode_package(
            self.identity,
            package.episode_id,
            package.package_dir,
            dataset,
            storage_format=self.storage_format,
            max_retries=self.max_retries,
            base_wait=self.base_wait,
            persist_dataset=True,
        )


def create_agent_policy(agent_type, env):
    """Create the configured agent policy for an environment."""
    try:
        return AGENT_POLICY_FACTORIES[agent_type](env)
    except KeyError:
        raise ValueError(f"Unknown agent type: {agent_type}") from None


def _recording_dataset_from_dict(data, storage_format):
    """Build a Dataset with the canonical gymrec recording feature casts."""
    dataset = Dataset.from_dict(data)
    if storage_format == STORAGE_FORMAT_IMAGES:
        dataset = dataset.cast_column("observations", HFImage())
    dataset = dataset.cast_column("episode_id", Value("binary"))
    dataset = dataset.cast_column("session_id", Value("binary"))
    for column in (
        "logical_env_id",
        "capture_backend",
        "state_name",
        "rom_sha256",
        "nes_button_order",
        "action_encoding",
        "policy_source_repo",
        "policy_source_revision",
        "policy_recipe_sha256",
        "policy_checkpoint_sha256",
    ):
        if column in dataset.column_names:
            dataset = dataset.cast_column(column, Value("string"))
    if "frame_skip" in dataset.column_names:
        dataset = dataset.cast_column("frame_skip", Value("int64"))
    if "sticky_action_prob" in dataset.column_names:
        dataset = dataset.cast_column("sticky_action_prob", Value("float64"))
    for column in ("policy_recipe_format_version", "policy_model_format_version"):
        if column in dataset.column_names:
            dataset = dataset.cast_column(column, Value("int64"))
    return dataset


def _align_environment_metadata_columns(dataset):
    """Add nullable environment columns so recordings append to pre-backend datasets."""
    feature_types = {
        "logical_env_id": "string",
        "capture_backend": "string",
        "state_name": "string",
        "rom_sha256": "string",
        "nes_button_order": "string",
        "action_encoding": "string",
        "frame_skip": "int64",
        "sticky_action_prob": "float64",
        "policy_source_repo": "string",
        "policy_source_revision": "string",
        "policy_recipe_sha256": "string",
        "policy_checkpoint_sha256": "string",
        "policy_recipe_format_version": "int64",
        "policy_model_format_version": "int64",
    }
    for column, dtype in feature_types.items():
        if column not in dataset.column_names:
            dataset = dataset.add_column(column, [None] * len(dataset))
        dataset = dataset.cast_column(column, Value(dtype))
    return dataset


class DatasetRecorderWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for recording and replaying Atari gameplay as Hugging Face datasets.
    """

    def __init__(
        self,
        env,
        input_source=None,
        headless=False,
        collector="human",
        storage_format=None,
        live_upload_manager=None,
        initial_seed=None,
    ):
        _lazy_init()
        super().__init__(env)

        self.recording = False
        self.storage_format = _normalize_storage_format(
            storage_format or CONFIG["storage"].get("format", STORAGE_FORMAT_IMAGES)
        )
        self._ffmpeg_path = None
        if self.storage_format == STORAGE_FORMAT_LOSSLESS_VIDEO:
            self._ffmpeg_path, _ = _require_lossless_video_tools()
        self.frame_shape = None  # Delay initialization
        self.screen = None  # Delay initialization
        self.headless = headless
        self.input_source = input_source
        self.collector = collector
        self.live_upload_manager = live_upload_manager
        self.initial_seed = int(initial_seed) if initial_seed is not None else None
        self._gymrec_version = _get_gymrec_version()

        if not headless:
            pygame.init()
            # pygame.display.set_caption will be set after env_id is available

        self.current_keys = set()
        self.key_lock = threading.Lock()

        self.episode_ids = []
        self.seeds = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.terminations = []
        self.truncations = []
        self.infos = []
        self.session_ids = []
        self.video_paths = []
        self.frame_indices = []
        self.frame_sha256s = []
        self.frame_widths = []
        self.frame_heights = []
        self.episode_num_observations = []
        self._current_episode_uuid = None
        self._current_episode_seed = None
        self._session_uuid = None

        self.temp_dir = tempfile.mkdtemp()
        self.video_artifact_dir = os.path.join(self.temp_dir, VIDEO_ARTIFACT_DIR)
        self._current_episode_video_frames = []
        self._current_episode_video_hashes = []
        self._live_episode = None
        self._live_video_writer = None

        self._fps = None
        self._fps_changed_at = 0
        self._episode_count = 0
        self._cumulative_reward = 0.0
        self._overlay_visible = True
        self._playback_frame_index = None
        self._playback_total = None
        self._recorded_dataset = None
        self._env_metadata = None
        self._max_episodes = None  # None means unlimited

    @property
    def _live_upload_enabled(self):
        return self.live_upload_manager is not None

    def _clear_recording_buffers(self):
        self.episode_ids.clear()
        self.seeds.clear()
        self.frames.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.infos.clear()
        self.session_ids.clear()
        self.video_paths.clear()
        self.frame_indices.clear()
        self.frame_sha256s.clear()
        self.frame_widths.clear()
        self.frame_heights.clear()
        self.episode_num_observations.clear()
        self._current_episode_video_frames.clear()
        self._current_episode_video_hashes.clear()

    def _build_recorded_dataset(self):
        env_metadata = getattr(self, "_env_metadata", None) or {}
        row_count = len(self.frames)
        data = {
            "episode_id": self.episode_ids,
            "seed": self.seeds,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminations": self.terminations,
            "truncations": self.truncations,
            "infos": self.infos,
            "session_id": self.session_ids,
            "collector": [self.collector] * len(self.frames),
            "gymrec_version": [self._gymrec_version] * len(self.frames),
            "storage_format": [self.storage_format] * len(self.frames),
            "logical_env_id": [env_metadata.get("env_id")] * row_count,
            "capture_backend": [env_metadata.get("capture_backend")] * row_count,
            "state_name": [env_metadata.get("state_name")] * row_count,
            "rom_sha256": [env_metadata.get("rom_sha256")] * row_count,
            "nes_button_order": [
                json.dumps(env_metadata.get("nes_button_order"))
                if env_metadata.get("nes_button_order") is not None
                else None
            ]
            * row_count,
            "action_encoding": [env_metadata.get("action_encoding")] * row_count,
            "frame_skip": [env_metadata.get("frameskip")] * row_count,
            "sticky_action_prob": [env_metadata.get("sticky_actions")] * row_count,
            "policy_source_repo": [env_metadata.get("policy_source_repo")] * row_count,
            "policy_source_revision": [env_metadata.get("policy_source_revision")] * row_count,
            "policy_recipe_sha256": [env_metadata.get("policy_recipe_sha256")] * row_count,
            "policy_checkpoint_sha256": [env_metadata.get("policy_checkpoint_sha256")] * row_count,
            "policy_recipe_format_version": [env_metadata.get("policy_recipe_format_version")]
            * row_count,
            "policy_model_format_version": [env_metadata.get("policy_model_format_version")]
            * row_count,
        }
        if self.storage_format == STORAGE_FORMAT_IMAGES:
            data["observations"] = self.frames
        else:
            data.update(
                {
                    "video_path": self.video_paths,
                    "frame_index": self.frame_indices,
                    "frame_sha256": self.frame_sha256s,
                    "frame_width": self.frame_widths,
                    "frame_height": self.frame_heights,
                    "episode_num_observations": self.episode_num_observations,
                }
            )
        return _recording_dataset_from_dict(data, self.storage_format)

    def _start_live_episode(self, episode_uuid):
        if not self._live_upload_enabled:
            return
        self._live_episode = self.live_upload_manager.begin_episode(episode_uuid)
        self._live_video_writer = None

    def _relative_live_package_path(self, path):
        if not self._live_episode:
            return path
        return os.path.relpath(path, self._live_episode.package_dir)

    def _write_jsonl_record(self, path, record):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=_json_default))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    def _write_live_step_journal(self, next_observation):
        if not self._live_upload_enabled or self._live_episode is None or not self.actions:
            return
        row_index = len(self.actions) - 1
        if row_index < 0 or row_index >= len(self.rewards):
            return

        terminal_frame = _observation_to_rgb_array(
            _recording_observation(self.env, next_observation)
        )
        PILImage.fromarray(terminal_frame).save(
            self._live_episode.terminal_candidate_path,
            format="WEBP",
            lossless=True,
            method=6,
        )
        record = {
            "type": "step",
            "row_index": row_index,
            "episode_id": self._current_episode_uuid.hex,
            "seed": self.seeds[row_index],
            "action": self.actions[row_index],
            "reward": self.rewards[row_index],
            "termination": self.terminations[row_index],
            "truncation": self.truncations[row_index],
            "info": self.infos[row_index],
            "session_id": self._session_uuid.hex,
            "collector": self.collector,
            "gymrec_version": self._gymrec_version,
            "storage_format": self.storage_format,
            "logical_env_id": (self._env_metadata or {}).get("env_id"),
            "capture_backend": (self._env_metadata or {}).get("capture_backend"),
            "state_name": (self._env_metadata or {}).get("state_name"),
            "rom_sha256": (self._env_metadata or {}).get("rom_sha256"),
            "nes_button_order": (
                json.dumps((self._env_metadata or {}).get("nes_button_order"))
                if (self._env_metadata or {}).get("nes_button_order") is not None
                else None
            ),
            "action_encoding": (self._env_metadata or {}).get("action_encoding"),
            "frame_skip": (self._env_metadata or {}).get("frameskip"),
            "sticky_action_prob": (self._env_metadata or {}).get("sticky_actions"),
            "policy_source_repo": (self._env_metadata or {}).get("policy_source_repo"),
            "policy_source_revision": (self._env_metadata or {}).get("policy_source_revision"),
            "policy_recipe_sha256": (self._env_metadata or {}).get("policy_recipe_sha256"),
            "policy_checkpoint_sha256": (self._env_metadata or {}).get("policy_checkpoint_sha256"),
            "policy_recipe_format_version": (self._env_metadata or {}).get(
                "policy_recipe_format_version"
            ),
            "policy_model_format_version": (self._env_metadata or {}).get(
                "policy_model_format_version"
            ),
            "fps": self._fps or get_default_fps(self.env),
            "terminal_candidate_path": self._relative_live_package_path(
                self._live_episode.terminal_candidate_path
            ),
            "terminal_candidate_sha256": _sha256_rgb(terminal_frame),
            "terminal_candidate_width": int(terminal_frame.shape[1]),
            "terminal_candidate_height": int(terminal_frame.shape[0]),
        }
        if self.storage_format == STORAGE_FORMAT_IMAGES:
            record["observation_path"] = self._relative_live_package_path(self.frames[row_index])
        else:
            record.update(
                {
                    "video_path": self.video_paths[row_index],
                    "frame_index": self.frame_indices[row_index],
                    "frame_sha256": self.frame_sha256s[row_index],
                    "frame_width": self.frame_widths[row_index],
                    "frame_height": self.frame_heights[row_index],
                }
            )
        self._write_jsonl_record(self._live_episode.journal_path, record)

    def _ensure_screen(self, frame):
        """
        Ensure pygame screen is initialized with the correct shape.
        """
        frame = _extract_observation_image(frame)
        if self.screen is None or self.frame_shape is None:
            self.frame_shape = frame.shape
            scale = CONFIG["display"]["scale_factor"]
            self.screen = pygame.display.set_mode(
                (self.frame_shape[1] * scale, self.frame_shape[0] * scale)
            )
            pygame.display.set_caption(getattr(self.env, "_env_id", "Gymnasium Recorder"))

    def _save_frame_image(self, frame):
        """Save a frame as lossless WebP and return the file path."""
        frame_uint8 = _observation_to_rgb_array(frame)
        base_dir = self.temp_dir
        if self._live_upload_enabled and self._live_episode is not None:
            base_dir = self._live_episode.frame_dir
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"frame_{len(self.frames):05d}.webp")
        img = PILImage.fromarray(frame_uint8)
        img.save(path, format="WEBP", lossless=True, method=6)
        return path

    def _record_observation(self, episode_uuid, frame):
        """Record an observation through the configured storage backend."""
        if self.storage_format == STORAGE_FORMAT_IMAGES:
            self.frames.append(self._save_frame_image(frame))
            return

        frame_array = _observation_to_rgb_array(frame)
        frame_index = len(self._current_episode_video_hashes)
        frame_hash = _sha256_rgb(frame_array)
        video_relpath = f"{VIDEO_ARTIFACT_DIR}/{episode_uuid.hex}{CANONICAL_VIDEO_SUFFIX}"

        if self._live_upload_enabled:
            if self._live_episode is None:
                self._start_live_episode(episode_uuid)
            video_path = os.path.join(self._live_episode.video_root, video_relpath)
            if self._live_video_writer is None:
                self._live_video_writer = _StreamingLosslessVideoWriter(
                    video_path,
                    self._fps or get_default_fps(self.env),
                    ffmpeg_path=self._ffmpeg_path,
                )
            self._live_video_writer.write(frame_array)
        else:
            self._current_episode_video_frames.append(frame_array.copy())
        self._current_episode_video_hashes.append(frame_hash)
        self.frames.append(None)
        self.video_paths.append(video_relpath)
        self.frame_indices.append(frame_index)
        self.frame_sha256s.append(frame_hash)
        self.frame_heights.append(int(frame_array.shape[0]))
        self.frame_widths.append(int(frame_array.shape[1]))
        self.episode_num_observations.append(None)

    def _finalize_video_episode(self, episode_uuid):
        """Encode and verify the current episode's canonical observation video."""
        if self.storage_format != STORAGE_FORMAT_LOSSLESS_VIDEO:
            return
        if not self._current_episode_video_hashes:
            return

        video_relpath = f"{VIDEO_ARTIFACT_DIR}/{episode_uuid.hex}{CANONICAL_VIDEO_SUFFIX}"
        expected_hashes = list(self._current_episode_video_hashes)
        frame_count = len(expected_hashes)

        if self._live_upload_enabled:
            if self._live_video_writer is None:
                raise RuntimeError("Live video episode has no active ffmpeg writer")
            video_path = self._live_video_writer.output_path
            width = self.frame_widths[-1]
            height = self.frame_heights[-1]
            self._live_video_writer.close()
            self._live_video_writer = None
            _verify_lossless_rgb_video_stream(
                video_path,
                width,
                height,
                expected_hashes,
                ffmpeg_path=self._ffmpeg_path,
            )
            start_index = len(self.episode_num_observations) - frame_count
            for row_index in range(start_index, len(self.episode_num_observations)):
                self.episode_num_observations[row_index] = frame_count
            self._current_episode_video_hashes.clear()
            return

        video_path = os.path.join(self.temp_dir, video_relpath)
        preview_relpath = f"{PREVIEW_VIDEO_ARTIFACT_DIR}/{episode_uuid.hex}{PREVIEW_VIDEO_SUFFIX}"
        preview_path = os.path.join(self.temp_dir, preview_relpath)
        _encode_lossless_rgb_video(
            self._current_episode_video_frames,
            video_path,
            self._fps or get_default_fps(self.env),
            ffmpeg_path=self._ffmpeg_path,
        )
        height, width = self._current_episode_video_frames[0].shape[:2]
        _verify_lossless_rgb_video_stream(
            video_path,
            width,
            height,
            expected_hashes,
            ffmpeg_path=self._ffmpeg_path,
        )

        start_index = len(self.episode_num_observations) - frame_count
        for row_index in range(start_index, len(self.episode_num_observations)):
            self.episode_num_observations[row_index] = frame_count
        _encode_browser_preview_video(
            self._current_episode_video_frames,
            preview_path,
            self._fps or get_default_fps(self.env),
            ffmpeg_path=self._ffmpeg_path,
        )
        self._current_episode_video_frames.clear()
        self._current_episode_video_hashes.clear()

    @staticmethod
    def _normalize_action(action):
        """Normalize action format for dataset storage."""
        if isinstance(action, np.ndarray):
            return action.tolist()
        elif isinstance(action, dict):
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in action.items()}
        else:
            return [int(action)]

    def _record_frame(self, episode_uuid, frame, action):
        """Save a frame and action to temporary storage."""
        if not self.recording:
            return

        self.episode_ids.append(episode_uuid.bytes)
        self.seeds.append(self._current_episode_seed)
        self._record_observation(episode_uuid, frame)
        self.actions.append(self._normalize_action(action))
        self.session_ids.append(self._session_uuid.bytes)

    def _record_terminal_observation(self, episode_uuid, frame):
        """Record a terminal observation row (Minari N+1 pattern).

        The N+1 observation captures the final state after the last step.
        It has an empty action and null values for reward/termination/truncation/info
        since no step was taken.
        """
        if not self.recording:
            return

        self.episode_ids.append(episode_uuid.bytes)
        self.seeds.append(self._current_episode_seed)
        self._record_observation(episode_uuid, frame)
        self.actions.append([])
        self.rewards.append(None)
        self.terminations.append(None)
        self.truncations.append(None)
        self.infos.append(None)
        self.session_ids.append(self._session_uuid.bytes)
        self._finalize_video_episode(episode_uuid)
        self._finish_live_episode()

    def _finish_live_episode(self):
        if not self._live_upload_enabled or not self.frames:
            return
        package = self._live_episode
        if package is None:
            raise RuntimeError("Live upload episode was not initialized")
        dataset = self._build_recorded_dataset()
        success = self.live_upload_manager.upload_episode(package, dataset)
        if not success:
            console.print(
                f"[{STYLE_INFO}]Episode {package.episode_id} kept for retry: "
                f"[{STYLE_CMD}]{_gymrec_cmd('upload', self.live_upload_manager.env_id)}[/]"
            )
        self._clear_recording_buffers()
        self._live_episode = None

    def _input_loop(self):
        """
        Handle pygame input events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_TAB:
                    self._overlay_visible = not self._overlay_visible
                    continue
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    if self._fps is not None:
                        self._fps = max(1, self._fps + 5)
                        self._fps_changed_at = time.monotonic()
                    continue
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    if self._fps is not None:
                        self._fps = max(1, self._fps - 5)
                        self._fps_changed_at = time.monotonic()
                    continue
                with self.key_lock:
                    self.current_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                with self.key_lock:
                    self.current_keys.discard(event.key)
        return True

    def _render_frame(self, frame):
        """
        Render a frame using pygame, scaled by the configured scale factor.
        Skip rendering in headless mode.
        """
        if self.headless:
            return

        frame = _extract_observation_image(frame)
        self._ensure_screen(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))

        scale = CONFIG["display"]["scale_factor"]
        w, h = surface.get_size()
        scaled_surface = pygame.transform.scale(surface, (w * scale, h * scale))

        # Update display with scaled frame
        self.screen.blit(scaled_surface, (0, 0))
        if self._overlay_visible:
            self._render_fps_overlay()
            self._render_episode_overlay()
        pygame.display.flip()

    def _render_fps_overlay(self):
        """Render a temporary FPS indicator in the top-right corner."""
        if self._fps is None or self.screen is None:
            return
        elapsed = time.monotonic() - self._fps_changed_at
        if elapsed >= 1.5:
            return

        # Compute alpha: full opacity for first 1.0s, fade over last 0.5s
        if elapsed < 1.0:
            alpha = 255
        else:
            alpha = int(255 * (1.5 - elapsed) / 0.5)

        font = pygame.font.Font(None, 24)
        text = font.render(f"{self._fps} FPS", True, (255, 255, 255))
        text_rect = text.get_rect()

        padding = 6
        bg_w = text_rect.width + padding * 2
        bg_h = text_rect.height + padding * 2
        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, min(alpha, 180)))

        text_alpha = pygame.Surface(text_rect.size, pygame.SRCALPHA)
        text_alpha.blit(text, (0, 0))
        text_alpha.set_alpha(alpha)

        screen_w = self.screen.get_width()
        bg_x = screen_w - bg_w - 8
        bg_y = 8
        self.screen.blit(bg, (bg_x, bg_y))
        self.screen.blit(text_alpha, (bg_x + padding, bg_y + padding))

    def _render_episode_overlay(self):
        """Render a persistent HUD badge in the top-left corner."""
        if self.screen is None:
            return

        parts = []
        if self.recording and self._episode_count >= 1:
            parts.extend([f"EP {self._episode_count}", f"R {self._cumulative_reward:.0f}"])

        if (
            self._playback_frame_index is not None
            and self._playback_total is not None
            and self._playback_total > 0
        ):
            parts.append(f"F {self._playback_frame_index}/{self._playback_total}")

        if self._fps is not None:
            parts.append(f"{self._fps} FPS")
        if not parts:
            return

        font = pygame.font.Font(None, 24)
        text = font.render("  ".join(parts), True, (255, 255, 255))
        text_rect = text.get_rect()

        padding = 6
        bg_w = text_rect.width + padding * 2
        bg_h = text_rect.height + padding * 2
        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))

        bg_x = 8
        bg_y = 8
        self.screen.blit(bg, (bg_x, bg_y))
        self.screen.blit(text, (bg_x + padding, bg_y + padding))

    def _print_keymappings(self):
        """Print the current key mappings to the console."""
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Key", justify="right", style=STYLE_KEY)
        table.add_column("Action", style=STYLE_ACTION)
        table.add_column("Index", style=STYLE_PATH)
        input_source = (
            self.input_source
            if isinstance(self.input_source, HumanInputSource)
            else HumanInputSource(self.env, self.key_lock, self.current_keys)
        )

        if hasattr(self.env, "_vizdoom") and self.env._vizdoom:
            env_type = "VizDoom"
            vizdoom_buttons = input_source._init_vizdoom_key_mapping()
            for key, action_name in VIZDOOM_KEY_BINDINGS.items():
                btn_idx = vizdoom_buttons.get(action_name)
                idx_str = f"btn {btn_idx}" if btn_idx is not None else ""
                table.add_row(pygame.key.name(key), action_name, idx_str)
            ml_idx = vizdoom_buttons.get("MOVE_LEFT")
            mr_idx = vizdoom_buttons.get("MOVE_RIGHT")
            table.add_row("alt+left", "MOVE_LEFT", f"btn {ml_idx}" if ml_idx is not None else "")
            table.add_row("alt+right", "MOVE_RIGHT", f"btn {mr_idx}" if mr_idx is not None else "")
        elif bool(getattr(self.env, "_stable_retro", False)) or _is_nes_controller_env(self.env):
            platform = getattr(self.env.unwrapped, "system", None)
            backend = getattr(self.env, "_gymrec_backend", "stable-retro")
            env_type = f"{BACKEND_LABELS.get(backend, backend)} ({platform})"
            buttons = getattr(self.env.unwrapped, "buttons", None)
            mapping = STABLE_RETRO_KEY_BINDINGS.get(platform, {})
            # Group keys: D-pad first, then action buttons, then special (SELECT/START)
            dpad_keys = {pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT}
            special_labels = {"SELECT", "START"}
            group_dpad = []
            group_action = []
            group_special = []
            for key, idx in mapping.items():
                label = buttons[idx] if buttons and idx < len(buttons) else f"button {idx}"
                row = (pygame.key.name(key), label, f"idx {idx}")
                if key in dpad_keys:
                    group_dpad.append(row)
                elif label.upper() in special_labels:
                    group_special.append(row)
                else:
                    group_action.append(row)
            for row in group_dpad:
                table.add_row(*row)
            if group_dpad and group_action:
                table.add_section()
            for row in group_action:
                table.add_row(*row)
            if group_action and group_special:
                table.add_section()
            for row in group_special:
                table.add_row(*row)
        else:
            env_type = "Atari"
            if input_source.key_to_action is None:
                input_source._resolve_atari_key_mapping()
            try:
                meanings = self.env.unwrapped.get_action_meanings()
            except (AttributeError, TypeError):
                meanings = None
            for key, action_idx in input_source.key_to_action.items():
                label = (
                    meanings[action_idx]
                    if meanings and action_idx < len(meanings)
                    else f"action {action_idx}"
                )
                table.add_row(pygame.key.name(key), label, f"action {action_idx}")

        table.add_section()
        table.add_row("[dim]escape[/]", "[dim]Exit[/]", "")
        table.add_row("[dim]+/-[/]", "[dim]Adjust FPS (±5)[/]", "")

        console.print(
            Panel(
                table,
                title=f"[{STYLE_ENV}]{env_type}[/] Key Mappings",
                border_style=STYLE_INFO,
                expand=False,
            )
        )

    def _wait_for_start(self, start_key: int = None) -> bool:
        """Display overlay prompting the user to start.

        Returns True if the start key was pressed, False if the user closed the
        window or pressed ESC.
        """
        if start_key is None:
            start_key = START_KEY
        if self.screen is None:
            return True

        overlay_cfg = CONFIG["overlay"]
        font = pygame.font.Font(None, overlay_cfg["font_size"])
        text = font.render(overlay_cfg["text"], True, tuple(overlay_cfg["text_color"]))
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill(tuple(overlay_cfg["background_color"]))
        text_rect = text.get_rect(
            center=(self.screen.get_width() // 2, self.screen.get_height() // 2)
        )

        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == start_key:
                        return True
            # Redraw overlay each frame
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text, text_rect)
            pygame.display.flip()
            clock.tick(overlay_cfg["fps"])

    async def record(
        self,
        fps=None,
        max_episodes=None,
        max_steps=None,
        progress_callback=None,
        step_callback=None,
    ):
        """Record a gameplay session at the desired FPS.

        Args:
            fps: Frames per second for rendering (ignored if headless)
            max_episodes: Maximum number of episodes to record (None = unlimited for human, 1 for agent)
            max_steps: Maximum steps per episode (truncates episode after N steps)
            progress_callback: Optional callable(episode_number, steps_in_episode) called after each episode
            step_callback: Optional callable(episode_number, step_number) called during each step for live updates
        """
        if fps is None:
            fps = get_default_fps(self.env)
        self._max_episodes = max_episodes
        self._max_steps_per_episode = max_steps
        self._progress_callback = progress_callback
        self._step_callback = step_callback
        self.recording = True
        try:
            await self._play(fps)
            return self._recorded_dataset
        finally:
            self.recording = False
            if self._live_video_writer is not None:
                self._live_video_writer.abort()
                self._live_video_writer = None
            # Don't delete temp_dir here - dataset.save_to_disk() needs the image files
            # temp_dir cleanup happens in main() after save_dataset_locally()
            if not self.headless:
                pygame.quit()
            self.env.close()

    async def _play(self, fps=None):
        """
        Main loop for interactive gameplay and recording.
        Supports both human input (pygame) and agent input via InputSource abstraction.
        """
        if fps is None:
            fps = get_default_fps(self.env)
        self._fps = fps

        self._clear_recording_buffers()

        # Capture environment metadata for dataset card
        self._env_metadata = _capture_env_metadata(self.env)

        self._session_uuid = uuid.uuid4()
        self._current_episode_uuid = uuid.uuid4()
        self._current_episode_seed = (
            self.initial_seed if self.initial_seed is not None else int(time.time())
        )
        seed = self._current_episode_seed
        self._episode_count = 1
        self._cumulative_reward = 0.0
        self._start_live_episode(self._current_episode_uuid)
        obs, _ = self.env.reset(seed=seed)

        # Setup input source
        if self.input_source is None:
            # Default to human input
            self.input_source = HumanInputSource(self.env, self.key_lock, self.current_keys)
        elif hasattr(self.input_source, "reset"):
            self.input_source.reset(seed=seed, observation=obs)

        # For human input, show the start screen and keymappings
        if isinstance(self.input_source, HumanInputSource):
            self._ensure_screen(obs)
            self._render_frame(obs)
            self._print_keymappings()
            if not self._wait_for_start():
                return
            with self.key_lock:
                self.current_keys.clear()

        step = 0
        while True:
            frame_start = time.monotonic()

            # Use the wrapper event loop whenever a window is present so
            # shared controls like ESC, +/- FPS, and TAB overlay toggles work
            # consistently in both gameplay and playback.
            if not self.headless and not self._input_loop():
                break

            # Get action from input source
            action = self.input_source.get_action(obs)

            self._record_frame(
                self._current_episode_uuid,
                _recording_observation(self.env, obs),
                action,
            )
            obs, reward, terminated, truncated, info = self.env.step(action)
            if hasattr(self.input_source, "observe_step"):
                self.input_source.observe_step(reward, terminated, truncated, info)
            self._cumulative_reward += float(reward)
            if self.recording:
                self.rewards.append(float(reward))
                self.terminations.append(bool(terminated))
                self.truncations.append(bool(truncated))
                self.infos.append(json.dumps(info, default=_json_default))

            self._render_frame(_recording_observation(self.env, obs))

            # Frame pacing: skip if headless (run at max speed)
            if not self.headless:
                elapsed = time.monotonic() - frame_start
                remaining = (1.0 / self._fps) - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                else:
                    await asyncio.sleep(0)
            else:
                # In headless mode, yield control briefly to not block
                await asyncio.sleep(0)

            step += 1

            # Call step callback for live progress updates
            if self._step_callback is not None:
                self._step_callback(self._episode_count, step)

            # Truncate episode if max_steps reached
            if (
                self._max_steps_per_episode is not None
                and step >= self._max_steps_per_episode
                and not terminated
                and not truncated
            ):
                truncated = True
                if self.recording and self.truncations:
                    self.truncations[-1] = True

            self._write_live_step_journal(obs)

            if terminated or truncated:
                self._record_terminal_observation(
                    self._current_episode_uuid,
                    _recording_observation(self.env, obs),
                )

                if self._progress_callback is not None:
                    self._progress_callback(self._episode_count, step)

                # Check if we've reached the max episode count
                if self._max_episodes is not None and self._episode_count >= self._max_episodes:
                    break

                self._current_episode_uuid = uuid.uuid4()
                self._current_episode_seed = (
                    self.initial_seed + self._episode_count
                    if self.initial_seed is not None
                    else int(time.time())
                )
                seed = self._current_episode_seed
                self._start_live_episode(self._current_episode_uuid)
                obs, _ = self.env.reset(seed=seed)
                if hasattr(self.input_source, "reset"):
                    self.input_source.reset(seed=seed, observation=obs)
                self._episode_count += 1
                self._cumulative_reward = 0.0
                step = 0
                self._render_frame(_recording_observation(self.env, obs))

        # Record terminal observation on user exit (ESC) or when max_episodes reached
        if self.recording and self.frames and self.actions[-1] != []:
            # Mark last real step as truncated: user exited mid-episode.
            # Minari requires at least one True in terminations or truncations per episode.
            self.truncations[-1] = True
            self._record_terminal_observation(
                self._current_episode_uuid,
                _recording_observation(self.env, obs),
            )

        if self.recording and self.frames:
            self._recorded_dataset = self._build_recorded_dataset()

    def _convert_action(self, action):
        """Convert stored action back to the environment's expected format."""
        if isinstance(action, list):
            if isinstance(self.env.action_space, gym.spaces.Discrete) and len(action) == 1:
                return action[0]
            else:
                return np.array(action, dtype=np.int32)
        elif isinstance(action, dict):
            new_action = {}
            space = self.env.action_space
            if isinstance(space, gym.spaces.Dict):
                for k, v in action.items():
                    sub = space[k]
                    if isinstance(v, list):
                        new_action[k] = np.array(v, dtype=sub.dtype)
                    else:
                        new_action[k] = v
            else:
                for k, v in action.items():
                    new_action[k] = np.array(v) if isinstance(v, list) else v
            return new_action
        return action

    async def replay(self, actions=None, fps=None, total=None, verify=False, episodes=None):
        if fps is None:
            fps = get_default_fps(self.env)
        self._fps = fps
        self._playback_frame_index = 0
        self._playback_total = total

        mse_threshold = 5.0
        verify_metrics = [] if verify else None
        reward_mismatches = 0
        terminal_mismatches = 0
        if episodes is None:
            episodes = [{"seed": None, "items": actions or []}]
        printed_keymappings = False
        frame_number = 0
        stop_replay = False

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task("Replaying", total=total)
            try:
                for episode in episodes:
                    if stop_replay:
                        break
                    seed = episode.get("seed")
                    items = episode.get("items", [])
                    reset_kwargs = {} if seed is None else {"seed": int(seed)}
                    obs, _ = self.env.reset(**reset_kwargs)
                    self._ensure_screen(obs)
                    self._render_frame(obs)
                    if not printed_keymappings:
                        self._print_keymappings()
                        printed_keymappings = True

                    for item in items:
                        frame_start = time.monotonic()
                        if not self._input_loop():
                            stop_replay = True
                            break

                        frame_number += 1
                        self._playback_frame_index = frame_number

                        if verify:
                            (
                                action,
                                recorded_image,
                                recorded_reward,
                                recorded_terminated,
                                recorded_truncated,
                            ) = item
                        else:
                            action = item

                        action = self._convert_action(action)
                        obs, reward, terminated, truncated, _ = self.env.step(action)
                        self._render_frame(obs)

                        if verify:
                            obs_image = _extract_observation_image(obs)
                            if obs_image.dtype != np.uint8:
                                obs_image = obs_image.astype(np.uint8)
                            recorded_array = np.array(recorded_image, dtype=np.uint8)

                            if obs_image.shape == recorded_array.shape:
                                mse = float(
                                    np.mean(
                                        (
                                            obs_image.astype(np.float32)
                                            - recorded_array.astype(np.float32)
                                        )
                                        ** 2
                                    )
                                )
                            else:
                                console.print(
                                    f"  [yellow]Warning:[/] shape mismatch at frame {len(verify_metrics)}: "
                                    f"obs={obs_image.shape} vs recorded={recorded_array.shape}, skipping comparison"
                                )
                                mse = None

                            if float(reward) != float(recorded_reward):
                                reward_mismatches += 1
                            if bool(terminated) != bool(recorded_terminated) or bool(
                                truncated
                            ) != bool(recorded_truncated):
                                terminal_mismatches += 1

                            verify_metrics.append(mse)

                        elapsed = time.monotonic() - frame_start
                        remaining = (1.0 / self._fps) - elapsed
                        if remaining > 0:
                            await asyncio.sleep(remaining)
                        else:
                            await asyncio.sleep(0)
                        progress.advance(ptask)
            finally:
                self._playback_frame_index = None
                self._playback_total = None

        if verify and verify_metrics:
            valid_mses = [m for m in verify_metrics if m is not None]
            n_total = len(verify_metrics)
            n_skipped = n_total - len(valid_mses)

            lines = [f"Total frames: [{STYLE_INFO}]{n_total}[/]"]
            if n_skipped > 0:
                lines.append(f"Skipped (shape mismatch): [yellow]{n_skipped}[/]")

            if valid_mses:
                mean_mse = sum(valid_mses) / len(valid_mses)
                max_mse = max(valid_mses)
                min_mse = min(valid_mses)
                exceeded = sum(1 for m in valid_mses if m > mse_threshold)
                lines.append(
                    f"Frame MSE:  mean=[{STYLE_INFO}]{mean_mse:.2f}[/], max=[{STYLE_INFO}]{max_mse:.2f}[/], min=[{STYLE_INFO}]{min_mse:.2f}[/]"
                )
                lines.append(
                    f"Reward mismatches: [{STYLE_INFO}]{reward_mismatches}/{n_total}[/] frames"
                )
                lines.append(
                    f"Terminal state mismatches: [{STYLE_INFO}]{terminal_mismatches}/{n_total}[/] frames"
                )

                passed = exceeded == 0 and reward_mismatches == 0 and terminal_mismatches == 0
                if passed:
                    lines.append(
                        f"Result: [{STYLE_SUCCESS}]PASS[/] (all frames below threshold {mse_threshold})"
                    )
                    border_style = "green"
                else:
                    reasons = []
                    if exceeded > 0:
                        reasons.append(f"{exceeded} frames exceeded MSE threshold {mse_threshold}")
                    if reward_mismatches > 0:
                        reasons.append(f"{reward_mismatches} reward mismatches")
                    if terminal_mismatches > 0:
                        reasons.append(f"{terminal_mismatches} terminal state mismatches")
                    lines.append(f"Result: [{STYLE_FAIL}]FAIL[/] ({', '.join(reasons)})")
                    border_style = "red"
            else:
                lines.append("[yellow]No valid frame comparisons (all skipped).[/]")
                border_style = "yellow"

            console.print()
            console.print(
                Panel(
                    "\n".join(lines),
                    title="Determinism Verification Report",
                    border_style=border_style,
                    expand=False,
                )
            )

    def close(self):
        """
        Clean up resources and save dataset if needed.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        if not self.headless:
            pygame.quit()
        super().close()


def _encode_env_id_for_hf(env_id):
    """
    Encode env_id to a reversible string for HF dataset naming.

    Encoding scheme:
    - '/' -> '_slash_'
    - '-' -> '_dash_'
    - '_' -> '_underscore_'

    This allows perfect round-trip conversion between env_id and HF dataset name.
    """
    encoded = env_id.replace("_", "_underscore_")
    encoded = encoded.replace("-", "_dash_")
    encoded = encoded.replace("/", "_slash_")
    return encoded


def _decode_hf_repo_name(repo_name):
    """
    Decode HF dataset name back to original env_id.

    Reverse of _encode_env_id_for_hf().
    """
    # Must decode in reverse order to handle overlapping patterns correctly
    decoded = repo_name.replace("_slash_", "/")
    decoded = decoded.replace("_dash_", "-")
    decoded = decoded.replace("_underscore_", "_")
    return decoded


def env_id_to_hf_repo_id(env_id):
    encoded_env_id = _encode_env_id_for_hf(env_id)
    hf_repo_id = f"{_current_hf_username()}/{CONFIG['dataset']['repo_prefix']}{encoded_env_id}"
    return hf_repo_id


def normalize_dataset_repo_id(value):
    """Normalize an owner/repo or hf://owner/repo dataset reference."""
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        text = text.removeprefix(HUGGINGFACE_MODEL_SCHEME).strip("/")
    parts = [unquote(part) for part in text.split("/") if part]
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        raise ValueError(
            f"expected a Hugging Face dataset repo like owner/repo or hf://owner/repo, "
            f"got {value!r}"
        )
    return "/".join(parts)


@dataclass(frozen=True)
class RecordingIdentity:
    """Identify one recording collection independently from its environment."""

    env_id: str | None = None
    dataset_repo_id: str | None = None

    def __post_init__(self):
        if self.dataset_repo_id is not None:
            object.__setattr__(
                self,
                "dataset_repo_id",
                normalize_dataset_repo_id(self.dataset_repo_id),
            )
        if not self.env_id and not self.dataset_repo_id:
            raise ValueError("recording identity requires an environment or dataset repo")

    @property
    def display_ref(self):
        if self.dataset_repo_id:
            return f"{HUGGINGFACE_MODEL_SCHEME}{self.dataset_repo_id}"
        return self.env_id

    def with_env_id(self, env_id):
        if not env_id or env_id == self.env_id:
            return self
        return RecordingIdentity(env_id=str(env_id), dataset_repo_id=self.dataset_repo_id)


def _coerce_recording_identity(value, *, env_id=None):
    if isinstance(value, RecordingIdentity):
        return value.with_env_id(env_id)
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        return RecordingIdentity(
            env_id=env_id,
            dataset_repo_id=normalize_dataset_repo_id(text),
        )
    return RecordingIdentity(env_id=env_id or text)


def _identity_hf_repo_id(value):
    identity = _coerce_recording_identity(value)
    if identity.dataset_repo_id:
        return identity.dataset_repo_id
    return env_id_to_hf_repo_id(identity.env_id)


def _policy_recording_identity(policy_source, dataset_repo=None):
    """Build the dataset identity paired with a resolved HF policy source."""
    return RecordingIdentity(
        env_id=policy_source.env_id,
        dataset_repo_id=dataset_repo or policy_source.repo_id,
    )


def hf_repo_id_to_env_id(hf_repo_id):
    """Convert HF repo_id back to original env_id."""
    prefix = CONFIG["dataset"]["repo_prefix"]
    # Extract just the repo name part (after username/)
    if "/" in hf_repo_id:
        repo_name = hf_repo_id.split("/", 1)[1]
    else:
        repo_name = hf_repo_id

    if not repo_name.startswith(prefix):
        return None

    encoded_env_id = repo_name[len(prefix) :]
    return _decode_hf_repo_name(encoded_env_id)


def _get_recording_root(value):
    identity = _coerce_recording_identity(value)
    if identity.dataset_repo_id:
        owner, repo = identity.dataset_repo_id.split("/", 1)
        return os.path.join(CONFIG["storage"]["local_dir"], "repos", owner, repo)
    return None


def get_local_dataset_path(value):
    """Return the local dataset path for an environment or repo-keyed identity."""
    identity = _coerce_recording_identity(value)
    recording_root = _get_recording_root(identity)
    if recording_root:
        return os.path.join(recording_root, "dataset")
    encoded_env_id = _encode_env_id_for_hf(identity.env_id)
    return os.path.join(CONFIG["storage"]["local_dir"], encoded_env_id)


def _get_available_envs_from_local():
    """Get list of env_ids that have local recordings."""
    local_dir = CONFIG["storage"]["local_dir"]
    if not os.path.exists(local_dir):
        return []

    available = []
    for entry in os.listdir(local_dir):
        entry_path = os.path.join(local_dir, entry)
        if os.path.isdir(entry_path):
            # Check if this is a valid dataset directory
            if os.path.exists(os.path.join(entry_path, "dataset_info.json")):
                env_id = _decode_hf_repo_name(entry)
                # Skip directories that do not round-trip through the current
                # reversible encoding scheme.
                if _encode_env_id_for_hf(env_id) != entry:
                    continue
                available.append(env_id)
    return sorted(set(available))


def _get_available_recording_refs_from_local():
    """Return environment IDs and repo-keyed refs available on local disk."""
    available = list(_get_available_envs_from_local())
    repos_dir = os.path.join(CONFIG["storage"]["local_dir"], "repos")
    if not os.path.isdir(repos_dir):
        return sorted(set(available))
    for owner in os.listdir(repos_dir):
        owner_dir = os.path.join(repos_dir, owner)
        if not os.path.isdir(owner_dir):
            continue
        for repo in os.listdir(owner_dir):
            dataset_dir = os.path.join(owner_dir, repo, "dataset")
            if os.path.exists(os.path.join(dataset_dir, "dataset_info.json")):
                available.append(f"{HUGGINGFACE_MODEL_SCHEME}{owner}/{repo}")
    return sorted(set(available))


def _get_available_envs_from_hf():
    """Get list of env_ids that have HF Hub recordings."""
    try:
        username = _current_hf_username()
    except Exception:
        return []

    prefix = CONFIG["dataset"]["repo_prefix"]
    available = []

    try:
        api = HfApi()
        # List all datasets for this user
        datasets = api.list_datasets(author=username)
        for ds in datasets:
            if ds.id and ds.id.startswith(f"{username}/{prefix}"):
                env_id = hf_repo_id_to_env_id(ds.id)
                if env_id:
                    available.append(env_id)
    except Exception:
        pass

    return sorted(set(available))


def _get_available_recording_refs_from_hf():
    """Return gymrec environment IDs and repo-keyed refs found on the Hub."""
    try:
        username = _current_hf_username()
        datasets = HfApi().list_datasets(author=username)
    except Exception:
        return []

    prefix = CONFIG["dataset"]["repo_prefix"]
    available = []
    api = HfApi()
    for dataset_info in datasets:
        repo_id = getattr(dataset_info, "id", None)
        if not repo_id:
            continue
        repo_name = repo_id.split("/", 1)[-1]
        if repo_name.startswith(prefix):
            env_id = hf_repo_id_to_env_id(repo_id)
            if env_id:
                available.append(env_id)
            continue
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        except Exception:
            continue
        if ENVIRONMENT_METADATA_FILENAME in files:
            available.append(f"{HUGGINGFACE_MODEL_SCHEME}{repo_id}")
    return sorted(set(available))


def _get_metadata_path(value):
    """Return the recording metadata sidecar path."""
    identity = _coerce_recording_identity(value)
    recording_root = _get_recording_root(identity)
    if recording_root:
        return os.path.join(recording_root, "metadata.json")
    encoded_env_id = _encode_env_id_for_hf(identity.env_id)
    return os.path.join(CONFIG["storage"]["local_dir"], f"{encoded_env_id}_metadata.json")


def _get_uploaded_episodes_path(value):
    """Return the uploaded-episode tracking path for a recording identity."""
    identity = _coerce_recording_identity(value)
    recording_root = _get_recording_root(identity)
    if recording_root:
        return os.path.join(recording_root, "uploaded.json")
    encoded_env_id = _encode_env_id_for_hf(identity.env_id)
    return os.path.join(CONFIG["storage"]["local_dir"], f"{encoded_env_id}_uploaded.json")


def _load_uploaded_episode_ids(value):
    """Load the set of already-uploaded episode IDs from local tracking file."""
    path = _get_uploaded_episodes_path(value)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f))


def _save_uploaded_episode_ids(value, episode_ids: set):
    """Save the set of uploaded episode IDs to local tracking file."""
    path = _get_uploaded_episodes_path(value)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(episode_ids), f)


def _remote_dataset_episode_ids(value):
    """Return episode ids currently present on the Hub, or None if unavailable."""
    hf_repo_id = _identity_hf_repo_id(value)
    api = HfApi()
    try:
        repo_exists, _, remote_files = _hf_repo_state(api, hf_repo_id, create=False)
    except Exception:
        return None
    if not repo_exists:
        return set()
    has_data_shard = any(
        path.startswith("data/") and path.endswith(".parquet") for path in remote_files
    )
    if not has_data_shard:
        return set()

    try:
        dataset = load_dataset(hf_repo_id, split="train", streaming=True)
        return {
            _normalize_episode_id(row["episode_id"])
            for row in dataset
            if row.get("episode_id") is not None
        }
    except Exception:
        return None


def _get_live_upload_queue_dir(value):
    """Return the local resumable live-upload queue directory."""
    identity = _coerce_recording_identity(value)
    recording_root = _get_recording_root(identity)
    if recording_root:
        return os.path.join(recording_root, "live_pending")
    encoded_env_id = _encode_env_id_for_hf(identity.env_id)
    return os.path.join(CONFIG["storage"]["local_dir"], f"{encoded_env_id}_live_pending")


def _get_live_upload_manifest_path(env_id):
    return os.path.join(_get_live_upload_queue_dir(env_id), "manifest.json")


def _load_live_upload_manifest(env_id):
    path = _get_live_upload_manifest_path(env_id)
    if not os.path.exists(path):
        return {"version": 1, "episodes": {}}
    with open(path, "r") as f:
        manifest = json.load(f)
    manifest.setdefault("version", 1)
    manifest.setdefault("episodes", {})
    return manifest


def _save_live_upload_manifest(env_id, manifest):
    path = _get_live_upload_manifest_path(env_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)


def _set_live_upload_manifest_entry(
    env_id,
    episode_id,
    *,
    state,
    package_dir,
    storage_format,
    frames,
    error=None,
):
    manifest = _load_live_upload_manifest(env_id)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    entry = manifest["episodes"].get(episode_id, {})
    entry.update(
        {
            "state": state,
            "package_dir": package_dir,
            "storage_format": storage_format,
            "frames": int(frames),
            "updated_at": now,
        }
    )
    entry.setdefault("created_at", now)
    if error:
        entry["error"] = str(error)
    elif "error" in entry:
        del entry["error"]
    manifest["episodes"][episode_id] = entry
    _save_live_upload_manifest(env_id, manifest)


def _pending_live_upload_entries(env_id):
    manifest = _load_live_upload_manifest(env_id)
    for episode_id, entry in sorted(manifest.get("episodes", {}).items()):
        if entry.get("state") in {"recording", "pending", "failed"}:
            package_dir = entry.get("package_dir")
            if package_dir and os.path.exists(package_dir):
                yield episode_id, entry


def _copy_artifact_tree(src_root, dst_root, relative_dir):
    """Copy an artifact subdirectory into a saved local dataset directory."""
    if not src_root:
        return
    src_dir = os.path.join(src_root, relative_dir)
    if not os.path.exists(src_dir):
        return
    dst_dir = os.path.join(dst_root, relative_dir)
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst_dir, name))


def _copy_video_artifacts(src_root, dst_root):
    """Copy canonical and preview video artifacts into a saved local dataset directory."""
    for relative_dir in {
        VIDEO_ARTIFACT_DIR,
        PREVIEW_VIDEO_ARTIFACT_DIR,
    }:
        _copy_artifact_tree(src_root, dst_root, relative_dir)


def save_dataset_locally(dataset, value, metadata=None, video_artifact_dir=None):
    """Save dataset to local disk, appending to any existing data."""
    identity = _coerce_recording_identity(value)
    path = get_local_dataset_path(identity)
    metadata_path = _get_metadata_path(identity)
    dataset = _align_environment_metadata_columns(_strip_runtime_columns(dataset))
    new_storage_format = _dataset_storage_format(dataset)
    new_episode_count = (
        len(set(dataset["episode_id"])) if "episode_id" in dataset.column_names else 0
    )
    new_frame_count = len(dataset)
    new_collectors = set(dataset["collector"]) if "collector" in dataset.column_names else set()
    new_versions = (
        set(dataset["gymrec_version"]) if "gymrec_version" in dataset.column_names else set()
    )

    existing_video_dir = path
    if os.path.exists(path):
        # Load existing dataset - UUIDs are already unique, no offsetting needed
        existing_dataset = _align_environment_metadata_columns(
            load_from_disk(path, keep_in_memory=True)
        )
        existing_storage_format = _dataset_storage_format(existing_dataset)
        if existing_storage_format != new_storage_format:
            raise ValueError(
                "Cannot append "
                f"{new_storage_format} recordings to existing {existing_storage_format} dataset "
                f"at {path}. Use a different [storage].local_dir or migrate explicitly."
            )

        # Concatenate datasets
        from datasets import concatenate_datasets

        dataset = concatenate_datasets([existing_dataset, dataset])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        dataset.save_to_disk(tmp_path)
        if os.path.exists(existing_video_dir):
            _copy_video_artifacts(existing_video_dir, tmp_path)
        _copy_video_artifacts(video_artifact_dir, tmp_path)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.replace(tmp_path, path)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    # Save/update metadata
    if metadata is not None:
        existing_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
        # Update with new metadata (newer values take precedence)
        existing_metadata.update(metadata)
        if identity.dataset_repo_id:
            existing_metadata["dataset_repo_id"] = identity.dataset_repo_id
        existing_metadata["storage_format"] = new_storage_format
        # Add recording timestamp
        if "recordings" not in existing_metadata:
            existing_metadata["recordings"] = []
        recording_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "episodes": new_episode_count,
            "frames": new_frame_count,
            "storage_format": new_storage_format,
        }
        if new_collectors:
            recording_entry["collectors"] = sorted(new_collectors)
        if new_versions:
            recording_entry["gymrec_versions"] = sorted(new_versions)
        for key in (
            "capture_backend",
            "state_name",
            "rom_sha256",
            "nes_button_order",
            "action_encoding",
            "frameskip",
            "sticky_actions",
            "policy_source_repo",
            "policy_source_revision",
            "policy_recipe_sha256",
            "policy_checkpoint_sha256",
            "policy_recipe_format_version",
            "policy_model_format_version",
        ):
            if key in metadata:
                recording_entry[key] = metadata[key]
        existing_metadata["recordings"].append(recording_entry)
        with open(metadata_path, "w") as f:
            json.dump(existing_metadata, f, indent=2, default=_json_default)

    console.print(f"Dataset saved locally ([{STYLE_PATH}]{path}[/])")
    return path


def load_local_metadata(value):
    """Load metadata from local disk. Returns None if not found."""
    metadata_path = _get_metadata_path(value)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_local_dataset(value, attach_runtime=True):
    """Load dataset from local disk. Returns None if not found."""
    path = get_local_dataset_path(value)
    if not os.path.exists(path):
        return None
    dataset = load_from_disk(path)
    if attach_runtime:
        dataset = _attach_video_runtime_source(dataset, local_base_path=path)
    return dataset


def load_recorded_dataset(value):
    """Load a recorded dataset from local disk first, then from the Hub."""
    identity = _coerce_recording_identity(value)
    local_dataset = load_local_dataset(identity)
    if local_dataset is not None:
        return local_dataset, "local"

    try:
        hf_repo_id = _identity_hf_repo_id(identity)
        dataset = load_dataset(hf_repo_id, split="train")
    except Exception:
        return None, None

    dataset = _attach_hub_environment_metadata(dataset, hf_repo_id)
    dataset = _attach_video_runtime_source(dataset, hf_repo_id=hf_repo_id)
    return dataset, "hub"


def _recording_env_id(value, dataset=None, metadata=None):
    """Recover a recording's logical environment from identity, metadata, or rows."""
    identity = _coerce_recording_identity(value)
    if identity.env_id:
        return identity.env_id
    metadata = metadata or load_local_metadata(identity) or {}
    env_id = metadata.get("env_id") or metadata.get("logical_env_id")
    if env_id:
        return str(env_id)
    dataset_metadata = _environment_metadata_from_dataset(dataset) if dataset is not None else {}
    env_id = dataset_metadata.get("env_id")
    return str(env_id) if env_id else None


def _print_missing_dataset(value):
    identity = _coerce_recording_identity(value)
    console.print(f"[{STYLE_FAIL}]No dataset found for {identity.display_ref}.[/]")
    console.print(f"  Local path: [{STYLE_PATH}]{get_local_dataset_path(identity)}[/]")
    console.print(
        f"  Record a session first: [{STYLE_CMD}]{_gymrec_cmd('record', identity.display_ref)}[/]"
    )


def _is_terminal_action(action):
    """Return True for terminal observation rows with no action."""
    return action is None or (isinstance(action, list) and len(action) == 0)


def _dataset_storage_format(dataset):
    """Infer a dataset's observation storage format from columns and values."""
    column_names = getattr(dataset, "column_names", None) or []
    if "storage_format" in column_names and len(dataset) > 0:
        for value in dataset["storage_format"]:
            if value:
                return _normalize_storage_format(value)
    if "video_path" in column_names:
        return STORAGE_FORMAT_LOSSLESS_VIDEO
    return STORAGE_FORMAT_IMAGES


def _is_video_row(row):
    return row.get("video_path") not in (None, "")


def _attach_video_runtime_source(dataset, local_base_path=None, hf_repo_id=None):
    """Attach runtime-only source columns needed to resolve relative video files."""
    if "video_path" not in dataset.column_names:
        return dataset
    if local_base_path is not None and RUNTIME_VIDEO_BASE_COLUMN not in dataset.column_names:
        dataset = dataset.add_column(RUNTIME_VIDEO_BASE_COLUMN, [local_base_path] * len(dataset))
    if hf_repo_id is not None and RUNTIME_HF_REPO_COLUMN not in dataset.column_names:
        dataset = dataset.add_column(RUNTIME_HF_REPO_COLUMN, [hf_repo_id] * len(dataset))
    return dataset


def _strip_runtime_columns(dataset):
    """Remove local-only resolver columns before writing/uploading datasets."""
    runtime_columns = [
        column
        for column in (RUNTIME_VIDEO_BASE_COLUMN, RUNTIME_HF_REPO_COLUMN)
        if column in dataset.column_names
    ]
    if runtime_columns:
        dataset = dataset.remove_columns(runtime_columns)
    return dataset


def _is_step_row(row):
    """Filter out terminal observation rows."""
    return not _is_terminal_action(row.get("actions"))


def _get_default_fps_for_env_id(env_id, metadata=None):
    """Infer a sensible FPS without instantiating the environment."""
    if metadata and metadata.get("fps") is not None:
        try:
            return max(int(round(float(metadata["fps"]))), 1)
        except (TypeError, ValueError):
            pass

    backend = classify_env_id(env_id)
    if backend == "stable-retro":
        return CONFIG["fps_defaults"]["retro"]
    if backend == "vizdoom":
        return CONFIG["fps_defaults"]["vizdoom"]
    return CONFIG["fps_defaults"]["atari"]


def _human_record_env_make_kwargs(backend):
    """Return deterministic control kwargs for human recording on supported backends."""
    if backend == "atari":
        return {
            "frameskip": HUMAN_RECORD_FRAMESKIP,
            "repeat_action_probability": HUMAN_RECORD_STICKY_ACTION_PROB,
        }
    if backend in {"stable-retro", "stable-retro-turbo", "supermariobrosnes-turbo"}:
        return {
            "frame_skip": HUMAN_RECORD_FRAMESKIP,
            "sticky_action_prob": HUMAN_RECORD_STICKY_ACTION_PROB,
        }
    return {}


def _env_make_kwargs_from_metadata(env_id, metadata, backend=None):
    """Rebuild environment creation kwargs from saved recording metadata."""
    if not metadata:
        return {}

    saved_kwargs = metadata.get("env_make_kwargs")
    if isinstance(saved_kwargs, dict):
        return dict(saved_kwargs)

    backend = backend or metadata.get("backend") or classify_env_id(env_id)
    try:
        frameskip = _metadata_int(metadata, ("frameskip",), default=None)
    except (TypeError, ValueError):
        frameskip = None
    try:
        sticky = _metadata_float(metadata, ("sticky_actions",), default=None)
    except (TypeError, ValueError):
        sticky = None
    kwargs = {}

    if backend == "atari":
        if frameskip is not None:
            kwargs["frameskip"] = max(frameskip, 1)
        if sticky is not None:
            kwargs["repeat_action_probability"] = sticky
    elif backend in {"stable-retro", "stable-retro-turbo", "supermariobrosnes-turbo"}:
        if frameskip is not None:
            kwargs["frame_skip"] = max(frameskip, 1)
        if sticky is not None:
            kwargs["sticky_action_prob"] = sticky

    return kwargs


def _stable_retro_state_from_metadata(metadata):
    if not metadata:
        return None
    return _metadata_str(
        metadata,
        ("state_name",),
        ("stable_retro_state",),
        ("retro_state",),
        ("state",),
        default=None,
    )


def _normalize_episode_id(eid):
    """Normalize episode identifiers to a stable string key."""
    if isinstance(eid, bytes):
        try:
            return uuid.UUID(bytes=eid).hex
        except ValueError:
            return eid.hex()
    if isinstance(eid, uuid.UUID):
        return eid.hex
    return str(eid)


def _ordered_episode_rows(dataset):
    """Return episode keys in encounter order and the row indices for each episode."""
    episode_keys = []
    row_indices_by_episode = {}

    for row_index, eid in enumerate(dataset["episode_id"]):
        key = _normalize_episode_id(eid)
        if key not in row_indices_by_episode:
            row_indices_by_episode[key] = []
            episode_keys.append(key)
        row_indices_by_episode[key].append(row_index)

    return episode_keys, row_indices_by_episode


def _episode_reset_seed(dataset, row_indices):
    """Return the first usable reset seed recorded for an episode."""
    for row_index in row_indices:
        seed = dataset[row_index].get("seed")
        if seed is not None:
            try:
                return int(seed)
            except (TypeError, ValueError):
                continue
    return None


def _iter_playback_items(dataset, row_indices, verify=False):
    """Yield replay items for one episode, excluding terminal observation rows."""
    for position, row_index in enumerate(row_indices):
        row = dataset[row_index]
        if not _is_step_row(row):
            continue
        if verify:
            if position + 1 >= len(row_indices):
                episode_key = _normalize_episode_id(row.get("episode_id"))
                raise ValueError(f"Episode {episode_key} is missing the terminal observation row")
            next_row = dataset[row_indices[position + 1]]
            yield (
                row.get("actions"),
                _get_row_observation(next_row),
                row.get("rewards"),
                row.get("terminations"),
                row.get("truncations"),
            )
        else:
            yield row.get("actions")


def _dataset_playback_episodes(dataset, verify=False):
    """Group a flat N+1 dataset into replayable episodes."""
    episode_keys, row_indices_by_episode = _ordered_episode_rows(dataset)
    episodes = []
    total_steps = 0

    for episode_key in episode_keys:
        row_indices = tuple(row_indices_by_episode[episode_key])
        step_count = 0
        for row_index in row_indices:
            if _is_step_row(dataset[row_index]):
                step_count += 1
        if step_count == 0:
            continue

        episodes.append(
            {
                "seed": _episode_reset_seed(dataset, row_indices),
                "items": _iter_playback_items(dataset, row_indices, verify=verify),
            }
        )
        total_steps += step_count

    return episodes, total_steps


def _parse_episode_range(value, total_episodes):
    """Parse a 1-based inclusive episode range like '3-7' or '3:7'."""
    text = str(value).strip()
    for separator in ("..", "-", ":"):
        if separator in text:
            parts = text.split(separator, 1)
            break
    else:
        raise ValueError("Episode range must look like START-END, START:END, or START..END")

    try:
        start = int(parts[0].strip())
        end = int(parts[1].strip())
    except (TypeError, ValueError):
        raise ValueError(
            "Episode range must look like START-END, START:END, or START..END"
        ) from None

    if start < 1 or end < 1:
        raise ValueError("Episode numbers are 1-based and must be >= 1")
    if start > end:
        raise ValueError("Episode range start must be <= end")
    if end > total_episodes:
        raise ValueError(
            f"Episode range {start}-{end} exceeds dataset size ({total_episodes} episodes)"
        )

    return start, end


def _select_episode_numbers(total_episodes, episode_range=None, first=None, last=None):
    """Select 1-based episode numbers from a dataset."""
    if total_episodes <= 0:
        return []

    selectors = [episode_range is not None, first is not None, last is not None]
    if sum(selectors) > 1:
        raise ValueError("Use only one of --range, --first, or --last")

    if episode_range is not None:
        start, end = _parse_episode_range(episode_range, total_episodes)
        return list(range(start, end + 1))

    if first is not None:
        if first < 1:
            raise ValueError("--first must be >= 1")
        return list(range(1, min(first, total_episodes) + 1))

    if last is not None:
        if last < 1:
            raise ValueError("--last must be >= 1")
        start = max(1, total_episodes - last + 1)
        return list(range(start, total_episodes + 1))

    return list(range(1, total_episodes + 1))


def _get_row_observation(row):
    """Return the observation image from a dataset row."""
    if _is_video_row(row):
        return _get_video_row_observation(row)
    observation = row.get("observations")
    if observation is None:
        raise ValueError("Dataset row is missing observations")
    return observation


def _resolve_video_path(row):
    """Resolve a row's relative video path to a local file path."""
    video_path = row.get("video_path")
    if os.path.isabs(video_path) and os.path.exists(video_path):
        return video_path

    local_base_path = row.get(RUNTIME_VIDEO_BASE_COLUMN)
    if local_base_path:
        candidate = os.path.join(local_base_path, video_path)
        if os.path.exists(candidate):
            return candidate

    hf_repo_id = row.get(RUNTIME_HF_REPO_COLUMN)
    if hf_repo_id:
        return hf_hub_download(repo_id=hf_repo_id, filename=video_path, repo_type="dataset")

    if os.path.exists(video_path):
        return video_path
    raise FileNotFoundError(f"Could not resolve video-backed observation: {video_path}")


def _get_video_row_observation(row):
    """Decode and verify one video-backed observation row."""
    video_path = _resolve_video_path(row)
    width = int(row.get("frame_width"))
    height = int(row.get("frame_height"))
    frame_index = int(row.get("frame_index"))
    frames = _decode_lossless_rgb_video(video_path, width, height)
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(
            f"Frame index {frame_index} out of range for {video_path} ({len(frames)} frames)"
        )
    frame = np.array(frames[frame_index], copy=True)
    expected_hash = row.get("frame_sha256")
    if expected_hash and _sha256_rgb(frame) != expected_hash:
        raise RuntimeError(f"Decoded frame hash mismatch for {video_path} frame {frame_index}")
    return frame


def _overlay_episode_number(frame_array, episode_number):
    """Draw an episode label in the top-left corner of a frame."""
    from PIL import ImageDraw, ImageFont

    image = PILImage.fromarray(frame_array)
    draw = ImageDraw.Draw(image)
    font_size = max(14, image.height // 18)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    label = f"Episode {episode_number}"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    padding = max(6, image.height // 54)
    box_left = padding
    box_top = padding
    box_right = box_left + (right - left) + padding * 2
    box_bottom = box_top + (bottom - top) + padding * 2

    draw.rectangle((box_left, box_top, box_right, box_bottom), fill=(0, 0, 0))
    draw.text(
        (box_left + padding, box_top + padding - top),
        label,
        fill=(255, 255, 255),
        font=font,
    )
    return np.asarray(image, dtype=np.uint8)


def _default_video_output_path(env_id, selected_episode_numbers):
    """Build a deterministic default output path for a video export."""
    encoded_env_id = _encode_env_id_for_hf(env_id)
    if not selected_episode_numbers:
        suffix = "episodes"
    elif len(selected_episode_numbers) == 1:
        suffix = f"episode_{selected_episode_numbers[0]}"
    elif len(selected_episode_numbers) > 1 and selected_episode_numbers == list(
        range(selected_episode_numbers[0], selected_episode_numbers[-1] + 1)
    ):
        suffix = f"episodes_{selected_episode_numbers[0]}_{selected_episode_numbers[-1]}"
    else:
        suffix = f"episodes_{len(selected_episode_numbers)}"

    return os.path.abspath(f"{encoded_env_id}_{suffix}.mp4")


def export_dataset_video(
    env_id,
    dataset,
    output_path=None,
    fps=None,
    episode_range=None,
    first=None,
    last=None,
):
    """Render selected dataset episodes to an MP4 video with episode overlay labels."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        console.print(f"[{STYLE_FAIL}]ffmpeg is required to export videos.[/]")
        return False
    if fps is None or int(fps) < 1:
        console.print(f"[{STYLE_FAIL}]FPS must be a positive integer.[/]")
        return False

    episode_keys, row_indices_by_episode = _ordered_episode_rows(dataset)
    total_episodes = len(episode_keys)
    if total_episodes == 0:
        console.print(f"[{STYLE_FAIL}]Dataset has no episodes.[/]")
        return False

    try:
        selected_episode_numbers = _select_episode_numbers(
            total_episodes, episode_range=episode_range, first=first, last=last
        )
    except ValueError as e:
        console.print(f"[{STYLE_FAIL}]{e}[/]")
        return False

    selected_episodes = [
        (episode_number, row_indices_by_episode[episode_keys[episode_number - 1]])
        for episode_number in selected_episode_numbers
    ]
    total_frames = sum(len(row_indices) for _, row_indices in selected_episodes)
    if total_frames == 0:
        console.print(f"[{STYLE_FAIL}]No frames found for selected episodes.[/]")
        return False

    first_frame = _observation_to_rgb_array(
        _get_row_observation(dataset[selected_episodes[0][1][0]])
    )
    height, width = first_frame.shape[:2]

    if output_path is None:
        output_path = _default_video_output_path(env_id, selected_episode_numbers)
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ffmpeg_cmd = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    console.print(
        f"[{STYLE_INFO}]Exporting {len(selected_episode_numbers)} episode(s) "
        f"({total_frames} frames) to [{STYLE_PATH}]{output_path}[/] "
        f"at {fps} FPS[/]"
    )

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        with _episode_progress(transient=False) as progress:
            task_id = progress.add_task("[bold]Encoding[/]", total=total_frames)
            for episode_number, row_indices in selected_episodes:
                for row_index in row_indices:
                    row = dataset[row_index]
                    frame = _observation_to_rgb_array(_get_row_observation(row))
                    if frame.shape[:2] != (height, width):
                        raise ValueError(
                            "All exported frames must have the same dimensions; "
                            f"expected {(height, width)}, got {frame.shape[:2]}"
                        )
                    frame = _overlay_episode_number(frame, episode_number)
                    process.stdin.write(frame.tobytes())
                    progress.advance(task_id)

        process.stdin.close()
        ffmpeg_stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(ffmpeg_stderr or f"ffmpeg exited with status {return_code}")
    except Exception as e:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        console.print(f"[{STYLE_FAIL}]Video export failed: {e}[/]")
        return False
    finally:
        if process.stderr is not None:
            process.stderr.close()

    console.print(f"[{STYLE_SUCCESS}]Video exported: [{STYLE_PATH}]{output_path}[/][/]")
    return True


REMOTE_STORAGE_FORMAT_UNSUPPORTED = "unsupported"


def _preview_video_relpath(episode_stem):
    return f"{PREVIEW_VIDEO_ARTIFACT_DIR}/{episode_stem}{PREVIEW_VIDEO_SUFFIX}"


def _remote_storage_format_from_files(file_paths):
    """Infer the remote dataset storage layout from committed Hub files."""
    has_data_shard = any(
        path.startswith("data/") and path.endswith(".parquet") for path in file_paths
    )
    has_canonical_video = any(
        path.startswith(f"{VIDEO_ARTIFACT_DIR}/") and path.endswith(CANONICAL_VIDEO_SUFFIX)
        for path in file_paths
    )
    has_unsupported_video = any(
        path.startswith(f"{VIDEO_ARTIFACT_DIR}/")
        and not (path.endswith(CANONICAL_VIDEO_SUFFIX) or path.endswith(PREVIEW_VIDEO_SUFFIX))
        for path in file_paths
    )

    if has_canonical_video:
        return STORAGE_FORMAT_LOSSLESS_VIDEO
    if has_unsupported_video:
        return REMOTE_STORAGE_FORMAT_UNSUPPORTED
    if has_data_shard:
        return STORAGE_FORMAT_IMAGES
    return None


def _remote_storage_conflict_message(env_id, hf_repo_id, local_format, remote_format):
    if remote_format is None or remote_format == local_format:
        return None
    remote_label = remote_format
    detail = "Its files do not match the current local observation schema."

    return (
        f"Remote dataset {hf_repo_id} already contains {remote_label} data, "
        f"but the local dataset is {local_format}. {detail} "
        "Refusing to append because Hugging Face would show a mixed/stale schema. "
        f"Run `{_gymrec_cmd('upload', env_id)} --replace` to intentionally replace "
        "the remote files with the current local dataset."
    )


def _remote_parquet_columns(hf_repo_id, remote_files):
    data_files = sorted(
        path for path in remote_files if path.startswith("data/") and path.endswith(".parquet")
    )
    if not data_files:
        return None
    shard_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=data_files[0],
        repo_type="dataset",
    )
    return Dataset.from_parquet(shard_path).column_names


def _match_remote_parquet_schema(dataset, remote_columns):
    """Keep append shards loadable when a legacy Hub repo has fewer columns."""
    if not remote_columns or dataset.column_names == remote_columns:
        return dataset
    missing = [column for column in remote_columns if column not in dataset.column_names]
    if missing:
        raise ValueError(
            "Local data is missing remote dataset columns: " + ", ".join(sorted(missing))
        )
    return dataset.select_columns(remote_columns)


def _hf_repo_state(api, hf_repo_id, create=False):
    """Return repo existence, parent commit, and file list, optionally creating the repo."""
    repo_exists = False
    parent_commit = None
    remote_files = []
    try:
        repo_info = api.repo_info(repo_id=hf_repo_id, repo_type="dataset")
        repo_exists = True
        parent_commit = repo_info.sha
        remote_files = list(api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset"))
    except Exception:
        if not create:
            return repo_exists, parent_commit, remote_files

    if not repo_exists and create:
        api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True)
        repo_info = api.repo_info(repo_id=hf_repo_id, repo_type="dataset")
        repo_exists = True
        parent_commit = repo_info.sha
        remote_files = list(api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset"))
    return repo_exists, parent_commit, remote_files


def _next_hf_shard_index(api, hf_repo_id, repo_exists, replace=False):
    if not repo_exists or replace:
        return 0
    next_shard_idx = 0
    try:
        for item in api.list_repo_tree(hf_repo_id, repo_type="dataset", path_in_repo="data"):
            name = item.rfilename if hasattr(item, "rfilename") else str(item)
            if name.startswith("data/train-") and name.endswith(".parquet"):
                next_shard_idx += 1
    except Exception:
        pass
    return next_shard_idx


def _upload_dataset_shard_to_hub(
    recording_identity,
    dataset,
    *,
    storage_format,
    local_root,
    episode_ids,
    replace=False,
    include_previews=True,
    max_retries=5,
    base_wait=1.0,
):
    """Upload an already-materialized dataset shard and its artifacts to the Hub."""
    identity = _coerce_recording_identity(recording_identity)
    env_id = _recording_env_id(identity, dataset=dataset) or "unknown"
    hf_repo_id = _identity_hf_repo_id(identity)
    api = HfApi()
    episode_ids = set(episode_ids)

    for attempt in range(1, max_retries + 1):
        try:
            repo_exists, parent_commit, remote_files = _hf_repo_state(api, hf_repo_id, create=True)
            if repo_exists and not replace:
                remote_format = _remote_storage_format_from_files(remote_files)
                conflict_message = _remote_storage_conflict_message(
                    identity.display_ref, hf_repo_id, storage_format, remote_format
                )
                if conflict_message:
                    console.print(f"[{STYLE_FAIL}]{conflict_message}[/]")
                    return False

            next_shard_idx = _next_hf_shard_index(api, hf_repo_id, repo_exists, replace)
            operations = []
            if replace:
                operations.extend(
                    CommitOperationDelete(path_in_repo=path)
                    for path in remote_files
                    if path != ".gitattributes"
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                shard_path = os.path.join(tmpdir, "shard.parquet")
                upload_dataset = _strip_runtime_columns(dataset)
                if repo_exists and not replace:
                    remote_columns = _remote_parquet_columns(hf_repo_id, remote_files)
                    upload_dataset = _match_remote_parquet_schema(upload_dataset, remote_columns)
                upload_dataset.to_parquet(shard_path)
                shard_name = (
                    "data/train-00000-of-00001.parquet"
                    if replace
                    else f"data/train-{next_shard_idx:05d}-of-{next_shard_idx + 1:05d}.parquet"
                )
                operations.append(
                    CommitOperationAdd(path_in_repo=shard_name, path_or_fileobj=shard_path)
                )

                sidecar_metadata = {
                    **_environment_metadata_from_dataset(dataset),
                    **(load_local_metadata(identity) or {}),
                }
                if identity.dataset_repo_id:
                    sidecar_metadata["dataset_repo_id"] = identity.dataset_repo_id
                if sidecar_metadata:
                    sidecar_path = os.path.join(tmpdir, ENVIRONMENT_METADATA_FILENAME)
                    with open(sidecar_path, "w") as file_obj:
                        json.dump(sidecar_metadata, file_obj, indent=2, default=_json_default)
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=ENVIRONMENT_METADATA_FILENAME,
                            path_or_fileobj=sidecar_path,
                        )
                    )

                if storage_format == STORAGE_FORMAT_LOSSLESS_VIDEO:
                    video_paths = sorted(
                        {
                            row["video_path"]
                            for row in upload_dataset
                            if "video_path" in row and row["video_path"]
                        }
                    )
                    for video_relpath in video_paths:
                        video_path = os.path.join(local_root, video_relpath)
                        if not os.path.exists(video_path):
                            raise FileNotFoundError(
                                f"Missing video artifact for upload: {video_path}"
                            )
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=video_relpath,
                                path_or_fileobj=video_path,
                            )
                        )
                        if include_previews:
                            video_name = os.path.basename(video_relpath)
                            if video_name.endswith(CANONICAL_VIDEO_SUFFIX):
                                episode_stem = video_name[: -len(CANONICAL_VIDEO_SUFFIX)]
                                preview_relpath = _preview_video_relpath(episode_stem)
                                preview_path = os.path.join(local_root, preview_relpath)
                                if os.path.exists(preview_path):
                                    operations.append(
                                        CommitOperationAdd(
                                            path_in_repo=preview_relpath,
                                            path_or_fileobj=preview_path,
                                        )
                                    )

                card_content = _build_dataset_card_content(
                    identity,
                    env_id,
                    hf_repo_id,
                    new_frames=len(upload_dataset),
                    new_episodes=len(episode_ids),
                    repo_exists=repo_exists,
                )
                if card_content:
                    card_path = os.path.join(tmpdir, "README.md")
                    with open(card_path, "w") as f:
                        f.write(card_content)
                    operations.append(
                        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_path)
                    )

                api.preupload_lfs_files(
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                    additions=[op for op in operations if isinstance(op, CommitOperationAdd)],
                )
                api.create_commit(
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=(
                        f"Replace recordings from {env_id}"
                        if replace
                        else f"Add recordings from {env_id}"
                    ),
                    parent_commit=parent_commit,
                )

            uploaded_episode_ids = (
                episode_ids if replace else _load_uploaded_episode_ids(identity) | episode_ids
            )
            _save_uploaded_episode_ids(identity, uploaded_episode_ids)
            console.print(
                f"[{STYLE_SUCCESS}]Dataset uploaded: https://huggingface.co/datasets/{hf_repo_id}[/]"
            )
            return True

        except Exception as e:
            error_msg = str(e)
            if (
                "parent_commit" in error_msg.lower()
                or "conflict" in error_msg.lower()
                or "outdated" in error_msg.lower()
                or "412" in error_msg
                or "commit has happened" in error_msg.lower()
                or "precondition" in error_msg.lower()
            ):
                if attempt < max_retries:
                    wait_time = base_wait * (2 ** (attempt - 1))
                    console.print(
                        f"[{STYLE_INFO}]Conflict detected, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})...[/]"
                    )
                    time.sleep(wait_time)
                    continue
                console.print(f"[{STYLE_FAIL}]Max retries ({max_retries}) exceeded.[/]")
                console.print(f"[{STYLE_INFO}]Another client may be uploading. Try again later.[/]")
                return False
            console.print(f"[{STYLE_FAIL}]Upload failed: {e}[/]")
            if identity.dataset_repo_id:
                console.print(
                    f"[{STYLE_INFO}]If {hf_repo_id} is not writable, record with "
                    f"[{STYLE_CMD}]--dataset-repo <writable-owner/repo>[/]."
                )
            return False

    return False


def _load_live_episode_package_dataset(package_dir):
    shard_path = os.path.join(package_dir, "episode.parquet")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Missing pending live-upload shard: {shard_path}")
    return Dataset.from_parquet(shard_path)


def _load_live_episode_journal(package_dir):
    journal_path = os.path.join(package_dir, "journal.jsonl")
    if not os.path.exists(journal_path):
        return []
    records = []
    with open(journal_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _package_path(package_dir, relpath):
    return relpath if os.path.isabs(relpath) else os.path.join(package_dir, relpath)


def _recover_live_video_artifact(package_dir, records):
    first = records[0]
    last = records[-1]
    video_relpath = first["video_path"]
    video_path = _package_path(package_dir, video_relpath)
    width = int(first["frame_width"])
    height = int(first["frame_height"])
    step_hashes = [record["frame_sha256"] for record in records]
    terminal_path = _package_path(package_dir, last["terminal_candidate_path"])
    terminal_frame = _observation_to_rgb_array(PILImage.open(terminal_path))
    terminal_hash = _sha256_rgb(terminal_frame)
    fps = last.get("fps") or CONFIG["fps_defaults"]["atari"]

    try:
        _verify_lossless_rgb_video_stream(
            video_path,
            width,
            height,
            step_hashes + [terminal_hash],
        )
        return video_relpath, terminal_hash, len(step_hashes) + 1
    except Exception:
        pass

    _verify_lossless_rgb_video_stream(video_path, width, height, step_hashes)
    frames = _decode_lossless_rgb_video(video_path, width, height, cache=False)
    if len(frames) != len(step_hashes):
        raise RuntimeError(
            f"Recovered video has {len(frames)} step frames; expected {len(step_hashes)}"
        )
    recovered_frames = [frame.copy() for frame in frames]
    recovered_frames.append(terminal_frame)
    tmp_video_path = f"{video_path}.recovered"
    try:
        _encode_lossless_rgb_video(recovered_frames, tmp_video_path, fps)
        _verify_lossless_rgb_video_stream(
            tmp_video_path,
            width,
            height,
            step_hashes + [terminal_hash],
        )
        os.replace(tmp_video_path, video_path)
    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
    return video_relpath, terminal_hash, len(step_hashes) + 1


def _dataset_from_recovered_live_records(package_dir, episode_id, records):
    if not records:
        raise RuntimeError(f"No recoverable completed steps found for {episode_id}")
    records = sorted(records, key=lambda record: int(record.get("row_index", 0)))
    storage_format = _normalize_storage_format(records[0]["storage_format"])
    last = records[-1]
    last["truncation"] = bool(last.get("truncation") or not last.get("termination"))

    episode_bytes = uuid.UUID(hex=episode_id).bytes
    session_bytes = uuid.UUID(hex=records[0]["session_id"]).bytes
    data = {
        "episode_id": [episode_bytes for _ in records],
        "seed": [record["seed"] for record in records],
        "actions": [record["action"] for record in records],
        "rewards": [record["reward"] for record in records],
        "terminations": [bool(record["termination"]) for record in records],
        "truncations": [bool(record["truncation"]) for record in records],
        "infos": [record["info"] for record in records],
        "session_id": [session_bytes for _ in records],
        "collector": [records[0]["collector"] for _ in records],
        "gymrec_version": [records[0]["gymrec_version"] for _ in records],
        "storage_format": [storage_format for _ in records],
    }
    environment_columns = (
        "logical_env_id",
        "capture_backend",
        "state_name",
        "rom_sha256",
        "nes_button_order",
        "action_encoding",
        "frame_skip",
        "sticky_action_prob",
        "policy_source_repo",
        "policy_source_revision",
        "policy_recipe_sha256",
        "policy_checkpoint_sha256",
        "policy_recipe_format_version",
        "policy_model_format_version",
    )
    for column in environment_columns:
        if column in records[0]:
            data[column] = [record.get(column) for record in records]

    terminal_record = {
        "episode_id": episode_bytes,
        "seed": records[0]["seed"],
        "actions": [],
        "rewards": None,
        "terminations": None,
        "truncations": None,
        "infos": None,
        "session_id": session_bytes,
        "collector": records[0]["collector"],
        "gymrec_version": records[0]["gymrec_version"],
        "storage_format": storage_format,
    }
    for column in environment_columns:
        if column in data:
            terminal_record[column] = records[0].get(column)

    if storage_format == STORAGE_FORMAT_IMAGES:
        data["observations"] = [
            _package_path(package_dir, record["observation_path"]) for record in records
        ]
        terminal_record["observations"] = _package_path(
            package_dir, last["terminal_candidate_path"]
        )
    else:
        video_relpath, terminal_hash, total_observations = _recover_live_video_artifact(
            package_dir, records
        )
        data.update(
            {
                "video_path": [record["video_path"] for record in records],
                "frame_index": [int(record["frame_index"]) for record in records],
                "frame_sha256": [record["frame_sha256"] for record in records],
                "frame_width": [int(record["frame_width"]) for record in records],
                "frame_height": [int(record["frame_height"]) for record in records],
                "episode_num_observations": [total_observations for _ in records],
            }
        )
        terminal_record.update(
            {
                "video_path": video_relpath,
                "frame_index": len(records),
                "frame_sha256": terminal_hash,
                "frame_width": int(last["terminal_candidate_width"]),
                "frame_height": int(last["terminal_candidate_height"]),
                "episode_num_observations": total_observations,
            }
        )

    for key, value in terminal_record.items():
        data[key].append(value)

    return _recording_dataset_from_dict(data, storage_format)


def _recover_live_recording_package(episode_id, entry):
    package_dir = entry["package_dir"]
    records = _load_live_episode_journal(package_dir)
    return _dataset_from_recovered_live_records(package_dir, episode_id, records)


def drain_live_upload_queue(recording_identity, max_retries=5, base_wait=1.0):
    """Retry verified live-upload episode packages left from interrupted sessions."""
    identity = _coerce_recording_identity(recording_identity)
    entries = list(_pending_live_upload_entries(identity))
    if not entries:
        return True

    console.print(f"[{STYLE_INFO}]Draining {len(entries)} pending live-upload episode(s)...[/]")
    ok = True
    already_uploaded = _remote_dataset_episode_ids(identity)
    if already_uploaded is None:
        already_uploaded = _load_uploaded_episode_ids(identity)
    for episode_id, entry in entries:
        package_dir = entry["package_dir"]
        if episode_id in already_uploaded:
            _set_live_upload_manifest_entry(
                identity,
                episode_id,
                state="uploaded",
                package_dir=package_dir,
                storage_format=entry.get("storage_format", STORAGE_FORMAT_IMAGES),
                frames=entry.get("frames", 0),
            )
            shutil.rmtree(package_dir, ignore_errors=True)
            continue
        try:
            if entry.get("state") == "recording":
                console.print(
                    f"[{STYLE_INFO}]Recovering interrupted live episode {episode_id}...[/]"
                )
                dataset = _recover_live_recording_package(episode_id, entry)
                persist_dataset = True
            else:
                dataset = _load_live_episode_package_dataset(package_dir)
                persist_dataset = False
            storage_format = _dataset_storage_format(dataset)
            success = _upload_live_episode_package(
                identity,
                episode_id,
                package_dir,
                dataset,
                storage_format=storage_format,
                max_retries=max_retries,
                base_wait=base_wait,
                persist_dataset=persist_dataset,
            )
            if not success:
                ok = False
        except Exception as e:
            ok = False
            _set_live_upload_manifest_entry(
                identity,
                episode_id,
                state="failed",
                package_dir=package_dir,
                storage_format=entry.get("storage_format", STORAGE_FORMAT_IMAGES),
                frames=entry.get("frames", 0),
                error=e,
            )
            console.print(f"[{STYLE_FAIL}]Pending live upload failed for {episode_id}: {e}[/]")
    return ok


def preflight_live_upload(recording_identity, storage_format):
    """Validate live upload can reach the target Hub dataset before gameplay starts."""
    identity = _coerce_recording_identity(recording_identity)
    if not ensure_hf_login():
        return False
    if storage_format == STORAGE_FORMAT_LOSSLESS_VIDEO:
        try:
            _require_lossless_video_tools()
        except Exception as e:
            console.print(f"[{STYLE_FAIL}]Live upload preflight failed: {e}[/]")
            return False

    hf_repo_id = _identity_hf_repo_id(identity)
    api = HfApi()
    try:
        _, _, remote_files = _hf_repo_state(api, hf_repo_id, create=True)
    except Exception as e:
        console.print(f"[{STYLE_FAIL}]Live upload preflight failed: {e}[/]")
        if identity.dataset_repo_id:
            console.print(
                f"[{STYLE_INFO}]Use [{STYLE_CMD}]--dataset-repo <writable-owner/repo>[/] "
                f"when {hf_repo_id} is not writable."
            )
        return False

    remote_format = _remote_storage_format_from_files(remote_files)
    conflict_message = _remote_storage_conflict_message(
        identity.display_ref, hf_repo_id, storage_format, remote_format
    )
    if conflict_message:
        console.print(f"[{STYLE_FAIL}]{conflict_message}[/]")
        return False
    console.print(f"[{STYLE_INFO}]Live upload target ready: {hf_repo_id}[/]")
    return True


def upload_local_dataset(recording_identity, max_retries=5, base_wait=1.0, replace=False):
    """Upload new episodes to HF Hub using append-only shard uploads.

    Only uploads episodes that have not been uploaded before (tracked in a local
    JSON file). Uploads new data as a parquet shard alongside existing shards —
    no remote data is downloaded or replaced. Uses optimistic locking via
    parent_commit in create_commit() to handle concurrent uploads safely.

    Args:
        recording_identity: Environment ID or hf:// dataset recording identity.
        max_retries: Maximum number of retry attempts on conflict (default: 5)
        base_wait: Base wait time between retries in seconds (default: 1.0)
        replace: Replace all remote dataset files with the local dataset.
    """
    if not ensure_hf_login():
        return False

    identity = _coerce_recording_identity(recording_identity)
    had_pending_live_uploads = any(True for _ in _pending_live_upload_entries(identity))
    pending_ok = drain_live_upload_queue(identity, max_retries=max_retries, base_wait=base_wait)
    local_dataset = load_local_dataset(identity, attach_runtime=False)
    if local_dataset is None:
        if not had_pending_live_uploads:
            console.print(f"[{STYLE_FAIL}]No local dataset found for {identity.display_ref}[/]")
            console.print(f"  Expected at: [{STYLE_PATH}]{get_local_dataset_path(identity)}[/]")
            return False
        return pending_ok
    storage_format = _dataset_storage_format(local_dataset)

    already_uploaded = _remote_dataset_episode_ids(identity)
    if already_uploaded is None:
        already_uploaded = _load_uploaded_episode_ids(identity)
    new_indices = []
    new_episode_ids = set()
    for i, row in enumerate(local_dataset):
        eid = _normalize_episode_id(row["episode_id"])
        if replace or eid not in already_uploaded:
            new_indices.append(i)
            new_episode_ids.add(eid)

    if not new_indices:
        if not replace:
            try:
                api = HfApi()
                hf_repo_id = _identity_hf_repo_id(identity)
                _, _, remote_files = _hf_repo_state(api, hf_repo_id, create=False)
                remote_format = _remote_storage_format_from_files(remote_files)
                conflict_message = _remote_storage_conflict_message(
                    identity.display_ref, hf_repo_id, storage_format, remote_format
                )
                if conflict_message:
                    console.print(f"[{STYLE_FAIL}]{conflict_message}[/]")
                    return False
            except Exception:
                pass
        console.print(f"[{STYLE_INFO}]All episodes already uploaded, nothing to do[/]")
        return pending_ok

    new_dataset = local_dataset.select(new_indices)
    n_new_episodes = len(new_episode_ids)
    if replace:
        console.print(
            f"[{STYLE_INFO}]Replacing remote dataset with {n_new_episodes} local episodes ({len(new_indices)} frames)...[/]"
        )
    else:
        console.print(
            f"[{STYLE_INFO}]Uploading {n_new_episodes} new episodes ({len(new_indices)} frames)...[/]"
        )
    local_ok = _upload_dataset_shard_to_hub(
        identity,
        new_dataset,
        storage_format=storage_format,
        local_root=get_local_dataset_path(identity),
        episode_ids=new_episode_ids,
        replace=replace,
        include_previews=True,
        max_retries=max_retries,
        base_wait=base_wait,
    )
    return pending_ok and local_ok


def minari_export(recording_identity, dataset_name=None, author=None):
    """Export a local HF dataset to Minari format for offline RL."""
    identity = _coerce_recording_identity(recording_identity)
    try:
        import minari
        from minari.data_collector import EpisodeBuffer
    except ImportError:
        console.print(f"[{STYLE_FAIL}]Minari is not installed.[/]")
        console.print(
            f"Tool install: [{STYLE_CMD}]uv tool install gymrec --with 'minari>=0.5.0' --reinstall[/]"
        )
        console.print(f"Repository development: [{STYLE_CMD}]uv sync --extra minari[/]")
        return False

    dataset = load_local_dataset(identity)
    if dataset is None:
        console.print(f"[{STYLE_FAIL}]No local dataset found for {identity.display_ref}[/]")
        console.print(f"  Expected at: [{STYLE_PATH}]{get_local_dataset_path(identity)}[/]")
        return False
    env_id = _recording_env_id(identity, dataset=dataset)
    if not env_id:
        console.print(
            f"[{STYLE_FAIL}]Could not determine the environment for {identity.display_ref}.[/]"
        )
        return False

    episode_keys, row_indices_by_episode = _ordered_episode_rows(dataset)

    # Try to extract action/observation spaces from the environment
    action_space = None
    observation_space = None
    try:
        env = create_env(env_id)
        action_space = env.action_space
        observation_space = env.observation_space
        env.close()
    except Exception:
        console.print("[yellow]Could not create env for space metadata; inferring from data.[/]")

    # Build EpisodeBuffers
    buffers = []
    total_steps = 0
    for ep_idx, episode_key in enumerate(sorted(episode_keys)):
        rows = [dataset[row_index] for row_index in row_indices_by_episode[episode_key]]
        if "step" in rows[0]:
            rows.sort(key=lambda row: row["step"])
        observations = []
        actions = []
        rewards = []
        terminations = []
        truncations = []
        ep_seed = rows[0].get("seed", 0)

        for row in rows:
            img = _get_row_observation(row)
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            observations.append(img)

            # Detect terminal observation by empty actions
            action = row.get("actions")
            if _is_terminal_action(action):
                continue

            if isinstance(action, list) and len(action) == 1:
                action = action[0]
            actions.append(action)

            reward = row.get("rewards")
            rewards.append(float(reward) if reward is not None else 0.0)
            term = row.get("terminations")
            terminations.append(bool(term) if term is not None else False)
            trunc = row.get("truncations")
            truncations.append(bool(trunc) if trunc is not None else False)

        if len(observations) != len(actions) + 1:
            console.print(
                f"[{STYLE_FAIL}]Episode {ep_idx} is malformed: expected N+1 observations "
                f"for {len(actions)} actions, found {len(observations)} observations.[/]"
            )
            return False

        buffers.append(
            EpisodeBuffer(
                id=ep_idx,
                seed=ep_seed,
                observations=observations,
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
            )
        )
        total_steps += len(actions)

    env_id_underscored = env_id.replace("-", "_").replace("/", "_")
    if dataset_name is None:
        dataset_name = f"gymrec/{env_id_underscored}/human-v0"

    # Delete existing dataset with same name if present
    try:
        minari.delete_dataset(dataset_name)
    except Exception:
        pass

    create_kwargs = dict(
        dataset_id=dataset_name,
        buffer=buffers,
        algorithm_name="human",
        description=f"Human gameplay of {env_id}, recorded with gymrec",
    )
    if author:
        create_kwargs["author"] = author
    if action_space is not None:
        create_kwargs["action_space"] = action_space
    if observation_space is not None:
        create_kwargs["observation_space"] = observation_space

    minari.create_dataset_from_buffers(**create_kwargs)

    lines = [
        f"Episodes: [{STYLE_INFO}]{len(buffers)}[/]",
        f"Total steps: [{STYLE_INFO}]{total_steps}[/]",
        f"Dataset ID: [{STYLE_ENV}]{dataset_name}[/]",
        "",
        "Load with:",
        f"  [{STYLE_CMD}]import minari[/]",
        f"  [{STYLE_CMD}]ds = minari.load_dataset('{dataset_name}')[/]",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title="Minari Export Complete",
            border_style="green",
            expand=False,
        )
    )
    return True


def _capture_env_metadata(env):
    """Capture environment configuration metadata for dataset card."""
    env_make_kwargs = dict(getattr(env, "_gymrec_make_kwargs", {}) or {})
    metadata = {
        "env_id": getattr(env, "_env_id", "unknown"),
        "backend": classify_env_id(getattr(env, "_env_id", "")),
        "capture_backend": getattr(
            env,
            "_gymrec_backend",
            classify_env_id(getattr(env, "_env_id", "")),
        ),
        "frameskip": get_frameskip(env),
        "fps": get_default_fps(env),
    }
    if env_make_kwargs:
        metadata["env_make_kwargs"] = env_make_kwargs
    policy_bundle = getattr(env, "_gymrec_policy_bundle", None)
    if isinstance(policy_bundle, dict):
        metadata.update(
            {
                "policy_source_repo": policy_bundle.get("repo_id"),
                "policy_source_revision": policy_bundle.get("revision"),
                "policy_recipe_sha256": policy_bundle.get("recipe_sha256"),
                "policy_checkpoint_sha256": policy_bundle.get("checkpoint_sha256"),
                "policy_recipe_format_version": policy_bundle.get("recipe_format_version"),
                "policy_model_format_version": policy_bundle.get("model_format_version"),
            }
        )

    # Action space info
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        metadata["action_space_type"] = "Discrete"
        metadata["n_actions"] = action_space.n
    elif isinstance(action_space, gym.spaces.MultiBinary):
        metadata["action_space_type"] = "MultiBinary"
        metadata["n_actions"] = action_space.n
    elif isinstance(action_space, gym.spaces.Dict):
        metadata["action_space_type"] = "Dict"
        for key, space in action_space.spaces.items():
            if isinstance(space, gym.spaces.Discrete):
                metadata[f"action_space_{key}"] = f"Discrete({space.n})"
            elif isinstance(space, gym.spaces.MultiBinary):
                metadata[f"action_space_{key}"] = f"MultiBinary({space.n})"
            else:
                metadata[f"action_space_{key}"] = str(space)
    else:
        metadata["action_space_type"] = type(action_space).__name__

    # Observation space shape
    obs_space = env.observation_space
    metadata["observation_space_type"] = type(obs_space).__name__
    obs_shape = getattr(obs_space, "shape", None)
    if obs_shape is not None:
        metadata["observation_shape"] = list(obs_shape)
    if getattr(obs_space, "dtype", None) is not None:
        metadata["observation_dtype"] = str(obs_space.dtype)
    if isinstance(obs_space, gym.spaces.Dict):
        for key, space in obs_space.spaces.items():
            metadata[f"observation_space_{key}"] = str(space)

    # Spec-based metadata
    spec = getattr(env, "spec", None)
    if spec is not None:
        if hasattr(spec, "max_episode_steps") and spec.max_episode_steps is not None:
            metadata["max_episode_steps"] = spec.max_episode_steps
        kwargs = getattr(spec, "kwargs", {}) or {}
        # Sticky actions (ALE)
        if "repeat_action_probability" in kwargs and "sticky_actions" not in metadata:
            metadata["sticky_actions"] = kwargs["repeat_action_probability"]

    if "repeat_action_probability" in env_make_kwargs:
        metadata["sticky_actions"] = env_make_kwargs["repeat_action_probability"]
    elif "sticky_action_prob" in env_make_kwargs:
        metadata["sticky_actions"] = env_make_kwargs["sticky_action_prob"]

    state_name = getattr(env, "_gymrec_state_name", None)
    if state_name:
        metadata["state_name"] = state_name
    rom_sha256 = getattr(env, "_gymrec_rom_sha256", None)
    if rom_sha256:
        metadata["rom_sha256"] = rom_sha256
    if _is_nes_controller_env(env):
        metadata.setdefault("sticky_actions", 0.0)
        metadata["nes_button_order"] = list(NES_BUTTON_ORDER)
        metadata["action_encoding"] = NES_ACTION_ENCODING
        metadata["frame_stack"] = int(env_make_kwargs.get("frame_stack", 1))
        metadata["noop_reset_max"] = int(env_make_kwargs.get("noop_reset_max", 0))

    # ALE-specific: check unwrapped for sticky actions
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None:
        sticky = getattr(unwrapped, "_repeat_action_probability", None)
        if sticky is not None and "sticky_actions" not in metadata:
            metadata["sticky_actions"] = sticky

    # Reward range
    if hasattr(env, "reward_range"):
        metadata["reward_range"] = list(env.reward_range)

    # Stable-Retro specific
    if hasattr(env, "_stable_retro") and env._stable_retro:
        stable_retro_state = getattr(env, "_gymrec_stable_retro_state", None)
        if stable_retro_state:
            metadata["stable_retro_state"] = stable_retro_state
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            metadata["retro_platform"] = getattr(unwrapped, "system", None)
            metadata["retro_game"] = getattr(unwrapped, "gamerom", None)
            buttons = getattr(unwrapped, "buttons", None)
            if buttons:
                metadata["retro_buttons"] = list(buttons)

    # VizDoom specific
    if hasattr(env, "_vizdoom") and env._vizdoom:
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None:
            metadata["vizdoom_scenario"] = getattr(unwrapped, "scenario", None)
            metadata["vizdoom_num_binary_buttons"] = getattr(unwrapped, "num_binary_buttons", None)
            metadata["vizdoom_num_delta_buttons"] = getattr(unwrapped, "num_delta_buttons", None)

    return metadata


def _size_category(n):
    """Return HF size_categories string for a frame count."""
    if n < 1000:
        return "n<1K"
    if n < 10000:
        return "1K<n<10K"
    if n < 100000:
        return "10K<n<100K"
    if n < 1000000:
        return "100K<n<1M"
    return "n>1M"


def _current_hf_username():
    user_info = whoami()
    return user_info.get("name") or user_info.get("user") or user_info.get("username") or "unknown"


def _provenance_from_metadata(metadata):
    collectors = set()
    versions = set()
    if metadata and "recordings" in metadata:
        for recording in metadata["recordings"]:
            collectors.update(recording.get("collectors", []))
            versions.update(recording.get("gymrec_versions", []))
    return sorted(collectors), sorted(versions)


def _dataset_card_intro(env_id, collectors):
    if collectors and collectors != ["human"]:
        collector_str = ", ".join(f"`{c}`" for c in collectors)
        return (
            f"Gameplay recordings (collected by: {collector_str}) from the Gymnasium "
            f"environment `{env_id}`,"
        )
    return f"Human gameplay recordings from the Gymnasium environment `{env_id}`,"


def _dataset_card_environment_lines(metadata, backend):
    if not metadata:
        return []

    lines = [
        "## Environment Configuration",
        "",
        "| Setting | Value |",
        "|---------|-------|",
    ]

    if metadata.get("capture_backend"):
        lines.append(f"| Capture Backend | {metadata['capture_backend']} |")
    if metadata.get("state_name"):
        lines.append(f"| State | {metadata['state_name']} |")
    if metadata.get("rom_sha256"):
        lines.append(f"| ROM SHA-256 | `{metadata['rom_sha256']}` |")
    if metadata.get("action_encoding"):
        lines.append(f"| Action Encoding | {metadata['action_encoding']} |")
    if metadata.get("nes_button_order"):
        lines.append(f"| NES Button Order | `{json.dumps(metadata['nes_button_order'])}` |")
    if metadata.get("policy_source_repo"):
        lines.append(f"| Policy Source | `{metadata['policy_source_repo']}` |")
    if metadata.get("policy_source_revision"):
        lines.append(f"| Policy Revision | `{metadata['policy_source_revision']}` |")
    if metadata.get("policy_recipe_sha256"):
        lines.append(f"| Policy Recipe SHA-256 | `{metadata['policy_recipe_sha256']}` |")
    if metadata.get("policy_checkpoint_sha256"):
        lines.append(f"| Policy Checkpoint SHA-256 | `{metadata['policy_checkpoint_sha256']}` |")

    for key, label in (
        ("frameskip", "Frameskip"),
        ("fps", "Target FPS"),
        ("sticky_actions", "Sticky Actions"),
        ("max_episode_steps", "Max Episode Steps"),
    ):
        if key in metadata:
            lines.append(f"| {label} | {metadata[key]} |")

    if "observation_shape" in metadata:
        shape = metadata["observation_shape"]
        lines.append(f"| Observation Shape | {' x '.join(str(s) for s in shape)} |")
    if "observation_dtype" in metadata:
        lines.append(f"| Observation Dtype | {metadata['observation_dtype']} |")
    if "action_space_type" in metadata:
        lines.append(f"| Action Space | {metadata['action_space_type']} |")
    if "n_actions" in metadata:
        lines.append(f"| Number of Actions | {metadata['n_actions']} |")
    if "reward_range" in metadata:
        rmin, rmax = metadata["reward_range"]
        lines.append(f"| Reward Range | [{rmin}, {rmax}] |")

    if backend in {"stable-retro", "stable-retro-turbo", "supermariobrosnes-turbo"}:
        if metadata.get("retro_platform"):
            lines.append(f"| Platform | {metadata['retro_platform']} |")
        if metadata.get("retro_game"):
            lines.append(f"| Game | {metadata['retro_game']} |")
        if metadata.get("retro_buttons"):
            named = [b for b in metadata["retro_buttons"][:8] if b is not None]
            buttons = ", ".join(named)
            total_named = sum(1 for b in metadata["retro_buttons"] if b is not None)
            if total_named > 8:
                buttons += f" (+{total_named - 8} more)"
            lines.append(f"| Buttons | {buttons} |")
    elif backend == "vizdoom":
        if metadata.get("vizdoom_scenario"):
            lines.append(f"| Scenario | {metadata['vizdoom_scenario']} |")
        if "vizdoom_num_binary_buttons" in metadata:
            lines.append(f"| Binary Buttons | {metadata['vizdoom_num_binary_buttons']} |")
        if "vizdoom_num_delta_buttons" in metadata:
            lines.append(f"| Delta Buttons | {metadata['vizdoom_num_delta_buttons']} |")

    lines.append("")
    return lines


def render_dataset_card_content(
    env_id,
    repo_id,
    frames,
    episodes,
    metadata=None,
    collectors=None,
    gymrec_versions=None,
    curator=None,
):
    """Render the shared Hugging Face dataset card Markdown."""
    backend = (metadata or {}).get("capture_backend") or classify_env_id(env_id)
    collectors = collectors or []
    gymrec_versions = gymrec_versions or []
    curator = curator or _current_hf_username()
    storage_format = (
        _normalize_storage_format(metadata.get("storage_format"))
        if metadata and metadata.get("storage_format")
        else STORAGE_FORMAT_IMAGES
    )
    card_data = DatasetCardData(
        language="en",
        license=CONFIG["dataset"]["license"],
        task_categories=CONFIG["dataset"]["task_categories"],
        tags=["gymnasium", backend, env_id],
        size_categories=[_size_category(frames)],
        pretty_name=f"{env_id} Gameplay Dataset",
    )

    content_lines = [
        "---",
        card_data.to_yaml(),
        "---",
        "",
        f"# {env_id} Gameplay Dataset",
        "",
        _dataset_card_intro(env_id, collectors),
        "captured using [gymrec](https://github.com/tsilva/gymrec).",
        "",
        "## Dataset Summary",
        "",
        "| Stat | Value |",
        "|------|-------|",
        f"| Total frames | {frames:,} |",
        f"| Episodes | {episodes:,} |",
        f"| Environment | `{env_id}` |",
        f"| Backend | {BACKEND_LABELS.get(backend, backend)} |",
        f"| Storage format | `{storage_format}` |",
    ]
    if collectors:
        content_lines.append(f"| Collector(s) | {', '.join(collectors)} |")
    if gymrec_versions:
        content_lines.append(f"| gymrec version(s) | {', '.join(gymrec_versions)} |")
    content_lines.append("")

    if storage_format == STORAGE_FORMAT_LOSSLESS_VIDEO:
        observation_lines = [
            f"- **video_path** (`string`): Relative path to the canonical lossless RGB observation stream under `{VIDEO_ARTIFACT_DIR}/`, named `*{CANONICAL_VIDEO_SUFFIX}`",
            "- **frame_index** (`int`): Observation frame index within `video_path`",
            "- **frame_sha256** (`string`): SHA-256 of the raw RGB frame bytes, verified after video encode/decode",
            "- **frame_width** / **frame_height** (`int`): Decoded RGB frame dimensions",
            "- **episode_num_observations** (`int`): Number of observations in the episode video",
            f"- Browser-friendly `*{PREVIEW_VIDEO_SUFFIX}` files under `{PREVIEW_VIDEO_ARTIFACT_DIR}/` are lossy previews only and are not used for trajectory replay/training",
        ]
    else:
        observation_lines = [
            "- **observations** (`Image`): RGB frame from the environment",
        ]

    content_lines.extend(_dataset_card_environment_lines(metadata, backend))
    content_lines.extend(
        [
            "## Dataset Structure",
            "",
            "Minari-compatible flat table format. Use `minari-export` for native [Minari](https://minari.farama.org/) HDF5 format.",
            "",
            "Each episode has N step rows plus one terminal observation row (N+1 pattern).",
            "The terminal observation is the final state after the last step - it has an empty action",
            "and null values for rewards/terminations/truncations/infos.",
            "",
            "- **episode_id** (`binary(16)`): Unique UUID identifier for each episode (16 bytes, universally unique across all recordings)",
            "- **seed** (`int` or `null`): RNG seed used for `env.reset()` (set on first row of each episode, `null` on other rows)",
            *observation_lines,
            "- **actions** (`list`): Action taken at this step (`[]` for terminal observations)",
            "- **rewards** (`float` or `null`): Reward received (`null` on terminal observation rows)",
            "- **terminations** (`bool` or `null`): Whether the episode terminated naturally (`null` on terminal observation rows)",
            "- **truncations** (`bool` or `null`): Whether the episode was truncated (`null` on terminal observation rows)",
            "- **infos** (`str` or `null`): Additional environment info as JSON (`null` on terminal observation rows)",
            "- **session_id** (`binary(16)`): UUID grouping all episodes from one `gymrec record` run",
            '- **collector** (`string`): Who collected the data (`"human"`, `"random"`, or future agent names)',
            '- **gymrec_version** (`string`): Version of gymrec used to record (e.g. `"0.1.0+abc1234"`)',
            "- **storage_format** (`string`): Observation storage backend (`images` or `lossless-video`)",
            "- **policy_source_repo** / **policy_source_revision** (`string` or `null`): Hugging Face policy bundle identity for agent-collected rows",
            "- **policy_recipe_sha256** / **policy_checkpoint_sha256** (`string` or `null`): Validated rlab bundle content hashes",
            "- **policy_recipe_format_version** / **policy_model_format_version** (`int` or `null`): Consumed rlab document versions",
            "",
            "## Usage",
            "",
            "```python",
            "from datasets import load_dataset",
            f'ds = load_dataset("{repo_id}")',
            "```",
            "",
            "## About",
            "",
            "Recorded with [gymrec](https://github.com/tsilva/gymrec).",
            f"Curated by: {curator}",
        ]
    )
    return "\n".join(content_lines)


def _read_existing_dataset_card_counts(repo_id):
    """Return frame and episode counts parsed from an existing Hub README."""
    import re

    try:
        readme_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="README.md",
        )
        with open(readme_path) as f:
            readme_content = f.read()
        frames_match = re.search(r"\|\s*Total frames\s*\|\s*([\d,]+)\s*\|", readme_content)
        episodes_match = re.search(r"\|\s*Episodes\s*\|\s*([\d,]+)\s*\|", readme_content)
        frames = int(frames_match.group(1).replace(",", "")) if frames_match else 0
        episodes = int(episodes_match.group(1).replace(",", "")) if episodes_match else 0
        return frames, episodes
    except Exception:
        return 0, 0


def _build_dataset_card_content(
    recording_identity,
    env_id,
    repo_id,
    new_frames,
    new_episodes,
    repo_exists,
):
    """Build dataset card content string for an append-only upload.

    If the repo exists, downloads the current README and parses existing frame/episode
    counts to compute running totals. Otherwise starts from new_frames/new_episodes.
    Returns the full card content string (does not push to Hub).
    """
    total_frames = new_frames
    total_episodes = new_episodes

    if repo_exists:
        existing_frames, existing_episodes = _read_existing_dataset_card_counts(repo_id)
        total_frames += existing_frames
        total_episodes += existing_episodes

    metadata = load_local_metadata(recording_identity)
    collectors, gymrec_versions = _provenance_from_metadata(metadata)
    return render_dataset_card_content(
        env_id,
        repo_id,
        frames=total_frames,
        episodes=total_episodes,
        metadata=metadata,
        collectors=collectors,
        gymrec_versions=gymrec_versions,
    )


def _is_unexpected_env_kwarg_error(exc):
    text = str(exc).lower()
    return "unexpected keyword" in text or "got an unexpected" in text


def _warn_unsupported_env_kwargs(env_id, kwargs):
    if not kwargs:
        return
    keys = ", ".join(sorted(kwargs))
    console.print(
        f"[yellow]Warning:[/] {env_id} does not accept env kwargs ({keys}); using backend defaults."
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_supermariobrosnes_rom_path(roms_path=None):
    """Resolve the canonical SMB NES ROM from a file or directory by SHA-256."""
    source = roms_path or _get_roms_path()
    if not source:
        raise FileNotFoundError(
            "SuperMarioBros-Nes-turbo requires ROMS_PATH or --roms-path to point to "
            "the canonical Super Mario Bros NES ROM or a directory containing it."
        )

    source = os.path.abspath(os.path.expanduser(os.fspath(source)))
    if os.path.isfile(source):
        candidates = [source]
    elif os.path.isdir(source):
        candidates = [
            os.path.join(root, filename)
            for root, _dirs, files in os.walk(source)
            for filename in sorted(files)
            if filename.lower().endswith(".nes")
        ]
    else:
        raise FileNotFoundError(f"ROMS_PATH does not exist: {source}")

    for candidate in candidates:
        if _sha256_file(candidate) == SUPERMARIOBROS_NES_ROM_SHA256:
            return candidate

    if os.path.isfile(source):
        detail = "the selected file has the wrong SHA-256"
    elif not candidates:
        detail = "the directory contains no .nes files"
    else:
        detail = f"none of the {len(candidates)} .nes files match the canonical SHA-256"
    raise ValueError(f"Could not resolve the canonical Super Mario Bros NES ROM: {detail}.")


def _lane_zero_value(value):
    if isinstance(value, dict):
        return _vector_info_to_lane_info(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        value = value[0]
    elif isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _vector_info_to_lane_info(vector_info):
    """Convert Gymnasium vector-info dictionaries into an ordinary lane-zero dict."""
    if not vector_info:
        return {}
    lane_info = {}
    for key, value in vector_info.items():
        if key.startswith("_"):
            continue
        mask = vector_info.get(f"_{key}")
        if mask is not None and not bool(_lane_zero_value(mask)):
            continue
        lane_info[key] = _lane_zero_value(value)
    return lane_info


class SuperMarioBrosNesTurboEnvAdapter(gym.Env):
    """Expose a one-lane native Retro VectorEnv as an ordinary Gymnasium Env."""

    def __init__(self, vector_env):
        _lazy_init()
        if getattr(vector_env, "num_envs", None) != 1:
            raise ValueError("The gymrec turbo adapter requires num_envs=1")
        self.vector_env = vector_env
        self.action_space = vector_env.single_action_space
        self.observation_space = vector_env.single_observation_space
        self.metadata = dict(getattr(vector_env, "metadata", {}) or {})
        self.render_mode = getattr(vector_env, "render_mode", "rgb_array")
        self.autoreset_mode = getattr(vector_env, "autoreset_mode", None)
        self.system = getattr(vector_env, "system", "Nes")
        self.buttons = list(getattr(vector_env, "buttons", NES_BUTTON_ORDER))
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observations, infos = self.vector_env.reset(seed=seed, options=options)
        self._needs_reset = False
        return observations[0], _vector_info_to_lane_info(infos)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset() must be called before step() or after a terminal step")
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = int(np.asarray(action).reshape(-1)[0])
            if not self.action_space.contains(action):
                raise ValueError(f"Expected action in {self.action_space}; got {action}")
            batched_action = np.asarray([action], dtype=self.action_space.dtype)
        else:
            action = np.asarray(action, dtype=self.action_space.dtype)
            if action.shape != self.action_space.shape or not self.action_space.contains(action):
                raise ValueError(
                    f"Expected a Stable-Retro-compatible action with shape "
                    f"{self.action_space.shape}; got {action.shape}"
                )
            batched_action = action[np.newaxis, ...]
        observations, rewards, terminations, truncations, infos = self.vector_env.step(
            batched_action
        )
        terminated = bool(terminations[0])
        truncated = bool(truncations[0])
        if terminated or truncated:
            self._needs_reset = True
        return (
            observations[0],
            float(rewards[0]),
            terminated,
            truncated,
            _vector_info_to_lane_info(infos),
        )

    def render(self):
        return self.vector_env.render()

    def close(self):
        self.vector_env.close()


class MarioRecipeTaskEnv(gym.Wrapper):
    """Apply the declarative rlab Mario reward and episode contract to one lane."""

    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        self.signals = task.get("signals") or {}
        self.events = task.get("events") or {}
        self.termination = task.get("termination") or {}
        self.reward_config = task.get("reward") or {}
        self._validate_contract()
        self._gymrec_policy_observations = bool(getattr(env, "_gymrec_policy_observations", False))
        self._episode_steps = 0
        self._last_progress_step = 0

    def _validate_contract(self):
        expected_events = {
            "life_loss": ("lives", "decrease"),
            "level_change": ("level", "change"),
            "stalled": ("x", "unchanged_for"),
        }
        unknown = sorted(set(self.events) - set(expected_events))
        if unknown:
            raise ValueError(f"Unsupported Mario recipe event(s): {', '.join(unknown)}")
        for name, expected in expected_events.items():
            if name not in self.events:
                continue
            rule = self.events[name]
            actual = (rule.get("signal"), rule.get("operation"))
            if actual != expected:
                raise ValueError(
                    f"Mario recipe event {name!r} requires signal={expected[0]!r}, "
                    f"operation={expected[1]!r}"
                )
        reward_mode = self.reward_config.get("reward_mode", "native")
        if reward_mode not in {"native", "bounded", "baseline", "score", "additive"}:
            raise ValueError(f"Unsupported Mario recipe reward mode {reward_mode!r}")

    @staticmethod
    def _signal(info, source, *, pair=False):
        names = (source,) if isinstance(source, str) else tuple(source)
        try:
            values = tuple(int(info[name]) for name in names)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Mario recipe signal {source!r} is absent from provider info"
            ) from exc
        if pair:
            if len(values) != 2:
                raise ValueError(f"Mario recipe pair signal {source!r} must contain two fields")
            return values
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return values[0] * 256 + values[1]
        raise ValueError(f"Mario recipe signal {source!r} has unsupported width {len(values)}")

    def _read_state(self, info):
        return {
            "x": self._signal(info, self.signals.get("x", ("xscrollHi", "xscrollLo"))),
            "score": self._signal(info, self.signals.get("score", "score")),
            "lives": self._signal(info, self.signals.get("lives", "lives")),
            "level": self._signal(
                info, self.signals.get("level", ("levelHi", "levelLo")), pair=True
            ),
        }

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        state = self._read_state(info)
        self._x = state["x"]
        self._level_max_x = state["x"]
        self._completed_level_base = 0
        self._max_global_x = state["x"]
        self._score = state["score"]
        self._lives = state["lives"]
        self._level = state["level"]
        self._episode_steps = 0
        self._last_progress_step = 0
        return observation, info

    def step(self, action):
        observation, native_reward, provider_terminated, provider_truncated, info = self.env.step(
            action
        )
        info = dict(info)
        state = self._read_state(info)
        lost_life = state["lives"] < self._lives
        changed_level = state["level"] != self._level
        completed = changed_level and not lost_life

        if completed:
            self._completed_level_base += self._level_max_x
            self._level_max_x = 0
        effective_x = 0 if changed_level else state["x"]
        self._level_max_x = max(self._level_max_x, effective_x)
        global_x = self._completed_level_base + effective_x
        global_max_x = self._completed_level_base + self._level_max_x
        progress_delta = max(global_max_x - self._max_global_x, 0)
        self._max_global_x = max(self._max_global_x, global_max_x)
        score_delta = max(state["score"] - self._score, 0)

        no_progress_min_delta = int(self.termination.get("no_progress_min_delta", 0))
        if progress_delta > no_progress_min_delta:
            self._last_progress_step = self._episode_steps
        self._episode_steps += 1
        stalled_rule = self.events.get("stalled") or {}
        stalled_steps = int(stalled_rule.get("steps", 0))
        stalled = bool(
            stalled_steps > 0 and self._episode_steps - self._last_progress_step >= stalled_steps
        )

        failure_events = set(self.termination.get("failure") or ())
        success_events = set(self.termination.get("success") or ())
        task_terminated = (
            (lost_life and "life_loss" in failure_events)
            or (completed and "level_change" in success_events)
            or (stalled and "stalled" in failure_events)
        )
        max_episode_steps = int(self.termination.get("max_episode_steps", 0))
        task_truncated = (stalled and "stalled" not in failure_events) or (
            max_episode_steps > 0 and self._episode_steps >= max_episode_steps
        )
        if task_terminated:
            task_truncated = False

        terminated = bool(provider_terminated or task_terminated)
        truncated = bool((provider_truncated or task_truncated) and not terminated)
        reward = self._shape_reward(
            float(native_reward),
            progress_delta=progress_delta,
            score_delta=score_delta,
            completed=completed,
            lost_life=lost_life,
            done=terminated or truncated,
        )

        emitted_events = []
        if lost_life and "life_loss" in self.events:
            emitted_events.append("life_loss")
        if changed_level and "level_change" in self.events:
            emitted_events.append("level_change")
        if stalled and "stalled" in self.events:
            emitted_events.append("stalled")
        outcome = "neutral"
        if truncated:
            outcome = "timeout"
        if completed and not lost_life:
            outcome = "success"
        if (
            lost_life
            or (stalled and "stalled" in failure_events)
            or (provider_terminated and not completed)
        ):
            outcome = "failure"
        info.update(
            {
                "task_events": emitted_events,
                "task_outcome": outcome,
                "level_complete": completed,
                "life_loss": lost_life,
                "progress_delta": progress_delta,
                "score_delta": score_delta,
                "global_x_pos": global_x,
                "global_max_x_pos": self._max_global_x,
                "native_reward": float(native_reward),
                "shaped_reward": reward,
            }
        )

        self._x = effective_x
        self._score = state["score"]
        self._lives = state["lives"]
        self._level = state["level"]
        return observation, reward, terminated, truncated, info

    def _shape_reward(
        self,
        native_reward,
        *,
        progress_delta,
        score_delta,
        completed,
        lost_life,
        done,
    ):
        config = self.reward_config
        mode = str(config.get("reward_mode", "native"))
        reward_scale = float(config.get("reward_scale", 10.0)) or 1.0
        terminal_reward = float(config.get("terminal_reward", 50.0))
        progress_cap = float(config.get("progress_reward_cap", 30.0))
        capped_progress = min(float(progress_delta), progress_cap)
        if mode == "native":
            shaped = native_reward
        elif mode == "bounded":
            raw = (
                terminal_reward if completed else -terminal_reward if lost_life else capped_progress
            )
            shaped = raw / reward_scale
        elif mode == "baseline":
            raw = native_reward + float(score_delta) / 40.0
            if completed:
                raw += terminal_reward
            elif done:
                raw -= terminal_reward
            shaped = raw / reward_scale
        else:
            native_component = native_reward if config.get("use_native_reward", False) else 0.0
            if mode == "score":
                progress = (
                    capped_progress
                    if config.get("score_progress_clipped", False)
                    else float(progress_delta)
                )
                score_component = 0.01 * float(score_delta)
            else:
                progress = float(progress_delta)
                score_component = 0.0
            shaped = (
                native_component
                + float(config.get("progress_reward_scale", 1.0)) * progress
                + score_component
                + (float(config.get("completion_reward", 0.0)) if completed else 0.0)
                - (float(config.get("death_penalty", 25.0)) if lost_life else 0.0)
            )
        shaped -= float(config.get("time_penalty", 0.0))
        if config.get("clip_rewards", False):
            shaped = 1.0 if shaped > 0 else -1.0 if shaped < 0 else 0.0
        return float(shaped)


def _create_env__stableretro(env_id, state=None, env_make_kwargs=None):
    _ensure_stableretro_roms_path_imported()

    import stable_retro as retro

    make_kwargs = {"render_mode": "rgb_array"}
    if state:
        make_kwargs["state"] = state
    optional_kwargs = dict(env_make_kwargs or {})
    try:
        env = retro.make(env_id, **make_kwargs, **optional_kwargs)
        env._gymrec_make_kwargs = optional_kwargs
    except FileNotFoundError:
        console.print(f"\n[{STYLE_FAIL}]Error: ROM not found for '{env_id}'.[/]")
        console.print("\nStable-retro requires ROM files to be imported separately.")
        console.print(
            f"Import ROMs with:  [{STYLE_CMD}]{_gymrec_cmd('import_roms', '/path/to/roms')}[/]"
        )
        console.print(
            f"\nUse [{STYLE_CMD}]{_gymrec_cmd('list_environments')}[/] to see which games have ROMs imported."
        )
        sys.exit(1)
    except TypeError as exc:
        if not optional_kwargs or not _is_unexpected_env_kwarg_error(exc):
            raise
        _warn_unsupported_env_kwargs(env_id, optional_kwargs)
        env = retro.make(env_id, **make_kwargs)
        env._gymrec_make_kwargs = {}
    env._stable_retro = True
    env._gymrec_backend = "stable-retro"
    env._gymrec_stable_retro_state = state
    env._gymrec_state_name = state
    env._gymrec_nes_controller = getattr(getattr(env, "unwrapped", None), "system", None) == "Nes"
    try:
        rom_path = retro.data.get_romfile_path(env_id, retro.data.Integrations.ALL)
        env._gymrec_rom_sha256 = _sha256_file(rom_path)
    except (FileNotFoundError, OSError, AttributeError):
        pass
    return env


def _create_env__supermariobrosnes_turbo(env_id, state=None, env_make_kwargs=None):
    if env_id != SUPERMARIOBROS_NES_ENV_ID:
        raise ValueError(
            f"The supermariobrosnes-turbo backend only supports {SUPERMARIOBROS_NES_ENV_ID}."
        )

    from supermariobrosnes_turbo import Actions, SuperMarioBrosNesTurboVecEnv

    state = state or "Level1-1"
    optional_kwargs = dict(env_make_kwargs or {})
    rom_path = optional_kwargs.get("rom_path") or resolve_supermariobrosnes_rom_path()
    optional_kwargs.update(
        {
            "num_envs": 1,
            "rom_path": rom_path,
            "state": state,
        }
    )
    optional_kwargs.setdefault("num_threads", 1)
    optional_kwargs.setdefault("render_mode", "rgb_array")
    optional_kwargs.setdefault("use_restricted_actions", Actions.ALL)
    optional_kwargs.setdefault("obs_layout", "hwc")
    optional_kwargs.setdefault("frame_stack", 1)
    optional_kwargs.setdefault("frame_skip", 1)
    optional_kwargs.setdefault("sticky_action_prob", 0.0)
    optional_kwargs.setdefault("noop_reset_max", 0)
    optional_kwargs.setdefault("info_filter", "all")
    vector_env = SuperMarioBrosNesTurboVecEnv(
        SUPERMARIOBROS_NES_ENV_ID,
        **optional_kwargs,
    )
    env = SuperMarioBrosNesTurboEnvAdapter(vector_env)
    env._gymrec_backend = "supermariobrosnes-turbo"
    env._gymrec_nes_controller = True
    env._gymrec_state_name = state
    env._gymrec_rom_sha256 = SUPERMARIOBROS_NES_ROM_SHA256
    env._gymrec_policy_observations = bool(
        optional_kwargs.get("obs_grayscale")
        or int(optional_kwargs.get("frame_stack", 1)) > 1
        or optional_kwargs.get("obs_resize") is not None
        or optional_kwargs.get("obs_crop") is not None
        or optional_kwargs.get("obs_layout") != "hwc"
    )
    env._gymrec_make_kwargs = {
        key: value
        for key, value in optional_kwargs.items()
        if key not in {"num_envs", "rom_path", "state"}
    }
    return env


def _create_env__stable_retro_turbo(env_id, state=None, env_make_kwargs=None):
    """Create the one-lane native RetroVecEnv declared by an rlab recipe."""
    _ensure_stableretro_roms_path_imported()
    import stable_retro as retro

    optional_kwargs = dict(env_make_kwargs or {})
    optional_kwargs.update({"num_envs": 1, "state": state or retro.State.DEFAULT})
    optional_kwargs.setdefault("num_threads", 1)
    optional_kwargs.setdefault("render_mode", "rgb_array")
    optional_kwargs.setdefault("use_restricted_actions", retro.Actions.ALL)
    optional_kwargs.setdefault("obs_layout", "hwc")
    optional_kwargs.setdefault("frame_stack", 1)
    optional_kwargs.setdefault("frame_skip", 1)
    optional_kwargs.setdefault("sticky_action_prob", 0.0)
    optional_kwargs.setdefault("noop_reset_max", 0)
    optional_kwargs.setdefault("info_filter", "all")
    vector_env = retro.RetroVecEnv(env_id, **optional_kwargs)
    env = SuperMarioBrosNesTurboEnvAdapter(vector_env)
    env._stable_retro = True
    env._gymrec_backend = "stable-retro-turbo"
    env._gymrec_state_name = state
    env._gymrec_nes_controller = getattr(env, "system", None) == "Nes"
    env._gymrec_policy_observations = bool(
        optional_kwargs.get("obs_grayscale")
        or int(optional_kwargs.get("frame_stack", 1)) > 1
        or optional_kwargs.get("obs_resize") is not None
        or optional_kwargs.get("obs_crop") is not None
        or optional_kwargs.get("obs_layout") != "hwc"
    )
    env._gymrec_make_kwargs = {
        key: value
        for key, value in optional_kwargs.items()
        if key not in {"num_envs", "rom_path", "state"}
    }
    try:
        rom_path = retro.data.get_romfile_path(env_id, retro.data.Integrations.ALL)
        env._gymrec_rom_sha256 = _sha256_file(rom_path)
    except (FileNotFoundError, OSError, AttributeError):
        pass
    return env


def _create_env__vizdoom(env_id):
    import vizdoom.gymnasium_wrapper  # noqa: F401

    env = gym.make(env_id, render_mode="rgb_array", max_buttons_pressed=0)
    env._vizdoom = True
    return env


def _create_env__alepy(env_id, env_make_kwargs=None):
    _configure_atari_roms_path()

    import ale_py

    gym.register_envs(ale_py)
    has_rom, rom_status = _get_atari_rom_status(env_id)
    if not has_rom:
        game = _get_atari_game_name(env_id) or env_id
        console.print(f"\n[{STYLE_FAIL}]Error: Atari ROM not available for '{env_id}'.[/]")
        console.print(
            f"\nALE-py registered this environment, but the ROM file for [{STYLE_ENV}]{game}[/] is not usable in this Python environment ([{STYLE_FAIL}]{rom_status}[/])."
        )
        console.print(
            f"Use [{STYLE_CMD}]{_gymrec_cmd('list_environments')}[/] to see which Atari environments have ROMs installed."
        )
        sys.exit(1)

    optional_kwargs = dict(env_make_kwargs or {})
    try:
        env = gym.make(env_id, render_mode="rgb_array", **optional_kwargs)
        env._gymrec_make_kwargs = optional_kwargs
        return env
    except TypeError as exc:
        if not optional_kwargs or not _is_unexpected_env_kwarg_error(exc):
            raise
        _warn_unsupported_env_kwargs(env_id, optional_kwargs)
        env = gym.make(env_id, render_mode="rgb_array")
        env._gymrec_make_kwargs = {}
        return env
    except FileNotFoundError:
        game = _get_atari_game_name(env_id) or env_id
        console.print(f"\n[{STYLE_FAIL}]Error: Atari ROM not found for '{env_id}'.[/]")
        console.print(
            f"\nALE-py registered this environment, but the ROM file for [{STYLE_ENV}]{game}[/] is not installed in this Python environment."
        )
        console.print(
            f"Use [{STYLE_CMD}]{_gymrec_cmd('list_environments')}[/] to see which Atari environments have ROMs installed."
        )
        sys.exit(1)
    except OSError as e:
        console.print(f"\n[{STYLE_FAIL}]Error: Atari ROM failed validation for '{env_id}'.[/]")
        console.print(f"[dim]{e}[/]")
        console.print(
            f"Use [{STYLE_CMD}]{_gymrec_cmd('list_environments')}[/] to see which Atari environments have valid ROMs installed."
        )
        sys.exit(1)


def create_env(
    env_id,
    *,
    stable_retro_state=None,
    backend=None,
    human_recording=False,
    metadata=None,
    env_make_kwargs=None,
):
    """Create a Gymnasium environment with the appropriate backend."""

    if backend is None and not (metadata or {}).get("capture_backend"):
        stable_retro_envs = set(_get_stableretro_envs())
        backend = classify_env_id(env_id, stable_retro_envs=stable_retro_envs)
    backend = resolve_runtime_backend(env_id, requested=backend, metadata=metadata)
    if env_make_kwargs is None:
        if human_recording:
            env_make_kwargs = _human_record_env_make_kwargs(backend)
        else:
            env_make_kwargs = _env_make_kwargs_from_metadata(env_id, metadata, backend=backend)
    else:
        env_make_kwargs = dict(env_make_kwargs)
    if (
        backend
        in {
            "stable-retro",
            "stable-retro-turbo",
            "supermariobrosnes-turbo",
        }
        and stable_retro_state is None
    ):
        stable_retro_state = _stable_retro_state_from_metadata(metadata)
    if env_id == SUPERMARIOBROS_NES_ENV_ID and stable_retro_state is None:
        stable_retro_state = "Level1-1"

    if backend == "stable-retro":
        env = _create_env__stableretro(
            env_id,
            state=stable_retro_state,
            env_make_kwargs=env_make_kwargs,
        )
    elif backend == "stable-retro-turbo":
        env = _create_env__stable_retro_turbo(
            env_id,
            state=stable_retro_state,
            env_make_kwargs=env_make_kwargs,
        )
    elif backend == "supermariobrosnes-turbo":
        env = _create_env__supermariobrosnes_turbo(
            env_id,
            state=stable_retro_state,
            env_make_kwargs=env_make_kwargs,
        )
    elif backend == "vizdoom":
        env = _create_env__vizdoom(env_id)
        env._gymrec_make_kwargs = {}
    else:
        env = _create_env__alepy(env_id, env_make_kwargs=env_make_kwargs)

    env._env_id = env_id
    env._gymrec_backend = backend
    return env


def _normalize_frameskip_value(value):
    """Normalize a scalar or two-value frameskip to a positive integer."""
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            value = (value[0] + value[1]) / 2
        return max(int(round(value)), 1)
    except (TypeError, ValueError):
        return None


def get_frameskip(env) -> int:
    """Detect the frameskip value for an environment.

    Returns the number of internal frames per env.step() call.
    For stochastic frameskip tuples like (2, 5), returns the average.
    """
    env_id = getattr(env, "_env_id", "")
    make_kwargs = getattr(env, "_gymrec_make_kwargs", {}) or {}
    for key in ("frameskip", "frame_skip"):
        frameskip = _normalize_frameskip_value(make_kwargs.get(key))
        if frameskip is not None:
            return frameskip

    # VizDoom and Retro have different frameskip semantics; skip detection
    if hasattr(env, "_vizdoom") and env._vizdoom:
        return 1
    if bool(getattr(env, "_stable_retro", False)) or _is_nes_controller_env(env):
        return 1

    # Check spec kwargs (works for gym.make() environments)
    spec = getattr(env, "spec", None)
    if spec is not None:
        kwargs = getattr(spec, "kwargs", {}) or {}
        frameskip = _normalize_frameskip_value(kwargs.get("frameskip"))
        if frameskip is not None:
            return frameskip

    # ALE-specific attribute fallback
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None:
        frameskip = _normalize_frameskip_value(getattr(unwrapped, "_frameskip", None))
        if frameskip is not None:
            return frameskip

    # Name-based: NoFrameskip means frameskip=1
    if "NoFrameskip" in env_id:
        return 1

    return 1


def get_default_fps(env):
    """Determine a sensible default FPS for an environment."""

    env_id = getattr(env, "_env_id", "")
    backend = classify_env_id(env_id)
    base_fps = CONFIG["fps_defaults"]["atari"] if backend == "atari" else None

    for key in ("render_fps", "video.frames_per_second"):
        if base_fps is not None:
            break
        fps = env.metadata.get(key)
        if fps:
            try:
                base_fps = int(round(float(fps)))
                break
            except (TypeError, ValueError):
                pass

    if base_fps is None:
        if backend == "stable-retro":
            base_fps = CONFIG["fps_defaults"]["retro"]
        elif backend == "vizdoom":
            base_fps = CONFIG["fps_defaults"]["vizdoom"]
        else:
            base_fps = CONFIG["fps_defaults"]["atari"]

    frameskip = get_frameskip(env)
    if frameskip > 1:
        adjusted_fps = max(int(round(base_fps / frameskip)), 1)
        console.print(
            f"[{STYLE_INFO}]\\[FPS][/] Base FPS={base_fps}, frameskip={frameskip} → adjusted to [{STYLE_INFO}]{adjusted_fps}[/] FPS for human playability"
        )
        return adjusted_fps

    return base_fps


def _get_atari_game_name(env_id: str) -> str | None:
    try:
        spec = gym.spec(env_id)
    except Exception:
        return None
    kwargs = getattr(spec, "kwargs", {}) or {}
    game = kwargs.get("game")
    return str(game) if game else None


def _get_atari_rom_status(env_id: str) -> tuple[bool, str]:
    game = _get_atari_game_name(env_id)
    if not game:
        return False, "missing ROM metadata"

    try:
        import contextlib
        import io

        from ale_py import roms

        with contextlib.redirect_stdout(io.StringIO()):
            rom_path = roms.get_rom_path(game)
        if rom_path and os.path.exists(rom_path):
            return True, "installed"
        return False, "missing ROM"
    except FileNotFoundError:
        return False, "missing ROM"
    except OSError:
        return False, "invalid ROM"
    except Exception:
        return False, "ROM unavailable"


def _get_atari_envs(imported_only: bool = False) -> list[str]:
    try:
        _configure_atari_roms_path()

        import ale_py

        gym.register_envs(ale_py)
        atari_ids = sorted(
            env_id
            for env_id in gym.envs.registry.keys()
            if str(gym.spec(env_id).entry_point) == "ale_py.env:AtariEnv"
            and env_id.startswith("ALE/")
        )
        if imported_only:
            return [env_id for env_id in atari_ids if _get_atari_rom_status(env_id)[0]]
        return atari_ids
    except Exception:
        return []


def _get_stableretro_envs(imported_only: bool = False) -> list[str]:
    try:
        _ensure_stableretro_roms_path_imported()

        import stable_retro as retro

        all_games = sorted(retro.data.list_games(retro.data.Integrations.ALL))
        if imported_only:
            result = []
            for game in all_games:
                try:
                    retro.data.get_romfile_path(game, retro.data.Integrations.ALL)
                    result.append(game)
                except FileNotFoundError:
                    pass
            return result
        return all_games
    except Exception:
        return []


def _get_vizdoom_envs() -> list[str]:
    try:
        import vizdoom.gymnasium_wrapper  # noqa: F401

        return sorted(env_id for env_id in gym.envs.registry.keys() if env_id.startswith("Vizdoom"))
    except Exception:
        return []


def _get_env_platform(env_id: str) -> str:
    """Determine the platform type for an env_id."""
    return {
        "atari": "Atari",
        "vizdoom": "VizDoom",
        "stable-retro": "Stable-Retro",
    }[classify_env_id(env_id)]


def _build_environment_menu_entries(grouped_envs):
    """Build menu entries and env lookup, preserving backend display order."""
    entries = []
    env_id_map = []
    groups = [(label, envs) for label, envs in grouped_envs if envs]

    for group_index, (label, envs) in enumerate(groups):
        if group_index > 0:
            entries.append("")
            env_id_map.append(None)
        for env_id in envs:
            entries.append(f"[{label}]  {env_id}")
            env_id_map.append(env_id)

    return entries, env_id_map


def _terminal_menu_supported() -> bool:
    if os.environ.get("GYMREC_TEXT_MENU", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("TERM") in (None, "", "dumb"):
        return False
    size = shutil.get_terminal_size(fallback=(0, 0))
    return size.columns > 0 and size.lines > 0 and sys.stdin.isatty() and sys.stdout.isatty()


def _select_environment_text_fallback(entries, env_id_map, title: str) -> str:
    selectable = [
        (entry, env_id) for entry, env_id in zip(entries, env_id_map, strict=False) if env_id
    ]
    if not selectable:
        console.print(f"[{STYLE_FAIL}]No environments found.[/]")
        raise SystemExit(1)

    console.print(
        f"[bold]{title.strip()}[/]: [{STYLE_INFO}]{len(selectable)}[/] environments available"
    )
    console.print(
        "[dim]Terminal menu is unavailable here. Type part of a name to search, or paste an exact env id.[/]"
    )

    while True:
        query = Prompt.ask("Search").strip()
        if not query:
            matches = selectable[:25]
        else:
            query_lower = query.lower()
            exact_matches = [
                (entry, env_id) for entry, env_id in selectable if env_id.lower() == query_lower
            ]
            if exact_matches:
                return exact_matches[0][1]
            matches = [
                (entry, env_id)
                for entry, env_id in selectable
                if query_lower in env_id.lower() or query_lower in entry.lower()
            ][:25]

        if not matches:
            console.print(f"[{STYLE_FAIL}]No matches. Try a shorter search.[/]")
            continue

        for index, (entry, _env_id) in enumerate(matches, start=1):
            console.print(f"{index:2d}. {entry}")

        choice = Prompt.ask("Select number, or search again").strip()
        if not choice:
            continue
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(matches):
                return matches[selected_index - 1][1]
        query_lower = choice.lower()
        exact_matches = [
            (entry, env_id) for entry, env_id in selectable if env_id.lower() == query_lower
        ]
        if exact_matches:
            return exact_matches[0][1]

        matches = [
            (entry, env_id)
            for entry, env_id in selectable
            if query_lower in env_id.lower() or query_lower in entry.lower()
        ][:25]
        if len(matches) == 1:
            return matches[0][1]
        if matches:
            for index, (entry, _env_id) in enumerate(matches, start=1):
                console.print(f"{index:2d}. {entry}")
        else:
            console.print(f"[{STYLE_FAIL}]No matches. Try a shorter search.[/]")


def select_environment_interactive(available_recordings_only: bool = False) -> str:
    from simple_term_menu import TerminalMenu

    status_bar = "  ↑↓ navigate · / search · Enter select · Esc cancel"
    if available_recordings_only:
        local_refs = _get_available_recording_refs_from_local()
        hf_refs = _get_available_recording_refs_from_hf()
        all_recording_refs = sorted(set(local_refs + hf_refs))

        if not all_recording_refs:
            console.print(
                f"[{STYLE_FAIL}]No recordings found.[/]\n"
                f"  Local path: [{STYLE_PATH}]{CONFIG['storage']['local_dir']}[/]\n"
                f"  Record first: [{STYLE_CMD}]{_gymrec_cmd('record', '<env_id>')}[/]"
            )
            raise SystemExit(1)

        repo_refs = [ref for ref in all_recording_refs if ref.startswith(HUGGINGFACE_MODEL_SCHEME)]
        env_refs = [
            ref for ref in all_recording_refs if not ref.startswith(HUGGINGFACE_MODEL_SCHEME)
        ]
        atari_envs = [e for e in env_refs if _get_env_platform(e) == "Atari"]
        retro_envs = [e for e in env_refs if _get_env_platform(e) == "Stable-Retro"]
        vizdoom_envs = [e for e in env_refs if _get_env_platform(e) == "VizDoom"]
        entries, env_id_map = _build_environment_menu_entries(
            (
                ("HF Policy Dataset", repo_refs),
                ("Atari", atari_envs),
                ("Stable-Retro", retro_envs),
                ("VizDoom", vizdoom_envs),
            )
        )
        title = "  Select Recording\n"
    else:
        # Original behavior: list all available environments
        atari_envs = _get_atari_envs(imported_only=True)
        retro_envs = _get_stableretro_envs(imported_only=True)
        vizdoom_envs = _get_vizdoom_envs()
        entries, env_id_map = _build_environment_menu_entries(
            (("Atari", atari_envs), ("Stable-Retro", retro_envs), ("VizDoom", vizdoom_envs))
        )

        if not entries:
            console.print(
                "[dim]No environments found. Install ale-py, stable-retro, or vizdoom.[/]"
            )
            raise SystemExit(1)

        title = "  Select Environment\n"

    if not _terminal_menu_supported():
        return _select_environment_text_fallback(entries, env_id_map, title)

    menu = TerminalMenu(
        entries,
        title=title,
        menu_cursor="  > ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("bold", "fg_cyan"),
        search_highlight_style=("fg_black", "bg_cyan", "bold"),
        show_search_hint=True,
        show_search_hint_text="  (type / to search)",
        search_key="/",
        skip_empty_entries=True,
        status_bar=status_bar,
        status_bar_style=("fg_gray",),
    )

    selected_index = menu.show()
    if selected_index is None:
        console.print("[dim]No environment selected.[/]")
        raise SystemExit(0)

    return env_id_map[selected_index]


def _list_environments__alepy():
    atari_ids = _get_atari_envs()
    if atari_ids:
        lines = []
        for env_id in atari_ids:
            has_rom, status = _get_atari_rom_status(env_id)
            style = STYLE_SUCCESS if has_rom else STYLE_FAIL
            lines.append(f"{env_id} [{style}]({status})[/]")
        lines = "\n".join(lines)
    else:
        lines = "[dim]Could not list Atari environments.[/]"
    console.print(
        Panel(
            lines,
            title="[bold]Atari Environments[/]",
            border_style=STYLE_INFO,
            expand=False,
        )
    )


def _list_environments__stableretro():
    all_games = _get_stableretro_envs()
    if all_games:
        lines = []
        try:
            import stable_retro as retro

            for game in all_games:
                try:
                    retro.data.get_romfile_path(game, retro.data.Integrations.ALL)
                    lines.append(f"{game} [{STYLE_SUCCESS}](imported)[/]")
                except FileNotFoundError:
                    lines.append(f"{game} [{STYLE_FAIL}](missing ROM)[/]")
        except Exception as e:
            lines.append(f"[{STYLE_FAIL}]Stable-Retro not installed: {e}[/]")
        console.print(
            Panel(
                "\n".join(lines),
                title="[bold]Stable-Retro Games[/]",
                border_style=STYLE_INFO,
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                "[dim]Stable-Retro not installed or no games found.[/]",
                title="[bold]Stable-Retro Games[/]",
                border_style=STYLE_INFO,
                expand=False,
            )
        )


def _list_environments__vizdoom():
    vizdoom_ids = _get_vizdoom_envs()
    if vizdoom_ids:
        lines = list(vizdoom_ids)
        console.print(
            Panel(
                "\n".join(lines),
                title="[bold]VizDoom Environments[/]",
                border_style=STYLE_INFO,
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                "[dim]VizDoom not installed.[/]",
                title="[bold]VizDoom Environments[/]",
                border_style=STYLE_INFO,
                expand=False,
            )
        )


def list_environments():
    """Print available Atari, stable-retro and VizDoom environments."""
    _list_environments__alepy()
    _list_environments__stableretro()
    _list_environments__vizdoom()


def reindex_games():
    """Force ROMS_PATH re-scan and refresh cached game availability."""
    roms_path = _get_roms_path()
    if not roms_path:
        console.print(f"[{STYLE_FAIL}]ROMS_PATH is not set.[/]")
        console.print(
            f"Set it in .env or pass [{STYLE_CMD}]{_gymrec_cmd('reindex_games', '--roms-path', '/path/to/roms')}[/]"
        )
        raise SystemExit(1)

    if not os.path.exists(roms_path):
        console.print(f"[{STYLE_FAIL}]ROMS_PATH does not exist: {roms_path}[/]")
        raise SystemExit(1)

    console.print(f"[{STYLE_INFO}]Reindexing ROMS_PATH: [{STYLE_PATH}]{roms_path}[/]")
    stable_retro_imported = _ensure_stableretro_roms_path_imported(quiet=True, force=True)

    atari_envs = _get_atari_envs(imported_only=True)
    stable_retro_envs = _get_stableretro_envs(imported_only=True)
    vizdoom_envs = _get_vizdoom_envs()

    console.print(
        f"[{STYLE_SUCCESS}]Reindex complete[/]: "
        f"Atari [{STYLE_INFO}]{len(atari_envs)}[/], "
        f"Stable-Retro [{STYLE_INFO}]{len(stable_retro_envs)}[/], "
        f"VizDoom [{STYLE_INFO}]{len(vizdoom_envs)}[/]"
    )
    if stable_retro_imported:
        console.print(
            f"Stable-Retro ROMs matched during reindex: [{STYLE_INFO}]{stable_retro_imported}[/]"
        )
    console.print(f"Cache: [{STYLE_PATH}]{_roms_path_cache_file()}[/]")


def _import_roms(
    path: str,
    quiet: bool = False,
    source_label: str | None = None,
    return_games: bool = False,
) -> int | tuple[int, list[str]]:
    """Import ROMs into stable-retro from a directory or file."""
    import zipfile

    import stable_retro.data

    if not os.path.exists(path):
        if not quiet:
            console.print(f"[{STYLE_FAIL}]Error: Path not found: {path}[/]")
        return (0, []) if return_games else 0

    known_hashes = stable_retro.data.get_known_hashes()
    imported_games = 0
    matched_games = set()

    def save_if_matches(filename, f):
        nonlocal imported_games
        try:
            data, hash = stable_retro.data.groom_rom(filename, f)
        except (OSError, ValueError):
            return
        if hash in known_hashes:
            game, ext, curpath = known_hashes[hash]
            matched_games.add(game)
            game_path = os.path.join(curpath, game)
            rom_path = os.path.join(game_path, "rom%s" % ext)
            if os.path.exists(rom_path):
                try:
                    with open(rom_path, "rb") as existing:
                        if existing.read() == data:
                            return
                except OSError:
                    pass
            with open(rom_path, "wb") as f:
                f.write(data)

            metadata_path = os.path.join(game_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path) as mf:
                        metadata = json.load(mf)
                    original_name = metadata.get("original_rom_name")
                    if original_name:
                        with open(os.path.join(game_path, original_name), "wb") as of:
                            of.write(data)
                except (json.JSONDecodeError, OSError):
                    pass
            imported_games += 1

    def check_zipfile(f, process_f):
        with zipfile.ZipFile(f) as zf:
            for entry in zf.infolist():
                _root, ext = os.path.splitext(entry.filename)
                with zf.open(entry) as innerf:
                    if ext == ".zip":
                        check_zipfile(innerf, process_f)
                    else:
                        process_f(entry.filename, innerf)

    if os.path.isfile(path):
        # Single file
        with open(path, "rb") as f:
            _root, ext = os.path.splitext(path)
            if ext == ".zip":
                save_if_matches(os.path.basename(path), f)
                f.seek(0)
                try:
                    check_zipfile(f, save_if_matches)
                except zipfile.BadZipFile:
                    pass
            else:
                save_if_matches(os.path.basename(path), f)
    else:
        # Directory - walk recursively
        for root, dirs, files in os.walk(path):
            for filename in files:
                filepath = os.path.join(root, filename)
                with open(filepath, "rb") as f:
                    _root, ext = os.path.splitext(filename)
                    if ext == ".zip":
                        save_if_matches(filename, f)
                        f.seek(0)
                        try:
                            check_zipfile(f, save_if_matches)
                        except zipfile.BadZipFile:
                            pass
                    else:
                        save_if_matches(filename, f)

    if not quiet:
        label = f" from {source_label}" if source_label else ""
        console.print(f"[{STYLE_SUCCESS}]Imported {imported_games} ROM(s){label}[/]")
    if return_games:
        return imported_games, sorted(matched_games)
    return imported_games


def _add_env_id_arg(parser):
    parser.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Gymnasium environment id, or an hf://owner/repo model ref for record and "
            "dataset ref for upload/playback/video/export"
        ),
    )


def _add_fps_arg(parser, help_text):
    parser.add_argument("--fps", type=int, default=None, help=help_text)


def _add_scale_arg(parser):
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help=f"Display scale factor (default: {DEFAULT_CONFIG['display']['scale_factor']})",
    )


def _add_roms_path_arg(parser, default=None):
    parser.add_argument(
        "--roms-path",
        type=str,
        default=default,
        help="Path to ROM files. Also configurable with ROMS_PATH in .env or the shell.",
    )


def _add_backend_arg(parser):
    parser.add_argument(
        "--backend",
        choices=SELECTABLE_RUNTIME_BACKENDS,
        default=None,
        help=(
            "Runtime backend for Stable-Retro-compatible environments. Playback overrides "
            "the recorded capture backend when supplied."
        ),
    )


@dataclass(frozen=True)
class RecordPlan:
    agent: str
    headless: bool
    max_episodes: int | None
    max_steps: int | None
    upload_live: bool

    @property
    def human(self) -> bool:
        return self.agent == "human"


def _make_record_plan(args):
    """Normalize record-mode flags into one validated execution plan."""
    agent = getattr(args, "agent", "human")
    headless = bool(getattr(args, "headless", False))
    episodes = getattr(args, "episodes", None)
    max_steps = getattr(args, "max_steps", None)
    upload_live = bool(getattr(args, "upload_live", False))
    dry_run = bool(getattr(args, "dry_run", False))

    if episodes is not None and episodes < 1:
        return None, "--episodes must be >= 1"
    if upload_live and dry_run:
        return None, "--upload-live cannot be combined with --dry-run"

    if agent == "human":
        if headless:
            return None, "--headless can only be used with --agent (not human mode)"
        return (
            RecordPlan(
                agent=agent,
                headless=False,
                max_episodes=episodes,
                max_steps=max_steps,
                upload_live=upload_live,
            ),
            None,
        )

    if headless and episodes is None:
        return None, "--headless requires --episodes to be specified"

    return (
        RecordPlan(
            agent=agent,
            headless=headless,
            max_episodes=episodes if episodes is not None else 1,
            max_steps=max_steps,
            upload_live=upload_live,
        ),
        None,
    )


async def main():
    parser = argparse.ArgumentParser(description="Gymnasium Recorder/Playback")
    _add_roms_path_arg(parser)
    subparsers = parser.add_subparsers(dest="command")

    parser_record = subparsers.add_parser("record", help="Record gameplay")
    _add_roms_path_arg(parser_record, default=argparse.SUPPRESS)
    _add_env_id_arg(parser_record)
    _add_backend_arg(parser_record)
    _add_fps_arg(parser_record, "Frames per second for playback/recording")
    _add_scale_arg(parser_record)
    parser_record.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Record without uploading to Hugging Face (no HF account required)",
    )
    parser_record.add_argument(
        "--upload-live",
        action="store_true",
        default=False,
        help="Upload each completed episode during recording; incompatible with --dry-run",
    )
    parser_record.add_argument(
        "--agent",
        type=str,
        default="human",
        choices=["human", *AGENT_POLICY_FACTORIES.keys()],
        help=(
            "Input source: human, random, mario, or deterministic breakout policy (default: human)"
        ),
    )
    parser_record.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without display (agent mode only, runs at max speed)",
    )
    parser_record.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to record (default: unlimited for human, 1 for agent)",
    )
    parser_record.add_argument(
        "--max-steps",
        type=int,
        default=None,
        dest="max_steps",
        help="Maximum steps per episode (truncates episode after N steps)",
    )
    parser_record.add_argument(
        "--storage",
        type=str,
        default=None,
        choices=STORAGE_FORMATS,
        help="Observation storage backend: images or lossless-video (default: config.toml)",
    )
    parser_record.add_argument(
        "--dataset-repo",
        type=str,
        default=None,
        help=(
            "Hugging Face dataset target for policy recordings (owner/repo or "
            "hf://owner/repo; default: the source policy repo name)"
        ),
    )
    parser_record.add_argument(
        "--hf-file",
        type=str,
        default=None,
        help="Assert the checkpoint filename selected by the model repo's model.json",
    )
    parser_record.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Hugging Face model revision for policy refs (default: main)",
    )
    parser_record.add_argument(
        "--device",
        type=str,
        default="auto",
        help="SB3 policy device for Hugging Face model refs: auto, cpu, cuda, or mps",
    )
    hf_policy_mode = parser_record.add_mutually_exclusive_group()
    hf_policy_mode.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help="Override recipe.json and use deterministic argmax SB3 actions",
    )
    hf_policy_mode.add_argument(
        "--stochastic",
        action="store_false",
        dest="deterministic",
        help="Override recipe.json and sample stochastic SB3 actions",
    )
    parser_record.set_defaults(deterministic=None)

    parser_playback = subparsers.add_parser("playback", help="Replay a dataset")
    _add_roms_path_arg(parser_playback, default=argparse.SUPPRESS)
    _add_env_id_arg(parser_playback)
    _add_backend_arg(parser_playback)
    _add_fps_arg(parser_playback, "Frames per second for playback/recording")
    _add_scale_arg(parser_playback)
    parser_playback.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify determinism by comparing replayed frames against recorded frames (pixel MSE)",
    )

    parser_video = subparsers.add_parser(
        "video", help="Render recorded dataset episodes to a video file"
    )
    _add_roms_path_arg(parser_video, default=argparse.SUPPRESS)
    _add_env_id_arg(parser_video)
    _add_fps_arg(parser_video, "Frames per second for exported video")
    parser_video.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: ./<env_id>_episodes_...mp4)",
    )
    parser_video.add_argument(
        "--range",
        dest="episode_range",
        type=str,
        default=None,
        help="1-based inclusive episode range, e.g. 3-7 or 3:7",
    )
    parser_video.add_argument(
        "--first",
        type=int,
        default=None,
        help="Export the first N episodes",
    )
    parser_video.add_argument(
        "--last",
        type=int,
        default=None,
        help="Export the last N episodes",
    )

    parser_upload = subparsers.add_parser("upload", help="Upload local dataset to Hugging Face Hub")
    _add_roms_path_arg(parser_upload, default=argparse.SUPPRESS)
    _add_env_id_arg(parser_upload)
    parser_upload.add_argument(
        "--replace",
        action="store_true",
        help="Replace all remote dataset files with the current local dataset",
    )

    subparsers.add_parser("login", help="Log in to Hugging Face Hub")
    parser_list = subparsers.add_parser("list_environments", help="List available environments")
    _add_roms_path_arg(parser_list, default=argparse.SUPPRESS)
    parser_reindex = subparsers.add_parser(
        "reindex_games", help="Re-scan ROMS_PATH and refresh game availability cache"
    )
    _add_roms_path_arg(parser_reindex, default=argparse.SUPPRESS)

    parser_import = subparsers.add_parser(
        "import_roms", help="Import ROMs into stable-retro from a directory or file"
    )
    _add_roms_path_arg(parser_import, default=argparse.SUPPRESS)
    parser_import.add_argument(
        "path",
        type=str,
        help="Path to directory or file containing ROMs",
    )

    parser_minari = subparsers.add_parser(
        "minari-export", help="Export local dataset to Minari format"
    )
    _add_roms_path_arg(parser_minari, default=argparse.SUPPRESS)
    _add_env_id_arg(parser_minari)
    parser_minari.add_argument(
        "--name",
        type=str,
        default=None,
        help="Minari dataset name (default: gymrec/<env_id>/human-v0)",
    )
    parser_minari.add_argument(
        "--author", type=str, default=None, help="Author name for dataset metadata"
    )

    args = parser.parse_args()
    roms_path = getattr(args, "roms_path", None)
    if roms_path:
        os.environ["ROMS_PATH"] = os.path.abspath(os.path.expanduser(roms_path))

    if args.command is None:
        args.command = "record"
        for attr, default in [
            ("env_id", None),
            ("fps", None),
            ("scale", None),
            ("dry_run", False),
            ("upload_live", False),
            ("agent", "human"),
            ("headless", False),
            ("episodes", None),
            ("max_steps", None),
            ("storage", None),
            ("dataset_repo", None),
            ("hf_file", None),
            ("hf_revision", None),
            ("device", "auto"),
            ("deterministic", False),
            ("backend", None),
        ]:
            if not hasattr(args, attr):
                setattr(args, attr, default)

    if args.command == "login":
        _lazy_init()
        ensure_hf_login(force=True)
        return

    if args.command == "list_environments":
        list_environments()
        return

    if args.command == "reindex_games":
        reindex_games()
        return

    if args.command == "import_roms":
        _import_roms(args.path)
        return

    env_id = args.env_id
    hf_policy_source = None
    recording_identity = None

    if env_id is None:
        if args.command in ("playback", "video"):
            _lazy_init()
        # For playback, only show environments with available recordings
        is_recording_command = args.command in ("playback", "video")
        env_id = select_environment_interactive(available_recordings_only=is_recording_command)

    _lazy_init()

    if args.command == "record" and is_huggingface_model_ref(env_id):
        if getattr(args, "agent", "human") != "human":
            console.print(
                f"[{STYLE_FAIL}]Error: Hugging Face model refs already select the policy; "
                "do not combine them with --agent.[/]"
            )
            return
        hf_policy_source = resolve_huggingface_policy_source(
            env_id,
            filename=getattr(args, "hf_file", None),
            revision=getattr(args, "hf_revision", None),
            device=getattr(args, "device", "auto"),
            deterministic=getattr(args, "deterministic", None),
        )
        env_id = hf_policy_source.env_id
        try:
            dataset_repo_id = normalize_dataset_repo_id(
                getattr(args, "dataset_repo", None) or hf_policy_source.repo_id
            )
        except ValueError as exc:
            console.print(f"[{STYLE_FAIL}]Error: {exc}[/]")
            return
        recording_identity = _policy_recording_identity(
            hf_policy_source,
            dataset_repo=dataset_repo_id,
        )
        args.env_id = env_id
        args.agent = "hf"
        console.print(
            f"[{STYLE_INFO}]Resolved policy {hf_policy_source.collector} -> "
            f"{hf_policy_source.provider}:{env_id}"
            f"{' state=' + hf_policy_source.state if hf_policy_source.state else ''}, "
            f"action_set={hf_policy_source.action_set}, "
            f"frame_skip={hf_policy_source.frame_skip}, "
            f"mode={'deterministic' if hf_policy_source.deterministic else 'stochastic'}, "
            f"dataset={recording_identity.display_ref}[/]"
        )
    else:
        if args.command == "record" and getattr(args, "dataset_repo", None):
            console.print(
                f"[{STYLE_FAIL}]Error: --dataset-repo is only valid when recording "
                "a Hugging Face policy ref.[/]"
            )
            return
        try:
            recording_identity = _coerce_recording_identity(env_id)
        except ValueError as exc:
            console.print(f"[{STYLE_FAIL}]Error: {exc}[/]")
            return

    if args.command == "upload":
        upload_local_dataset(recording_identity, replace=args.replace)
        return

    if args.command == "minari-export":
        minari_export(recording_identity, dataset_name=args.name, author=args.author)
        return

    if hasattr(args, "scale") and args.scale is not None:
        CONFIG["display"]["scale_factor"] = args.scale
    storage_format = _normalize_storage_format(
        getattr(args, "storage", None) or CONFIG["storage"].get("format", STORAGE_FORMAT_IMAGES)
    )

    record_plan = None
    if args.command == "record":
        record_plan, plan_error = _make_record_plan(args)
        if plan_error:
            console.print(f"[{STYLE_FAIL}]Error: {plan_error}[/]")
            return
        if record_plan.upload_live and not preflight_live_upload(
            recording_identity, storage_format
        ):
            return

    playback_metadata = None
    loaded_dataset = None
    playback_source = None
    recorded_backend = None
    if args.command in ("playback", "video"):
        loaded_dataset, playback_source = load_recorded_dataset(recording_identity)
        if loaded_dataset is None:
            _print_missing_dataset(recording_identity)
            return
        dataset_metadata = _environment_metadata_from_dataset(loaded_dataset)
        local_metadata = load_local_metadata(recording_identity) or {}
        playback_metadata = {**dataset_metadata, **local_metadata}
        env_id = _recording_env_id(
            recording_identity,
            dataset=loaded_dataset,
            metadata=playback_metadata,
        )
        if not env_id:
            console.print(
                f"[{STYLE_FAIL}]Error: Could not determine the environment for "
                f"{recording_identity.display_ref}.[/]"
            )
            return
        recording_identity = recording_identity.with_env_id(env_id)
        recorded_backend = _capture_backend_from_dataset(loaded_dataset)

    env = None
    fps = args.fps
    if args.command == "playback" or args.command == "record":
        try:
            requested_backend = getattr(args, "backend", None)
            if (
                hf_policy_source is not None
                and requested_backend is not None
                and requested_backend != hf_policy_source.backend
            ):
                raise ValueError(
                    f"The policy recipe requires backend {hf_policy_source.backend!r}; "
                    f"--backend {requested_backend!r} would change its environment contract."
                )
            selected_backend = (
                hf_policy_source.backend
                if hf_policy_source is not None
                else resolve_runtime_backend(
                    env_id,
                    requested=requested_backend,
                    metadata=playback_metadata,
                    recorded_backend=recorded_backend,
                )
            )
            env = create_env(
                env_id,
                stable_retro_state=hf_policy_source.state if hf_policy_source else None,
                backend=selected_backend,
                human_recording=bool(record_plan and record_plan.human),
                metadata=playback_metadata,
                env_make_kwargs=(
                    hf_policy_source.env_make_kwargs if hf_policy_source is not None else None
                ),
            )
            if hf_policy_source is not None:
                provider_env = env
                env = MarioRecipeTaskEnv(provider_env, hf_policy_source.environment["task"])
                env._env_id = env_id
                env._gymrec_backend = hf_policy_source.backend
                env._gymrec_state_name = hf_policy_source.state
                env._gymrec_policy_observations = True
                env._gymrec_make_kwargs = dict(
                    getattr(provider_env, "_gymrec_make_kwargs", {}) or {}
                )
                env._gymrec_rom_sha256 = getattr(provider_env, "_gymrec_rom_sha256", None)
                env._gymrec_nes_controller = bool(
                    getattr(provider_env, "_gymrec_nes_controller", False)
                )
                env._stable_retro = bool(getattr(provider_env, "_stable_retro", False))
                env._gymrec_policy_bundle = {
                    "repo_id": hf_policy_source.repo_id,
                    "revision": hf_policy_source.revision,
                    "model_format_version": RLAB_MODEL_FORMAT_VERSION,
                    "recipe_format_version": RLAB_RECIPE_FORMAT_VERSION,
                    "checkpoint_sha256": hf_policy_source.checkpoint_sha256,
                    "recipe_sha256": hf_policy_source.recipe_sha256,
                }
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[{STYLE_FAIL}]Error: {exc}[/]")
            return
        if fps is None:
            if args.command == "playback" and playback_metadata:
                fps = _get_default_fps_for_env_id(env_id, metadata=playback_metadata)
            else:
                fps = get_default_fps(env)
    elif args.command == "video" and fps is None:
        fps = _get_default_fps_for_env_id(env_id, metadata=playback_metadata)

    if args.command == "record":
        recorder = None
        live_upload_manager = (
            LiveEpisodeUploadManager(recording_identity, storage_format)
            if record_plan.upload_live
            else None
        )

        if record_plan.human:
            input_source = None  # Will default to HumanInputSource in _play
            recorder = DatasetRecorderWrapper(
                env,
                input_source=input_source,
                headless=False,
                collector="human",
                storage_format=storage_format,
                live_upload_manager=live_upload_manager,
            )
            recorded_dataset = await recorder.record(
                fps=fps,
                max_episodes=record_plan.max_episodes,
                max_steps=record_plan.max_steps,
            )
        else:
            mode_str = "headless" if record_plan.headless else "with display"
            console.print(
                f"[{STYLE_INFO}]Recording with {record_plan.agent} agent ({mode_str}), {record_plan.max_episodes} episode(s)[/]"
            )

            if hf_policy_source is not None:
                policy = StableBaselines3Policy(
                    env.action_space,
                    hf_policy_source,
                    observation_space=env.observation_space,
                )
            else:
                policy = create_agent_policy(record_plan.agent, env)
            input_source = AgentInputSource(policy)

            recorder = DatasetRecorderWrapper(
                env,
                input_source=input_source,
                headless=record_plan.headless,
                collector=hf_policy_source.collector if hf_policy_source else record_plan.agent,
                storage_format=storage_format,
                live_upload_manager=live_upload_manager,
                initial_seed=(
                    hf_policy_source.evaluation.get("seed") if hf_policy_source else None
                ),
            )

            total_steps_counter = [0]

            def _make_progress_callback(task_id, progress_bar):
                def callback(episode_number, steps_in_episode):
                    total_steps_counter[0] += steps_in_episode
                    progress_bar.update(
                        task_id,
                        advance=1,
                        description=f"[bold]Episodes[/] [dim]({total_steps_counter[0]} steps total)[/]",
                    )

                return callback

            with _episode_progress(transient=False) as progress:
                task_id = progress.add_task("[bold]Episodes[/]", total=record_plan.max_episodes)
                recorded_dataset = await recorder.record(
                    fps=fps,
                    max_episodes=record_plan.max_episodes,
                    max_steps=record_plan.max_steps,
                    progress_callback=_make_progress_callback(task_id, progress),
                )

        if recorded_dataset is None:
            if recorder is not None:
                recorder.close()
            if record_plan.upload_live:
                console.print(
                    f"[{STYLE_INFO}]Live recording finished. Pending retries, if any: "
                    f"[{STYLE_CMD}]{_gymrec_cmd('upload', recording_identity.display_ref)}[/]"
                )
            return

        if record_plan.upload_live:
            if recorder is not None:
                recorder.close()
            console.print(
                f"[{STYLE_INFO}]Live recording finished. Pending retries, if any: "
                f"[{STYLE_CMD}]{_gymrec_cmd('upload', recording_identity.display_ref)}[/]"
            )
            return

        if recorder is not None:
            save_dataset_locally(
                recorded_dataset,
                recording_identity,
                metadata=recorder._env_metadata,
                video_artifact_dir=(
                    recorder.temp_dir
                    if recorder.storage_format == STORAGE_FORMAT_LOSSLESS_VIDEO
                    else None
                ),
            )
            recorder.close()  # cleanup temp files after dataset is saved
        console.print(
            f"To play back: "
            f"[{STYLE_CMD}]{_gymrec_cmd('playback', recording_identity.display_ref)}[/]"
        )

        if not args.dry_run:
            try:
                do_upload = Confirm.ask(
                    "Upload to Hugging Face Hub?", default=True, console=console
                )
            except EOFError:
                do_upload = False
            if do_upload:
                if not upload_local_dataset(recording_identity):
                    console.print(
                        f"To retry: "
                        f"[{STYLE_CMD}]{_gymrec_cmd('upload', recording_identity.display_ref)}[/]"
                    )
            else:
                console.print(
                    f"To upload later: "
                    f"[{STYLE_CMD}]{_gymrec_cmd('upload', recording_identity.display_ref)}[/]"
                )
    elif args.command == "playback":
        if playback_source == "local":
            console.print(
                f"[{STYLE_INFO}]Playing back from local dataset ({len(loaded_dataset)} frames)[/]"
            )
        else:
            console.print(f"[{STYLE_INFO}]Playing back from Hugging Face Hub[/]")
        recorder = DatasetRecorderWrapper(env, storage_format=STORAGE_FORMAT_IMAGES)

        playback_episodes, playback_total = _dataset_playback_episodes(
            loaded_dataset, verify=args.verify
        )
        await recorder.replay(
            fps=fps,
            total=playback_total,
            verify=args.verify,
            episodes=playback_episodes,
        )
    elif args.command == "video":
        total = len(loaded_dataset)
        if playback_source == "local":
            console.print(
                f"[{STYLE_INFO}]Loading local dataset for video export ({total} frames)[/]"
            )
        else:
            console.print(f"[{STYLE_INFO}]Loaded dataset from Hugging Face Hub ({total} frames)[/]")

        export_dataset_video(
            env_id,
            loaded_dataset,
            output_path=args.output,
            fps=fps,
            episode_range=args.episode_range,
            first=args.first,
            last=args.last,
        )


def cli():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")


if __name__ == "__main__":
    cli()
