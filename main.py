import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import gymnasium as gym
from dotenv import dotenv_values, find_dotenv
from jinja2 import Environment, FileSystemLoader, StrictUndefined
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

from provider_contract import (
    SUPPORTED_PROVIDER_IDS,
    EnvironmentContract,
    build_environment_document,
    create_session,
    discover_providers,
    space_contract,
    validate_environment_document,
)

console = Console()

STYLE_KEY = "bold cyan"
STYLE_ACTION = "bold white"
STYLE_ENV = "bold green"
STYLE_PATH = "dim"
STYLE_CMD = "bold yellow"
STYLE_SUCCESS = "bold green"
STYLE_FAIL = "bold red"
STYLE_INFO = "cyan"


def _load_environment_files(project_path, cwd_path, environ=None):
    """Load project and cwd dotenv files without overriding the process environment."""
    target = os.environ if environ is None else environ
    process_environment = dict(target)
    for path in dict.fromkeys((project_path, cwd_path)):
        if not path or not os.path.isfile(path):
            continue
        target.update(
            {key: value for key, value in dotenv_values(path).items() if value is not None}
        )
    target.update(process_environment)
    return target


_project_dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_load_environment_files(_project_dotenv, find_dotenv(usecwd=True))

_initialized = False

MAX_COMPATIBLE_SEED = 2**32 - 1
DEFAULT_RECORD_SEED = 0


def _get_gymrec_version():
    """Return version string like '0.1.0+abc1234' (or just '0.1.0' if git unavailable)."""
    version = _installed_package_version("gymrec") or "unknown"
    project_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(project_dir, "pyproject.toml")
    if not os.path.isfile(pyproject_path):
        return version

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=project_dir,
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
    """Load provider-neutral keyboard control profiles."""
    key_map = _build_key_name_map(pygame)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keymappings.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing bundled keymappings.toml at {config_path}. "
            "Reinstall gymrec or restore the repository config file."
        )

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    start_key = _resolve_key(config.get("general", {}).get("start_key", "space"), key_map)
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("keymappings.toml is missing required [profiles.*] sections")
    parsed = {
        profile: {
            _resolve_key(key_name, key_map): str(control_label)
            for key_name, control_label in bindings.items()
        }
        for profile, bindings in profiles.items()
    }
    return start_key, parsed


DEFAULT_CONFIG = {
    "display": {"scale_factor": 2},
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

PROVIDER_LABELS = {
    "stable-retro-turbo": "stable-retro-turbo",
    "supermariobrosnes-turbo": "SuperMarioBros-Nes-turbo",
}

HUGGINGFACE_MODEL_SCHEME = "hf://"
HUGGINGFACE_MODEL_URL_HOST = "huggingface.co"
RLAB_RECIPE_FILENAME = "recipe.json"
RLAB_MODEL_FILENAME = "model.json"
RLAB_RELEASE_MANIFEST_FILENAME = "release_manifest.json"
RLAB_RECIPE_DOCUMENT_TYPE = "rlab.recipe"
RLAB_MODEL_DOCUMENT_TYPE = "rlab.model"
RLAB_RELEASE_MANIFEST_DOCUMENT_TYPE = "rlab.release_manifest"
RLAB_RECIPE_FORMAT_VERSION = 1
RLAB_MODEL_FORMAT_VERSION = 1
RLAB_RELEASE_MANIFEST_FORMAT_VERSION = 1
RLAB_RECIPE_SCHEMA_VERSION = 2
OBSERVATION_IMAGE_KEYS = ("obs", "image", "screen")
STORAGE_FORMAT_IMAGES = "images"
STORAGE_FORMAT_LOSSLESS_VIDEO = "lossless-video"
STORAGE_FORMATS = (STORAGE_FORMAT_IMAGES, STORAGE_FORMAT_LOSSLESS_VIDEO)
VIDEO_ARTIFACT_DIR = "videos"
COLLECTOR_ARTIFACT_DIR = "collectors"
ENVIRONMENT_ARTIFACT_DIR = "environments"
ENVIRONMENT_DOCUMENT_FILENAME = "environment.json"
COLLECTION_DOCUMENT_TYPE = "gymrec.collection"
COLLECTION_FORMAT_VERSION = 2
DATASET_FORMAT_VERSION = 3
CANONICAL_VIDEO_SUFFIX = ".rgb.mkv.bin"
PREVIEW_VIDEO_SUFFIX = ".preview.mp4"
DATASET_REPLAY_FILENAME = "replay.mp4"
RUNTIME_VIDEO_BASE_COLUMN = "_gymrec_video_base_path"
RUNTIME_HF_REPO_COLUMN = "_gymrec_hf_repo_id"
ENVIRONMENT_METADATA_FILENAME = "gymrec-metadata.json"
_VIDEO_DECODE_CACHE = {}


def _extract_observation_image(observation):
    """Extract the image array from an observation, including dict observations."""
    if isinstance(observation, dict):
        for key in OBSERVATION_IMAGE_KEYS:
            if key in observation:
                return observation[key]
    return observation


def _normalize_storage_format(value):
    """Normalize and validate a configured storage format."""
    if value is None or not str(value).strip():
        raise ValueError("Storage format is required")
    value = str(value).strip().lower()
    if value not in STORAGE_FORMATS:
        raise ValueError(
            f"Unknown storage format '{value}'. Expected one of: {', '.join(STORAGE_FORMATS)}"
        )
    return value


def _configured_storage_format(value=None):
    """Resolve an explicit storage format or the single configured default."""
    config = CONFIG or DEFAULT_CONFIG
    return _normalize_storage_format(config["storage"]["format"] if value is None else value)


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


def _recording_observation(provider_session, observation):
    """Return the provider-selected canonical recording frame."""
    return provider_session.recording_observation(observation)


def _provider_fps(provider_session):
    fps = float(getattr(provider_session, "fps", 0) or 0)
    if fps <= 0:
        raise ValueError(
            f"Environment provider {provider_session.provider_id!r} did not expose a valid fps"
        )
    return max(int(round(fps)), 1)


def _sha256_rgb(frame_array):
    """Hash canonical RGB frame bytes."""
    return hashlib.sha256(np.ascontiguousarray(frame_array).tobytes()).hexdigest()


def _sha256_file(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require_video_tools(error_suffix):
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    missing = []
    if ffmpeg_path is None:
        missing.append("ffmpeg")
    if ffprobe_path is None:
        missing.append("ffprobe")
    if missing:
        raise RuntimeError(f"{' and '.join(missing)} {error_suffix}")
    return ffmpeg_path, ffprobe_path


def _require_lossless_video_tools():
    """Return ffmpeg/ffprobe paths or raise before starting a video-backed recording."""
    return _require_video_tools(f"required for --storage {STORAGE_FORMAT_LOSSLESS_VIDEO}")


def _require_dataset_replay_tools():
    """Return ffmpeg/ffprobe paths required to publish replay.mp4."""
    return _require_video_tools("required to publish replay.mp4")


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


class _StreamingBrowserPreviewWriter:
    """Write a browser-safe H.264 MP4 without buffering an episode in memory."""

    def __init__(self, output_path, fps, ffmpeg_path=None):
        self.output_path = output_path
        self.fps = fps
        self.ffmpeg_path = ffmpeg_path or _require_dataset_replay_tools()[0]
        self.process = None
        self.shape = None

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
            "libx264",
            "-tag:v",
            "avc1",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.output_path,
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self.process is None:
            self._start(frame)
        elif tuple(frame.shape) != self.shape:
            raise ValueError(f"All preview frames must have shape {self.shape}; got {frame.shape}")
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process is None:
            raise ValueError("Cannot encode an empty preview video")
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
    writer = _StreamingBrowserPreviewWriter(output_path, fps, ffmpeg_path=ffmpeg_path)
    try:
        for frame in frames:
            writer.write(frame)
        writer.close()
    except Exception:
        writer.abort()
        raise


def _transcode_browser_preview_video(input_path, output_path, ffmpeg_path=None):
    """Transcode an existing video to the browser-safe replay contract."""
    ffmpeg_path = ffmpeg_path or _require_dataset_replay_tools()[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            os.path.abspath(input_path),
            "-an",
            "-c:v",
            "libx264",
            "-tag:v",
            "avc1",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffmpeg exited with {completed.returncode}")


def _verify_browser_preview_video(path, ffprobe_path=None):
    """Validate the codec and streaming layout required by Hugging Face card playback."""
    ffprobe_path = ffprobe_path or shutil.which("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError("ffprobe is required to validate replay.mp4")
    completed = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,pix_fmt,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(completed.stdout)
    streams = probe.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("replay video does not contain a video stream")
    stream = streams[0]
    expected = {"codec_name": "h264", "codec_tag_string": "avc1", "pix_fmt": "yuv420p"}
    for key, value in expected.items():
        if stream.get(key) != value:
            raise ValueError(f"replay video {key} must be {value!r}, got {stream.get(key)!r}")
    duration = float(probe.get("format", {}).get("duration") or 0.0)
    frame_count = int(stream.get("nb_read_frames") or 0)
    if duration <= 0 or frame_count <= 0:
        raise ValueError("replay video must have a positive duration and frame count")
    with open(path, "rb") as file_obj:
        data = file_obj.read()
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    if moov < 0 or mdat < 0 or moov > mdat:
        raise ValueError("replay video must use faststart with moov before mdat")
    return {"duration_seconds": duration, "frames": frame_count, **expected}


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


def _first_dataset_value(dataset, column):
    if dataset is None or column not in (getattr(dataset, "column_names", None) or []):
        return None
    for value in reversed(dataset[column]):
        if value is not None and value != "":
            return value
    return None


def _environment_metadata_from_dataset(dataset):
    return {
        column: value
        for column in (
            "provider_id",
            "env_id",
            "environment_contract_id",
            "storage_format",
        )
        if (value := _first_dataset_value(dataset, column)) is not None
    }


def _playback_metadata(dataset, local_metadata=None):
    return {
        **dict(local_metadata or {}),
        **_environment_metadata_from_dataset(dataset),
    }


def _merge_config(defaults, overrides, path=""):
    """Merge a config mapping while rejecting unknown keys and incompatible types."""
    if not isinstance(overrides, dict):
        location = path or "configuration"
        raise ValueError(f"{location} must be a TOML table")

    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        location = f" in [{path}]" if path else ""
        raise ValueError(f"Unknown configuration key{location}: {', '.join(unknown)}")

    merged = copy.deepcopy(defaults)
    for key, value in overrides.items():
        key_path = f"{path}.{key}" if path else key
        default = defaults[key]
        if isinstance(default, dict):
            merged[key] = _merge_config(default, value, key_path)
            continue

        expected_type = type(default)
        valid_type = type(value) is expected_type
        if isinstance(default, list):
            valid_type = isinstance(value, list)
            if valid_type and default:
                item_type = type(default[0])
                valid_type = all(type(item) is item_type for item in value)
        if not valid_type:
            raise ValueError(
                f"Configuration key {key_path} must be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        merged[key] = value
    return merged


def _load_config():
    """Load and validate configuration from config.toml."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            user_config = tomllib.load(f)
        config = _merge_config(DEFAULT_CONFIG, user_config)
    config["storage"]["local_dir"] = os.path.abspath(
        os.path.expanduser(config["storage"]["local_dir"])
    )
    return config


def _gymrec_cmd(*parts):
    """Format a user-facing command with the installed CLI entrypoint."""
    return " ".join(("gymrec", *(str(part) for part in parts)))


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
    global START_KEY, CONTROL_PROFILES

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

    START_KEY, CONTROL_PROFILES = _load_keymappings(pygame)
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


class HumanInputSource:
    """Translate pressed keys through the provider's advertised control profile."""

    def __init__(self, provider_session, key_lock, current_keys):
        self.provider_session = provider_session
        self.key_lock = key_lock
        self.current_keys = current_keys
        profile = getattr(provider_session, "control_profile", None)
        if not profile:
            raise ValueError(
                f"Environment provider {provider_session.provider_id!r} does not advertise "
                "a human control profile"
            )
        try:
            self.key_to_label = CONTROL_PROFILES[profile]
        except KeyError as exc:
            raise ValueError(
                f"No keyboard mapping is installed for provider control profile {profile!r}"
            ) from exc

    def get_action(self, observation):
        del observation
        with self.key_lock:
            labels = {
                label for key, label in self.key_to_label.items() if key in self.current_keys
            }
        return self.provider_session.action_from_labels(labels)


class AgentInputSource:
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

    @property
    def policy_action(self):
        """Return the policy-space action selected before any environment adapter."""
        return getattr(self.policy, "last_policy_action", None)

    def observe_step(self, reward, terminated, truncated, info):
        """Forward step results to the policy when it needs feedback."""
        if hasattr(self.policy, "observe_step"):
            self.policy.observe_step(reward, terminated, truncated, info)


# =============================================================================
# Policies
# =============================================================================


class RandomPolicy:
    """Random policy that samples from the action space."""

    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.action_space.seed(int(seed))

    def __call__(self, observation):
        """Sample a random action from the action space."""
        return self.action_space.sample()


@dataclass(frozen=True)
class HFPolicySource:
    repo_id: str
    revision: str
    checkpoint_filename: str
    model_path: str
    model_json_path: str
    recipe_json_path: str
    release_manifest_path: str | None
    model_document: dict
    environment: dict
    deterministic: bool = False
    device: str = "auto"

    @property
    def environment_contract(self) -> EnvironmentContract:
        return EnvironmentContract.parse(self.environment, label="recipe.eval.environment")

    @property
    def provider(self) -> str:
        return self.environment_contract.provider_id

    @property
    def env_id(self) -> str:
        return self.environment_contract.environment_id

    @property
    def collector(self) -> str:
        return f"hf://{self.repo_id}@{self.revision}"

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.model_document["checkpoint"]["sha256"])


@dataclass(frozen=True)
class CollectorContract:
    contract_id: str
    policy_mode: str
    collection_document: dict
    model_json_path: str
    recipe_json_path: str
    release_manifest_path: str | None = None

    @property
    def relative_dir(self) -> str:
        return f"{COLLECTOR_ARTIFACT_DIR}/{self.contract_id}"


@dataclass(frozen=True)
class EnvironmentArtifact:
    contract_id: str
    document: dict

    @property
    def relative_dir(self) -> str:
        return f"{ENVIRONMENT_ARTIFACT_DIR}/{self.contract_id}"


def _environment_artifact(contract, provider_session):
    contract_id, document = build_environment_document(contract, provider_session)
    return EnvironmentArtifact(contract_id=contract_id, document=document)


class StableBaselines3Policy:
    """Run an SB3 policy against recipe-native provider observations."""

    def __init__(self, source: HFPolicySource, provider_session):
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
        self.provider_session = provider_session
        self.last_policy_action = None
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
        try:
            provider_session.validate_policy(self.model)
        except ValueError as exc:
            raise SystemExit(f"Policy is incompatible with its environment provider: {exc}") from exc

    def reset(self, **kwargs):
        self.last_policy_action = None
        seed = kwargs.get("seed")
        if seed is not None:
            seed = int(seed)
            if seed < 0 or seed > MAX_COMPATIBLE_SEED:
                raise ValueError(
                    f"Policy seed must be between 0 and {MAX_COMPATIBLE_SEED}; got {seed}."
                )
            self.model.set_random_seed(seed)

    def observe_step(self, reward, terminated, truncated, info):
        del reward, terminated, truncated, info

    def __call__(self, observation):
        policy_observation = self.provider_session.policy_observation(observation)
        action, _ = self.model.predict(
            policy_observation, deterministic=self.source.deterministic
        )
        raw_action = np.asarray(action)
        self.last_policy_action = (
            int(raw_action.reshape(-1)[0]) if raw_action.size == 1 else raw_action.tolist()
        )
        return self._copy_action(self.provider_session.adapt_policy_action(action))

    @staticmethod
    def _copy_action(action):
        if isinstance(action, np.ndarray):
            return action.copy()
        return action


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _installed_package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_collector_contract(
    source,
    provider_session,
    *,
    environment_contract_id,
    inference_device,
):
    """Build the immutable, seed-independent policy collector contract."""
    mode = "deterministic" if source.deterministic else "stochastic"
    source_document = {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "model": {
            "filename": RLAB_MODEL_FILENAME,
            "sha256": _sha256_file(source.model_json_path),
        },
        "recipe": {
            "filename": RLAB_RECIPE_FILENAME,
            "sha256": _sha256_file(source.recipe_json_path),
        },
        "checkpoint": {
            "filename": source.checkpoint_filename,
            "sha256": source.checkpoint_sha256,
        },
    }
    if source.release_manifest_path:
        source_document["release_manifest"] = {
            "filename": RLAB_RELEASE_MANIFEST_FILENAME,
            "sha256": _sha256_file(source.release_manifest_path),
        }
    collection_document = {
        "document_type": COLLECTION_DOCUMENT_TYPE,
        "format_version": COLLECTION_FORMAT_VERSION,
        "source": source_document,
        "policy": {
            "mode": mode,
            "seed_derivation": {
                "protocol": "base-plus-episode-index-v1",
                "base_seed_stored": False,
                "episode_seed_column": "policy_seed",
            },
        },
        "execution": {
            "environment_contract_id": environment_contract_id,
            "policy_observations": "provider-selected",
            "recorded_observations": "provider-selected-rgb",
            "recorded_actions": "exact-env-step-input",
            "recorded_policy_actions": "unadapted-policy-output",
            "action_space": space_contract(provider_session.env.action_space),
            "observation_space": space_contract(provider_session.env.observation_space),
        },
        "runtime": {
            "inference_device": str(inference_device),
            "packages": {
                name: _installed_package_version(name)
                for name in (
                    "gymrec",
                    "stable-baselines3",
                    "torch",
                    "numpy",
                    source.provider,
                )
            },
            "gymrec_policy_adapter_version": 3,
        },
    }
    contract_id = hashlib.sha256(_canonical_json_bytes(collection_document)).hexdigest()
    return CollectorContract(
        contract_id=contract_id,
        policy_mode=mode,
        collection_document=collection_document,
        model_json_path=source.model_json_path,
        recipe_json_path=source.recipe_json_path,
        release_manifest_path=source.release_manifest_path,
    )


def _materialize_collector_contract(contract, root):
    if contract is None:
        return None
    destination = os.path.join(root, contract.relative_dir)
    os.makedirs(destination, exist_ok=True)
    sources = {
        RLAB_MODEL_FILENAME: contract.model_json_path,
        RLAB_RECIPE_FILENAME: contract.recipe_json_path,
    }
    if contract.release_manifest_path:
        sources[RLAB_RELEASE_MANIFEST_FILENAME] = contract.release_manifest_path
    for filename, source_path in sources.items():
        destination_path = os.path.join(destination, filename)
        if os.path.exists(destination_path):
            if _sha256_file(destination_path) != _sha256_file(source_path):
                raise ValueError(
                    f"Collector contract conflict for {contract.contract_id}/{filename}"
                )
        else:
            shutil.copy2(source_path, destination_path)
    collection_path = os.path.join(destination, "collection.json")
    collection_bytes = _canonical_json_bytes(contract.collection_document) + b"\n"
    if os.path.exists(collection_path):
        with open(collection_path, "rb") as file_obj:
            if file_obj.read() != collection_bytes:
                raise ValueError(
                    f"Collector contract conflict for {contract.contract_id}/collection.json"
                )
    else:
        with open(collection_path, "wb") as file_obj:
            file_obj.write(collection_bytes)
    return destination


def _materialize_environment_artifact(artifact, root):
    if artifact is None:
        raise ValueError("Recording requires an immutable environment artifact")
    destination = os.path.join(root, artifact.relative_dir)
    os.makedirs(destination, exist_ok=True)
    document_path = os.path.join(destination, ENVIRONMENT_DOCUMENT_FILENAME)
    payload = _canonical_json_bytes(artifact.document) + b"\n"
    if os.path.exists(document_path):
        with open(document_path, "rb") as stream:
            if stream.read() != payload:
                raise ValueError(f"Environment contract conflict for {artifact.contract_id}")
    else:
        with open(document_path, "wb") as stream:
            stream.write(payload)
    return destination


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
        revision = None
        repo_name = parts[1]
        if "@" in repo_name:
            repo_name, revision = repo_name.rsplit("@", 1)
            if not repo_name or not revision:
                raise ValueError(f"invalid Hugging Face model revision in {value!r}")
        repo_id = f"{parts[0]}/{repo_name}"
        filename = "/".join(parts[2:]) or None
        return repo_id, filename, revision

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


def _resolve_huggingface_model_commit(repo_id, revision):
    """Resolve a model ref once and return its immutable Hub commit plus file list."""
    try:
        api = HfApi()
        info = api.model_info(repo_id=repo_id, revision=revision)
        commit = str(info.sha)
        siblings = getattr(info, "siblings", None) or []
        files = {
            str(getattr(item, "rfilename", item))
            for item in siblings
            if getattr(item, "rfilename", item)
        }
        if not files:
            files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=commit))
    except Exception as exc:
        raise SystemExit(
            f"Could not resolve Hugging Face model repo {repo_id}@{revision}: {exc}"
        ) from exc
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise SystemExit(
            f"Hugging Face returned a non-immutable revision for {repo_id}@{revision}: {commit!r}"
        )
    return commit, files


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


def _validate_release_manifest(
    manifest_document,
    *,
    manifest_path,
    repo_id,
    revision,
    downloaded_paths,
    required_artifacts=(),
):
    _validate_rlab_document(
        manifest_document,
        label=RLAB_RELEASE_MANIFEST_FILENAME,
        document_type=RLAB_RELEASE_MANIFEST_DOCUMENT_TYPE,
        format_version=RLAB_RELEASE_MANIFEST_FORMAT_VERSION,
    )
    declared_repo = _metadata_value(manifest_document, ("repository", "repo_id"))
    if declared_repo is not None and declared_repo != repo_id:
        raise SystemExit(
            f"{repo_id}/{RLAB_RELEASE_MANIFEST_FILENAME} declares repository {declared_repo!r}"
        )
    artifacts = manifest_document.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise SystemExit(
            f"{repo_id}/{RLAB_RELEASE_MANIFEST_FILENAME} must declare a non-empty artifacts object"
        )
    for required in (RLAB_MODEL_FILENAME, RLAB_RECIPE_FILENAME, *required_artifacts):
        if required not in artifacts:
            raise SystemExit(f"{repo_id}/{RLAB_RELEASE_MANIFEST_FILENAME} is missing {required!r}")
    for filename, binding in sorted(artifacts.items()):
        if (
            not isinstance(filename, str)
            or not filename
            or os.path.isabs(filename)
            or ".." in filename.split("/")
        ):
            raise SystemExit(
                f"{repo_id}/{RLAB_RELEASE_MANIFEST_FILENAME} has an unsafe artifact path"
            )
        path = downloaded_paths.get(filename)
        if path is None:
            path = _download_huggingface_policy_file(repo_id, revision, filename)
            downloaded_paths[filename] = path
        _validate_bound_file(path, binding, label=f"release artifact {filename}")

    evaluation = manifest_document.get("evaluation")
    if isinstance(evaluation, dict):
        for key, filename in (
            ("recipe_sha256", RLAB_RECIPE_FILENAME),
            ("checkpoint_sha256", None),
        ):
            expected = evaluation.get(key)
            if expected is None:
                continue
            if filename is None:
                matching = [
                    name
                    for name, binding in artifacts.items()
                    if isinstance(binding, dict) and binding.get("sha256") == expected
                ]
                if not matching:
                    raise SystemExit(
                        f"{repo_id}/{RLAB_RELEASE_MANIFEST_FILENAME} has an unbound {key}"
                    )
            elif artifacts[filename].get("sha256") != expected:
                raise SystemExit(
                    f"{repo_id}/{RLAB_RELEASE_MANIFEST_FILENAME} {key} conflicts with artifacts"
                )
    return manifest_path


def _recipe_evaluation_environment(recipe_document, model_document, *, repo_id):
    del model_document
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
        raise SystemExit(
            f"{repo_id}/recipe.json must declare recipe.eval.environment; "
            "training-environment fallback is not supported"
        )
    try:
        EnvironmentContract.parse(environment, label=f"{repo_id}/recipe.json environment")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    revision=None,
    device="auto",
    deterministic=None,
):
    _lazy_init()
    try:
        repo_id, _parsed_filename, parsed_revision = parse_huggingface_model_ref(ref)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    requested_revision = revision or parsed_revision or "main"
    resolved_revision, repo_files = _resolve_huggingface_model_commit(repo_id, requested_revision)
    for required_filename in (RLAB_MODEL_FILENAME, RLAB_RECIPE_FILENAME):
        if required_filename not in repo_files:
            raise SystemExit(
                f"{repo_id}@{resolved_revision} is missing required {required_filename}"
            )
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
    if checkpoint_filename not in repo_files:
        raise SystemExit(
            f"{repo_id}@{resolved_revision} is missing model.json checkpoint "
            f"{checkpoint_filename!r}"
        )
    model_path = _download_huggingface_policy_file(repo_id, resolved_revision, checkpoint_filename)
    _validate_bound_file(model_path, checkpoint, label=checkpoint_filename)

    release_manifest_path = None
    if RLAB_RELEASE_MANIFEST_FILENAME in repo_files:
        release_manifest_path = _download_huggingface_policy_file(
            repo_id, resolved_revision, RLAB_RELEASE_MANIFEST_FILENAME
        )
        release_manifest_document = _load_json_document(
            release_manifest_path,
            label=f"{repo_id}/{RLAB_RELEASE_MANIFEST_FILENAME}",
        )
        _validate_release_manifest(
            release_manifest_document,
            manifest_path=release_manifest_path,
            repo_id=repo_id,
            revision=resolved_revision,
            downloaded_paths={
                RLAB_MODEL_FILENAME: model_json_path,
                RLAB_RECIPE_FILENAME: recipe_json_path,
                checkpoint_filename: model_path,
            },
            required_artifacts=(checkpoint_filename,),
        )

    policy = model_document.get("policy")
    if not isinstance(policy, dict):
        raise SystemExit(f"{repo_id}/model.json is missing policy")
    for key in ("algorithm_id", "model_class"):
        if policy.get(key) != checkpoint.get(key):
            raise SystemExit(f"{repo_id}/model.json policy.{key} does not match checkpoint.{key}")

    evaluation, environment = _recipe_evaluation_environment(
        recipe_document, model_document, repo_id=repo_id
    )
    action_sampling = evaluation["action_sampling"]
    resolved_deterministic = (
        action_sampling == "deterministic" if deterministic is None else bool(deterministic)
    )

    return HFPolicySource(
        repo_id=repo_id,
        revision=resolved_revision,
        checkpoint_filename=checkpoint_filename,
        model_path=model_path,
        model_json_path=model_json_path,
        recipe_json_path=recipe_json_path,
        release_manifest_path=release_manifest_path,
        model_document=model_document,
        environment=environment,
        deterministic=resolved_deterministic,
        device=device,
    )


@dataclass(frozen=True)
class LiveEpisodePackage:
    episode_id: str
    package_dir: str

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
    fps,
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
        "fps": fps,
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
        preview_fps=fps,
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

    def __init__(
        self,
        recording_identity,
        storage_format,
        collector_contract=None,
        environment_artifact=None,
        fps=None,
        max_retries=5,
        base_wait=1.0,
    ):
        self.identity = _coerce_recording_identity(recording_identity)
        self.storage_format = storage_format
        self.collector_contract = collector_contract
        self.environment_artifact = environment_artifact
        self.fps = fps
        self.max_retries = max_retries
        self.base_wait = base_wait
        self.queue_dir = _get_live_upload_queue_dir(self.identity)

    def begin_episode(self, episode_uuid):
        episode_id = str(episode_uuid)
        package_dir = os.path.join(self.queue_dir, episode_id)
        if os.path.exists(package_dir):
            shutil.rmtree(package_dir)
        os.makedirs(package_dir, exist_ok=True)
        _materialize_collector_contract(self.collector_contract, package_dir)
        _materialize_environment_artifact(self.environment_artifact, package_dir)
        package = LiveEpisodePackage(episode_id=episode_id, package_dir=package_dir)
        _set_live_upload_manifest_entry(
            self.identity,
            package.episode_id,
            state="recording",
            package_dir=package.package_dir,
            storage_format=self.storage_format,
            frames=0,
            fps=self.fps,
        )
        return package

    def upload_episode(self, package, dataset):
        return _upload_live_episode_package(
            self.identity,
            package.episode_id,
            package.package_dir,
            dataset,
            storage_format=self.storage_format,
            fps=self.fps,
            max_retries=self.max_retries,
            base_wait=self.base_wait,
            persist_dataset=True,
        )

    def discard_episode(self, package):
        """Remove an unfinished episode without publishing a synthetic boundary."""
        shutil.rmtree(package.package_dir, ignore_errors=True)
        manifest = _load_live_upload_manifest(self.identity)
        manifest.get("episodes", {}).pop(package.episode_id, None)
        _save_live_upload_manifest(self.identity, manifest)


@dataclass(frozen=True)
class DatasetField:
    name: str
    type_label: str
    description: str
    cast: str | None = None

    def card_line(self):
        return f"- **{self.name}** (`{self.type_label}`): {self.description}"


COMMON_DATASET_FIELDS = (
    DatasetField("episode_id", "string", "Canonical UUID for the episode", "string"),
    DatasetField("step_index", "int64", "Zero-based row index within the episode", "int64"),
    DatasetField("seed", "int64", "Environment reset seed for this episode", "int64"),
    DatasetField(
        "actions",
        "environment action or null",
        "Exact action passed to `env.step`; null on the terminal-observation row",
    ),
    DatasetField(
        "policy_actions",
        "policy action or null",
        "Original policy output before provider adaptation; null when unavailable",
    ),
    DatasetField("rewards", "float64 or null", "Step reward; null on terminal rows", "float64"),
    DatasetField(
        "terminations",
        "bool or null",
        "Natural termination flag; null on terminal rows",
        "bool",
    ),
    DatasetField("truncations", "bool or null", "Truncation flag; null on terminal rows", "bool"),
    DatasetField(
        "infos",
        "string or null",
        "Environment info as JSON; null on terminal rows",
        "string",
    ),
    DatasetField("session_id", "string", "Canonical UUID for one `gymrec record` run", "string"),
    DatasetField(
        "dataset_format_version",
        "int64",
        "Version of the canonical gymrec row schema",
        "int64",
    ),
    DatasetField(
        "collector", "string", "Collector name, built-in agent, or immutable policy ref", "string"
    ),
    DatasetField("gymrec_version", "string", "gymrec version used for collection", "string"),
    DatasetField("storage_format", "string", "Observation storage backend", "string"),
    DatasetField(
        "provider_id",
        "string",
        "Environment provider distribution that produced the transition",
        "string",
    ),
    DatasetField(
        "env_id", "string", "Native-runtime logical environment identifier", "string"
    ),
    DatasetField(
        "environment_contract_id",
        "string",
        "SHA-256 key for the immutable environment document",
        "string",
    ),
    DatasetField(
        "collector_contract_id",
        "string or null",
        "SHA-256 key for immutable collector documents",
        "string",
    ),
    DatasetField(
        "policy_mode", "string or null", "Deterministic or stochastic policy mode", "string"
    ),
    DatasetField("policy_seed", "int64 or null", "Episode policy seed", "int64"),
    DatasetField(
        "collector_terminated",
        "bool",
        "True only on the final observation row when collection stopped before a provider boundary",
        "bool",
    ),
)

IMAGE_DATASET_FIELDS = (
    DatasetField("observations", "Image", "Lossless RGB environment observation", "image"),
)

VIDEO_DATASET_FIELDS = (
    DatasetField("video_path", "string", "Relative canonical lossless RGB stream path", "string"),
    DatasetField("frame_sha256", "string", "SHA-256 of decoded raw RGB frame bytes", "string"),
    DatasetField("frame_width", "int64", "Decoded RGB frame width", "int64"),
    DatasetField("frame_height", "int64", "Decoded RGB frame height", "int64"),
)

TRANSITION_DATASET_FIELD_NAMES = frozenset(
    {"actions", "policy_actions", "rewards", "terminations", "truncations", "infos"}
)
COMMON_DATASET_FIELD_NAMES = tuple(field.name for field in COMMON_DATASET_FIELDS)
ROW_CONTEXT_FIELD_NAMES = tuple(
    name for name in COMMON_DATASET_FIELD_NAMES if name not in TRANSITION_DATASET_FIELD_NAMES
)


def _canonical_dataset_row(**values):
    """Build a complete canonical common row and reject unknown fields."""
    unknown = sorted(set(values) - set(COMMON_DATASET_FIELD_NAMES))
    if unknown:
        raise ValueError(f"Unknown canonical dataset row field(s): {', '.join(unknown)}")
    row = {name: None for name in COMMON_DATASET_FIELD_NAMES}
    row.update(values)
    return row


def _dataset_fields(storage_format):
    storage_format = _normalize_storage_format(storage_format)
    storage_fields = (
        IMAGE_DATASET_FIELDS if storage_format == STORAGE_FORMAT_IMAGES else VIDEO_DATASET_FIELDS
    )
    return (*COMMON_DATASET_FIELDS, *storage_fields)


def _recording_dataset_from_dict(data, storage_format):
    """Build a Dataset using the canonical ordered field and cast specification."""
    dataset = Dataset.from_dict(data)
    for field in _dataset_fields(storage_format):
        if field.name not in dataset.column_names or field.cast is None:
            continue
        feature = HFImage() if field.cast == "image" else Value(field.cast)
        dataset = dataset.cast_column(field.name, feature)
    return dataset


def _canonical_column_order(storage_format):
    return [field.name for field in _dataset_fields(storage_format)]


def _canonical_columns(storage_format):
    return set(_canonical_column_order(storage_format))


def _validate_canonical_dataset_schema(dataset, *, label="dataset"):
    try:
        storage_format = _dataset_storage_format(dataset)
    except ValueError as exc:
        raise ValueError(
            f"{label} does not use the current canonical gymrec schema ({exc}). "
            "Legacy datasets are not migrated or aligned."
        ) from exc
    expected = _canonical_columns(storage_format)
    actual_columns = _strip_runtime_columns(dataset).column_names
    actual = set(actual_columns)
    if actual != expected or actual_columns != _canonical_column_order(storage_format):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported " + ", ".join(extra))
        if not missing and not extra:
            details.append("columns are out of canonical order")
        raise ValueError(
            f"{label} does not use the current canonical gymrec schema ({'; '.join(details)}). "
            "Legacy datasets are not migrated or aligned."
        )
    versions = set(dataset["dataset_format_version"])
    if versions != {DATASET_FORMAT_VERSION}:
        raise ValueError(
            f"{label} uses dataset_format_version {sorted(versions, key=str)!r}; "
            f"expected {DATASET_FORMAT_VERSION}. Legacy datasets are not migrated or aligned."
        )
    for column in ("episode_id", "session_id"):
        for value in set(dataset[column]):
            try:
                canonical = str(uuid.UUID(str(value)))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"{label} has invalid {column} {value!r}") from exc
            if value != canonical:
                raise ValueError(
                    f"{label} has non-canonical {column} {value!r}; expected {canonical!r}"
                )
    episode_keys, row_indices_by_episode = _ordered_episode_rows(dataset)
    for episode_key in episode_keys:
        row_indices = row_indices_by_episode[episode_key]
        step_indices = [int(dataset[index]["step_index"]) for index in row_indices]
        expected_steps = list(range(len(row_indices)))
        if step_indices != expected_steps:
            raise ValueError(
                f"{label} episode {episode_key} has step_index values {step_indices}; "
                f"expected {expected_steps}"
            )
        terminal_steps = [
            step_index
            for step_index, row_index in enumerate(row_indices)
            if _is_terminal_action(dataset[row_index].get("actions"))
        ]
        if terminal_steps != [len(row_indices) - 1]:
            raise ValueError(
                f"{label} episode {episode_key} must have exactly one terminal-observation "
                "row at the final step"
            )
        terminal_row = dataset[row_indices[-1]]
        non_null_terminal_fields = sorted(
            field for field in TRANSITION_DATASET_FIELD_NAMES if terminal_row.get(field) is not None
        )
        if non_null_terminal_fields:
            raise ValueError(
                f"{label} episode {episode_key} terminal-observation row has non-null "
                f"transition fields: {', '.join(non_null_terminal_fields)}"
            )
        collector_terminated = terminal_row.get("collector_terminated")
        if not isinstance(collector_terminated, bool):
            raise ValueError(
                f"{label} episode {episode_key} terminal-observation row must declare "
                "collector_terminated as a boolean"
            )
        for row_index in row_indices[:-1]:
            row = dataset[row_index]
            if row.get("collector_terminated") is not False:
                raise ValueError(
                    f"{label} episode {episode_key} step {row.get('step_index')} must set "
                    "collector_terminated to false"
                )
            missing_transition_fields = sorted(
                field
                for field in ("actions", "rewards", "terminations", "truncations", "infos")
                if row.get(field) is None
            )
            if missing_transition_fields:
                raise ValueError(
                    f"{label} episode {episode_key} step {row.get('step_index')} has null "
                    f"transition fields: {', '.join(missing_transition_fields)}"
                )
        if len(row_indices) < 2:
            raise ValueError(f"{label} episode {episode_key} has no recorded transitions")
        final_transition = dataset[row_indices[-2]]
        provider_ended = bool(final_transition.get("terminations")) or bool(
            final_transition.get("truncations")
        )
        if collector_terminated == provider_ended:
            expected = (
                "false after a provider termination or truncation"
                if provider_ended
                else "true when the provider did not end the trajectory"
            )
            raise ValueError(
                f"{label} episode {episode_key} collector_terminated must be {expected}"
            )
    return dataset


def _validate_environment_artifacts(dataset, root, *, label="dataset"):
    contract_ids = set(dataset["environment_contract_id"])
    for contract_id in sorted(contract_ids):
        if (
            not isinstance(contract_id, str)
            or len(contract_id) != 64
            or any(char not in "0123456789abcdef" for char in contract_id.lower())
        ):
            raise ValueError(f"{label} has an invalid environment_contract_id {contract_id!r}")
        if not root:
            raise ValueError(f"{label} is missing environment artifacts for {contract_id}")
        relative_path = (
            f"{ENVIRONMENT_ARTIFACT_DIR}/{contract_id}/{ENVIRONMENT_DOCUMENT_FILENAME}"
        )
        path = os.path.join(root, relative_path)
        if not os.path.isfile(path):
            raise ValueError(f"{label} is missing {relative_path}")
        with open(path) as stream:
            document = json.load(stream)
        validate_environment_document(document, expected_id=contract_id)
        matching_rows = [
            index
            for index, value in enumerate(dataset["environment_contract_id"])
            if value == contract_id
        ]
        if any(
            dataset[index]["provider_id"] != document["provider_id"]
            or dataset[index]["env_id"] != document["environment_id"]
            for index in matching_rows
        ):
            raise ValueError(f"{label} rows conflict with environment contract {contract_id}")
    return dataset


def _validate_collector_artifacts(dataset, root, *, label="dataset"):
    contract_ids = {value for value in dataset["collector_contract_id"] if value is not None}
    for contract_id in sorted(contract_ids):
        if (
            not isinstance(contract_id, str)
            or len(contract_id) != 64
            or any(char not in "0123456789abcdef" for char in contract_id.lower())
        ):
            raise ValueError(f"{label} has an invalid collector_contract_id {contract_id!r}")
        if not root:
            raise ValueError(f"{label} is missing collector artifacts for {contract_id}")
        collector_dir = os.path.join(root, COLLECTOR_ARTIFACT_DIR, contract_id)
        collection_path = os.path.join(collector_dir, "collection.json")
        if not os.path.isfile(collection_path):
            raise ValueError(f"{label} is missing {collection_path}")
        collection = _load_json_document(
            collection_path, label=f"collector {contract_id}/collection.json"
        )
        if (
            collection.get("document_type") != COLLECTION_DOCUMENT_TYPE
            or collection.get("format_version") != COLLECTION_FORMAT_VERSION
        ):
            raise ValueError(f"{label} has an unsupported collector contract {contract_id}")
        actual_id = hashlib.sha256(_canonical_json_bytes(collection)).hexdigest()
        if actual_id != contract_id:
            raise ValueError(
                f"{label} collector contract hash mismatch: {actual_id} != {contract_id}"
            )
        source = collection.get("source") or {}
        expected_files = {"collection.json"}
        for key in ("model", "recipe", "release_manifest"):
            binding = source.get(key)
            if binding is None and key == "release_manifest":
                continue
            if not isinstance(binding, dict):
                raise ValueError(f"{label} collector {contract_id} is missing source.{key}")
            filename = binding.get("filename")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"{label} collector {contract_id} has invalid source.{key}")
            expected_files.add(filename)
            artifact_path = os.path.join(collector_dir, filename)
            if not os.path.isfile(artifact_path):
                raise ValueError(f"{label} is missing collector artifact {artifact_path}")
            if _sha256_file(artifact_path) != binding.get("sha256"):
                raise ValueError(f"{label} collector artifact hash mismatch: {artifact_path}")
        actual_files = {
            name
            for name in os.listdir(collector_dir)
            if os.path.isfile(os.path.join(collector_dir, name))
        }
        if actual_files != expected_files:
            raise ValueError(f"{label} collector directory {contract_id} contains unexpected files")
    return dataset


def _load_dataset_environment_documents(dataset, *, label="dataset"):
    contract_ids = sorted(set(dataset["environment_contract_id"]))
    local_root = _first_dataset_value(dataset, RUNTIME_VIDEO_BASE_COLUMN)
    hf_repo_id = _first_dataset_value(dataset, RUNTIME_HF_REPO_COLUMN)
    documents = {}
    for contract_id in contract_ids:
        relative_path = (
            f"{ENVIRONMENT_ARTIFACT_DIR}/{contract_id}/{ENVIRONMENT_DOCUMENT_FILENAME}"
        )
        if local_root:
            path = os.path.join(local_root, relative_path)
        elif hf_repo_id:
            path = hf_hub_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                filename=relative_path,
            )
        else:
            raise ValueError(f"{label} has no source for environment {contract_id}")
        with open(path) as stream:
            document = json.load(stream)
        validate_environment_document(document, expected_id=contract_id)
        documents[contract_id] = document
    return documents


def _session_from_environment_document(document, *, render_mode):
    expected_version = (document.get("provenance") or {}).get("version")
    provider_id = document["provider_id"]
    actual_version = _installed_package_version(provider_id)
    if actual_version != expected_version:
        raise ValueError(
            f"Environment contract requires {provider_id}=={expected_version}; "
            f"installed version is {actual_version or 'missing'}"
        )
    contract = EnvironmentContract.parse(
        {
            "contract_version": document["provider_contract_version"],
            "provider_id": provider_id,
            "environment_id": document["environment_id"],
            "config": document["effective_config"],
        },
        label="recorded environment",
    )
    session = create_session(contract, render_mode=render_mode)
    if _canonical_json_bytes(session.effective_config) != _canonical_json_bytes(
        document["effective_config"]
    ):
        session.env.close()
        raise ValueError(
            "Installed provider effective config does not match the recorded environment contract"
        )
    if _canonical_json_bytes(session.provenance) != _canonical_json_bytes(document["provenance"]):
        session.env.close()
        raise ValueError("Installed provider assets do not match the recorded environment contract")
    if _canonical_json_bytes(space_contract(session.env.action_space)) != _canonical_json_bytes(
        document["action_space"]
    ):
        session.env.close()
        raise ValueError("Installed provider action space does not match the recording")
    if _canonical_json_bytes(space_contract(session.env.observation_space)) != _canonical_json_bytes(
        document["observation_space"]
    ):
        session.env.close()
        raise ValueError("Installed provider observation space does not match the recording")
    return contract, session


class DatasetRecorderWrapper(gym.Wrapper):
    """Record and replay the final Gymnasium contract exposed by a provider session."""

    def __init__(
        self,
        *,
        provider_session,
        environment_artifact,
        input_source=None,
        headless=False,
        collector="human",
        storage_format=None,
        live_upload_manager=None,
        initial_seed=None,
        collector_contract=None,
    ):
        _lazy_init()
        super().__init__(provider_session.env)

        self.recording = False
        self.storage_format = _configured_storage_format(storage_format)
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
        self.collector_contract = collector_contract
        self.provider_session = provider_session
        self.environment_artifact = environment_artifact
        self._gymrec_version = _get_gymrec_version()

        if not headless:
            pygame.init()
            # pygame.display.set_caption will be set after env_id is available

        self.current_keys = set()
        self.key_lock = threading.Lock()

        self._recording_rows = []
        self._current_episode_uuid = None
        self._current_episode_seed = None
        self._session_uuid = None

        self.temp_dir = tempfile.mkdtemp()
        _materialize_collector_contract(self.collector_contract, self.temp_dir)
        _materialize_environment_artifact(self.environment_artifact, self.temp_dir)
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
        self._recording_rows.clear()
        self._current_episode_video_frames.clear()
        self._current_episode_video_hashes.clear()

    def _recording_context_row(self, episode_uuid, step_index):
        contract = self.collector_contract
        return _canonical_dataset_row(
            episode_id=str(episode_uuid),
            step_index=int(step_index),
            seed=self._current_episode_seed,
            session_id=str(self._session_uuid),
            dataset_format_version=DATASET_FORMAT_VERSION,
            collector=self.collector,
            gymrec_version=self._gymrec_version,
            storage_format=self.storage_format,
            provider_id=self.provider_session.provider_id,
            env_id=self.provider_session.environment_id,
            environment_contract_id=self.environment_artifact.contract_id,
            collector_contract_id=contract.contract_id if contract else None,
            policy_mode=contract.policy_mode if contract else None,
            policy_seed=self._current_policy_seed if contract else None,
            collector_terminated=False,
        )

    def _build_recorded_dataset(self):
        data = {
            field.name: [row.get(field.name) for row in self._recording_rows]
            for field in _dataset_fields(self.storage_format)
        }
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
        if not self._live_upload_enabled or self._live_episode is None or not self._recording_rows:
            return
        row_index = len(self._recording_rows) - 1
        row = self._recording_rows[row_index]
        if row["rewards"] is None:
            return

        terminal_frame = _observation_to_rgb_array(
            _recording_observation(self.provider_session, next_observation)
        )
        PILImage.fromarray(terminal_frame).save(
            self._live_episode.terminal_candidate_path,
            format="WEBP",
            lossless=True,
            method=6,
        )
        journal_row = dict(row)
        if self.storage_format == STORAGE_FORMAT_IMAGES:
            journal_row["observations"] = self._relative_live_package_path(row["observations"])

        record = {
            "type": "step",
            "row_index": row_index,
            "row": journal_row,
            "fps": self._fps or _provider_fps(self.provider_session),
            "terminal_candidate_path": self._relative_live_package_path(
                self._live_episode.terminal_candidate_path
            ),
            "terminal_candidate_sha256": _sha256_rgb(terminal_frame),
            "terminal_candidate_width": int(terminal_frame.shape[1]),
            "terminal_candidate_height": int(terminal_frame.shape[0]),
        }
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
        path = os.path.join(base_dir, f"frame_{len(self._recording_rows):05d}.webp")
        img = PILImage.fromarray(frame_uint8)
        img.save(path, format="WEBP", lossless=True, method=6)
        return path

    def _record_observation(self, episode_uuid, frame):
        """Record an observation through the configured storage backend."""
        if self.storage_format == STORAGE_FORMAT_IMAGES:
            return {"observations": self._save_frame_image(frame)}

        frame_array = _observation_to_rgb_array(frame)
        frame_hash = _sha256_rgb(frame_array)
        video_relpath = f"{VIDEO_ARTIFACT_DIR}/{episode_uuid.hex}{CANONICAL_VIDEO_SUFFIX}"

        if self._live_upload_enabled:
            if self._live_episode is None:
                self._start_live_episode(episode_uuid)
            video_path = os.path.join(self._live_episode.package_dir, video_relpath)
            if self._live_video_writer is None:
                self._live_video_writer = _StreamingLosslessVideoWriter(
                    video_path,
                    self._fps or _provider_fps(self.provider_session),
                    ffmpeg_path=self._ffmpeg_path,
                )
            self._live_video_writer.write(frame_array)
        else:
            self._current_episode_video_frames.append(frame_array.copy())
        self._current_episode_video_hashes.append(frame_hash)
        return {
            "video_path": video_relpath,
            "frame_sha256": frame_hash,
            "frame_height": int(frame_array.shape[0]),
            "frame_width": int(frame_array.shape[1]),
        }

    def _finalize_video_episode(self, episode_uuid):
        """Encode and verify the current episode's canonical observation video."""
        if self.storage_format != STORAGE_FORMAT_LOSSLESS_VIDEO:
            return
        if not self._current_episode_video_hashes:
            return

        video_relpath = f"{VIDEO_ARTIFACT_DIR}/{episode_uuid.hex}{CANONICAL_VIDEO_SUFFIX}"
        expected_hashes = list(self._current_episode_video_hashes)

        if self._live_upload_enabled:
            if self._live_video_writer is None:
                raise RuntimeError("Live video episode has no active ffmpeg writer")
            video_path = self._live_video_writer.output_path
            width = self._recording_rows[-1]["frame_width"]
            height = self._recording_rows[-1]["frame_height"]
            self._live_video_writer.close()
            self._live_video_writer = None
            _verify_lossless_rgb_video_stream(
                video_path,
                width,
                height,
                expected_hashes,
                ffmpeg_path=self._ffmpeg_path,
            )
            self._current_episode_video_hashes.clear()
            return

        video_path = os.path.join(self.temp_dir, video_relpath)
        preview_relpath = f"{VIDEO_ARTIFACT_DIR}/{episode_uuid.hex}{PREVIEW_VIDEO_SUFFIX}"
        preview_path = os.path.join(self.temp_dir, preview_relpath)
        _encode_lossless_rgb_video(
            self._current_episode_video_frames,
            video_path,
            self._fps or _provider_fps(self.provider_session),
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

        _encode_browser_preview_video(
            self._current_episode_video_frames,
            preview_path,
            self._fps or _provider_fps(self.provider_session),
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
            return int(action)

    def _record_frame(self, episode_uuid, step_index, frame, action, policy_action=None):
        """Save a frame and action to temporary storage."""
        if not self.recording:
            return

        row = self._recording_context_row(episode_uuid, step_index)
        row.update(
            {
                "actions": self._normalize_action(action),
                "policy_actions": (
                    self._normalize_action(policy_action) if policy_action is not None else None
                ),
                "rewards": None,
                "terminations": None,
                "truncations": None,
                "infos": None,
                **self._record_observation(episode_uuid, frame),
            }
        )
        self._recording_rows.append(row)

    def _record_terminal_observation(
        self, episode_uuid, step_index, frame, *, collector_terminated=False
    ):
        """Record a terminal observation row (Minari N+1 pattern).

        The N+1 observation captures the final state after the last step.
        It has null action/transition values
        since no step was taken.
        """
        if not self.recording:
            return

        row = self._recording_context_row(episode_uuid, step_index)
        row.update(
            {
                "actions": None,
                "policy_actions": None,
                "rewards": None,
                "terminations": None,
                "truncations": None,
                "infos": None,
                "collector_terminated": bool(collector_terminated),
                **self._record_observation(episode_uuid, frame),
            }
        )
        self._recording_rows.append(row)
        self._finalize_video_episode(episode_uuid)
        self._finish_live_episode()

    def _finish_live_episode(self):
        if not self._live_upload_enabled or not self._recording_rows:
            return
        package = self._live_episode
        if package is None:
            raise RuntimeError("Live upload episode was not initialized")
        dataset = self._build_recorded_dataset()
        success = self.live_upload_manager.upload_episode(package, dataset)
        if not success:
            console.print(
                f"[{STYLE_INFO}]Episode {package.episode_id} kept for retry: "
                f"[{STYLE_CMD}]"
                f"{_gymrec_cmd('upload', self.live_upload_manager.identity.display_ref)}[/]"
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
        """Print the provider-neutral keyboard control profile."""
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Key", justify="right", style=STYLE_KEY)
        table.add_column("Control", style=STYLE_ACTION)
        input_source = (
            self.input_source
            if isinstance(self.input_source, HumanInputSource)
            else HumanInputSource(self.provider_session, self.key_lock, self.current_keys)
        )
        for key, label in input_source.key_to_label.items():
            table.add_row(pygame.key.name(key), label)
        table.add_section()
        table.add_row("[dim]escape[/]", "[dim]Exit[/]")
        table.add_row("[dim]+/-[/]", "[dim]Adjust FPS (±5)[/]")
        provider_label = PROVIDER_LABELS.get(
            self.provider_session.provider_id, self.provider_session.provider_id
        )
        console.print(
            Panel(
                table,
                title=f"[{STYLE_ENV}]{provider_label}[/] Key Mappings",
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
        progress_callback=None,
        step_callback=None,
    ):
        """Record a gameplay session at the desired FPS.

        Args:
            fps: Frames per second for rendering (ignored if headless)
            max_episodes: Maximum number of episodes to record (None = unlimited for human, 1 for agent)
            progress_callback: Optional callable(episode_number, steps_in_episode) called after each episode
            step_callback: Optional callable(episode_number, step_number) called during each step for live updates
        """
        if fps is None:
            fps = _provider_fps(self.provider_session)
        self._max_episodes = max_episodes
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
        Supports both human input (pygame) and policy-backed input sources.
        """
        if fps is None:
            fps = _provider_fps(self.provider_session)
        self._fps = fps

        self._clear_recording_buffers()

        self._env_metadata = {
            "provider_id": self.provider_session.provider_id,
            "env_id": self.provider_session.environment_id,
            "environment_contract_id": self.environment_artifact.contract_id,
            "fps": self._fps,
        }

        self._session_uuid = uuid.uuid4()
        self._current_episode_uuid = uuid.uuid4()
        if self.initial_seed is None:
            raise RuntimeError("Recording requires an explicit base environment seed")
        self._current_episode_seed = self.initial_seed
        self._current_policy_seed = self._current_episode_seed
        seed = self._current_episode_seed
        self._episode_count = 1
        self._cumulative_reward = 0.0
        self._start_live_episode(self._current_episode_uuid)
        obs, _ = self.env.reset(seed=seed)

        # Setup input source
        if self.input_source is None:
            # Default to human input
            self.input_source = HumanInputSource(
                self.provider_session, self.key_lock, self.current_keys
            )
        elif hasattr(self.input_source, "reset"):
            self.input_source.reset(seed=self._current_policy_seed, observation=obs)

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
            policy_action = getattr(self.input_source, "policy_action", None)

            self._record_frame(
                self._current_episode_uuid,
                step,
                _recording_observation(self.provider_session, obs),
                action,
                policy_action=policy_action,
            )
            obs, reward, terminated, truncated, info = self.env.step(action)
            if hasattr(self.input_source, "observe_step"):
                self.input_source.observe_step(reward, terminated, truncated, info)
            self._cumulative_reward += float(reward)
            if self.recording:
                self._recording_rows[-1].update(
                    {
                        "rewards": float(reward),
                        "terminations": bool(terminated),
                        "truncations": bool(truncated),
                        "infos": json.dumps(info, default=_json_default),
                    }
                )

            self._render_frame(_recording_observation(self.provider_session, obs))

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

            self._write_live_step_journal(obs)

            if terminated or truncated:
                self._record_terminal_observation(
                    self._current_episode_uuid,
                    step,
                    _recording_observation(self.provider_session, obs),
                )

                if self._progress_callback is not None:
                    self._progress_callback(self._episode_count, step)

                # Check if we've reached the max episode count
                if self._max_episodes is not None and self._episode_count >= self._max_episodes:
                    break

                self._current_episode_uuid = uuid.uuid4()
                self._current_episode_seed = self.initial_seed + self._episode_count
                self._current_policy_seed = self._current_episode_seed
                seed = self._current_episode_seed
                self._start_live_episode(self._current_episode_uuid)
                obs, _ = self.env.reset(seed=seed)
                if hasattr(self.input_source, "reset"):
                    self.input_source.reset(seed=self._current_policy_seed, observation=obs)
                self._episode_count += 1
                self._cumulative_reward = 0.0
                step = 0
                self._render_frame(_recording_observation(self.provider_session, obs))

        # A user exit is a collector boundary, not an environment boundary. Keep
        # the useful trajectory segment without changing the provider's exact
        # termination or truncation values.
        if (
            self.recording
            and self._recording_rows
            and self._recording_rows[-1]["actions"] is not None
        ):
            self._record_terminal_observation(
                self._current_episode_uuid,
                step,
                _recording_observation(self.provider_session, obs),
                collector_terminated=True,
            )

        if self._live_episode is not None:
            if self._live_video_writer is not None:
                self._live_video_writer.abort()
                self._live_video_writer = None
            self.live_upload_manager.discard_episode(self._live_episode)
            self._live_episode = None

        if self.recording and self._recording_rows:
            self._recorded_dataset = self._build_recorded_dataset()

    def _convert_action(self, action):
        """Convert stored action back to the environment's expected format."""
        if isinstance(action, list):
            if isinstance(self.env.action_space, gym.spaces.Discrete) and len(action) == 1:
                return action[0]
            else:
                return np.array(action, dtype=self.env.action_space.dtype)
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
            fps = _provider_fps(self.provider_session)
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
                    display_obs = _recording_observation(self.provider_session, obs)
                    if not self.headless:
                        self._ensure_screen(display_obs)
                        self._render_frame(display_obs)
                        if not printed_keymappings:
                            self._print_keymappings()
                            printed_keymappings = True

                    for item in items:
                        frame_start = time.monotonic()
                        if not self.headless and not self._input_loop():
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
                        display_obs = _recording_observation(self.provider_session, obs)
                        if not self.headless:
                            self._render_frame(display_obs)

                        if verify:
                            obs_image = _extract_observation_image(display_obs)
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
    replacements = {"slash": "/", "dash": "-", "underscore": "_"}
    return re.sub(
        r"_(slash|dash|underscore)_",
        lambda match: replacements[match.group(1)],
        repo_name,
    )


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


@dataclass(frozen=True)
class RecordingPaths:
    dataset: str
    metadata: str
    uploaded: str
    live_queue: str


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


def _recording_paths(value):
    """Derive every local path for one recording identity in one place."""
    identity = _coerce_recording_identity(value)
    local_dir = CONFIG["storage"]["local_dir"]
    if identity.dataset_repo_id:
        owner, repo = identity.dataset_repo_id.split("/", 1)
        root = os.path.join(local_dir, "repos", owner, repo)
        return RecordingPaths(
            dataset=os.path.join(root, "dataset"),
            metadata=os.path.join(root, "metadata.json"),
            uploaded=os.path.join(root, "uploaded.json"),
            live_queue=os.path.join(root, "live_pending"),
        )

    encoded_env_id = _encode_env_id_for_hf(identity.env_id)
    return RecordingPaths(
        dataset=os.path.join(local_dir, encoded_env_id),
        metadata=os.path.join(local_dir, f"{encoded_env_id}_metadata.json"),
        uploaded=os.path.join(local_dir, f"{encoded_env_id}_uploaded.json"),
        live_queue=os.path.join(local_dir, f"{encoded_env_id}_live_pending"),
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


def get_local_dataset_path(value):
    """Return the local dataset path for an environment or repo-keyed identity."""
    return _recording_paths(value).dataset


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
    return _recording_paths(value).metadata


def _get_uploaded_episodes_path(value):
    """Return the uploaded-episode tracking path for a recording identity."""
    return _recording_paths(value).uploaded


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


@dataclass(frozen=True)
class RemoteDatasetState:
    revision: str | None
    episode_ids: frozenset[str]
    frames: int

    @property
    def episodes(self):
        return len(self.episode_ids)


def _remote_dataset_state(
    value,
    *,
    revision=None,
    repo_exists=None,
    remote_files=None,
):
    """Return row-derived statistics for one exact Hub dataset revision."""
    hf_repo_id = _identity_hf_repo_id(value)
    if repo_exists is None or remote_files is None:
        try:
            repo_exists, resolved_revision, remote_files = _hf_repo_state(
                HfApi(), hf_repo_id, create=False
            )
        except Exception:
            return None
        revision = revision or resolved_revision
    if not repo_exists:
        return RemoteDatasetState(revision, frozenset(), 0)
    has_data_shard = any(
        path.startswith("data/") and path.endswith(".parquet") for path in remote_files
    )
    if not has_data_shard:
        return RemoteDatasetState(revision, frozenset(), 0)

    try:
        dataset = load_dataset(
            hf_repo_id,
            split="train",
            streaming=True,
            revision=revision,
        )
        episode_ids = set()
        frames = 0
        for row in dataset:
            frames += 1
            if row.get("episode_id") is not None:
                episode_ids.add(_normalize_episode_id(row["episode_id"]))
        return RemoteDatasetState(revision, frozenset(episode_ids), frames)
    except Exception:
        return None


def _get_live_upload_queue_dir(value):
    """Return the local resumable live-upload queue directory."""
    return _recording_paths(value).live_queue


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
    fps=None,
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
    if fps is not None:
        entry["fps"] = max(int(round(float(fps))), 1)
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
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def _copy_dataset_artifacts(src_root, dst_root):
    _copy_artifact_tree(src_root, dst_root, VIDEO_ARTIFACT_DIR)
    _copy_artifact_tree(src_root, dst_root, COLLECTOR_ARTIFACT_DIR)
    _copy_artifact_tree(src_root, dst_root, ENVIRONMENT_ARTIFACT_DIR)


def save_dataset_locally(dataset, value, *, artifact_root, metadata=None):
    """Save dataset to local disk, appending to any existing data."""
    identity = _coerce_recording_identity(value)
    path = get_local_dataset_path(identity)
    metadata_path = _get_metadata_path(identity)
    dataset = _strip_runtime_columns(dataset)
    _validate_canonical_dataset_schema(dataset, label="New recording")
    _validate_environment_artifacts(dataset, artifact_root, label="New recording")
    _validate_collector_artifacts(dataset, artifact_root, label="New recording")
    new_storage_format = _dataset_storage_format(dataset)
    new_episode_count = (
        len(set(dataset["episode_id"])) if "episode_id" in dataset.column_names else 0
    )
    new_frame_count = len(dataset)
    existing_video_dir = path
    if os.path.exists(path):
        # Load existing dataset - UUIDs are already unique, no offsetting needed
        existing_dataset = load_from_disk(path, keep_in_memory=True)
        _validate_canonical_dataset_schema(existing_dataset, label=f"Existing dataset at {path}")
        _validate_environment_artifacts(
            existing_dataset, path, label=f"Existing dataset at {path}"
        )
        _validate_collector_artifacts(existing_dataset, path, label=f"Existing dataset at {path}")
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
            _copy_dataset_artifacts(existing_video_dir, tmp_path)
        _copy_dataset_artifacts(artifact_root, tmp_path)
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
        # Update descriptive metadata while preserving one canonical vocabulary.
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
    _validate_canonical_dataset_schema(dataset, label=f"Local dataset at {path}")
    _validate_environment_artifacts(dataset, path, label=f"Local dataset at {path}")
    _validate_collector_artifacts(dataset, path, label=f"Local dataset at {path}")
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

    _validate_canonical_dataset_schema(dataset, label=f"Remote dataset {hf_repo_id}")
    dataset = _attach_video_runtime_source(dataset, hf_repo_id=hf_repo_id)
    return dataset, "hub"


def _recording_env_id(value, dataset=None, metadata=None):
    """Recover a recording's logical environment from identity, metadata, or rows."""
    identity = _coerce_recording_identity(value)
    if identity.env_id:
        return identity.env_id
    metadata = metadata or load_local_metadata(identity) or {}
    env_id = metadata.get("env_id")
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
    """Return the required, uniform observation storage format declared by a dataset."""
    column_names = getattr(dataset, "column_names", None) or []
    if "storage_format" not in column_names:
        raise ValueError("Dataset is missing the required storage_format column")

    values = list(dataset["storage_format"])
    if not values or any(value in (None, "") for value in values):
        raise ValueError("Dataset must declare storage_format on every row")

    formats = {_normalize_storage_format(value) for value in values}
    if len(formats) != 1:
        raise ValueError("Dataset contains mixed storage_format values")
    return formats.pop()


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


def _normalize_episode_id(eid):
    """Normalize episode identifiers to a canonical readable UUID when possible."""
    if isinstance(eid, bytes):
        try:
            return str(uuid.UUID(bytes=eid))
        except ValueError:
            return eid.hex()
    if isinstance(eid, uuid.UUID):
        return str(eid)
    try:
        return str(uuid.UUID(str(eid)))
    except (ValueError, AttributeError):
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

    if "step_index" in (getattr(dataset, "column_names", None) or []):
        for row_indices in row_indices_by_episode.values():
            row_indices.sort(key=lambda index: int(dataset[index]["step_index"]))

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
        contract_ids = {dataset[index]["environment_contract_id"] for index in row_indices}
        if len(contract_ids) != 1:
            raise ValueError(f"Episode {episode_key} spans multiple environment contracts")

        episodes.append(
            {
                "environment_contract_id": contract_ids.pop(),
                "seed": _episode_reset_seed(dataset, row_indices),
                "step_count": step_count,
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
    frame_index = int(row.get("step_index"))
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
    return f"{VIDEO_ARTIFACT_DIR}/{episode_stem}{PREVIEW_VIDEO_SUFFIX}"


def _dataset_replay_url(repo_id):
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{DATASET_REPLAY_FILENAME}"


def _dataset_card_has_replay(card_content):
    """Return whether a dataset card contains the playable root replay contract."""
    return "<video" in card_content and DATASET_REPLAY_FILENAME in card_content


def _remote_dataset_publication_needs_repair(repo_id, revision, remote_files):
    """Return whether replay.mp4 or its card player is missing at a pinned revision."""
    if DATASET_REPLAY_FILENAME not in remote_files or "README.md" not in remote_files:
        return True
    readme_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename="README.md",
    )
    with open(readme_path) as file_obj:
        return not _dataset_card_has_replay(file_obj.read())


def _materialize_dataset_replay(dataset, local_root, output_path, fps):
    """Create the root replay from the first finalized trajectory in dataset order."""
    episode_keys, rows_by_episode = _ordered_episode_rows(dataset)
    if not episode_keys:
        raise ValueError("Cannot publish replay.mp4 from an empty dataset")
    row_indices = None
    for episode_key in episode_keys:
        candidate_indices = rows_by_episode[episode_key]
        if len(candidate_indices) < 2:
            continue
        candidate_rows = [dataset[index] for index in candidate_indices]
        if not _is_terminal_action(candidate_rows[-1].get("actions")):
            continue
        step_rows = [row for row in candidate_rows[:-1] if _is_step_row(row)]
        provider_ended = any(
            bool(row.get("terminations")) or bool(row.get("truncations")) for row in step_rows
        )
        collector_ended = bool(candidate_rows[-1].get("collector_terminated"))
        if step_rows and (provider_ended or collector_ended):
            row_indices = candidate_indices
            break
    if row_indices is None:
        raise ValueError("Cannot publish replay.mp4 without a finalized trajectory")

    storage_format = _dataset_storage_format(dataset)
    if storage_format == STORAGE_FORMAT_LOSSLESS_VIDEO:
        first_row = dataset[row_indices[0]]
        video_relpath = first_row.get("video_path")
        if not video_relpath:
            raise ValueError("Representative episode is missing its canonical video path")
        canonical_path = os.path.join(local_root, video_relpath)
        if not os.path.isfile(canonical_path):
            raise FileNotFoundError(f"Missing representative video artifact: {canonical_path}")
        video_name = os.path.basename(video_relpath)
        if not video_name.endswith(CANONICAL_VIDEO_SUFFIX):
            raise ValueError(f"Unsupported canonical video path for replay: {video_relpath}")
        episode_stem = video_name[: -len(CANONICAL_VIDEO_SUFFIX)]
        preview_path = os.path.join(local_root, _preview_video_relpath(episode_stem))
        if os.path.isfile(preview_path):
            shutil.copy2(preview_path, output_path)
        else:
            _transcode_browser_preview_video(canonical_path, output_path)
    else:
        writer = _StreamingBrowserPreviewWriter(output_path, fps)
        try:
            for row_index in row_indices:
                frame = _observation_to_rgb_array(_get_row_observation(dataset[row_index]))
                writer.write(frame)
            writer.close()
        except Exception:
            writer.abort()
            raise

    _verify_browser_preview_video(output_path)
    return output_path


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


def _validate_remote_parquet_schema(hf_repo_id, remote_columns, storage_format):
    if remote_columns is None:
        return
    expected = _canonical_column_order(storage_format)
    if list(remote_columns) != expected:
        raise ValueError(
            f"Remote dataset {hf_repo_id} does not use the current canonical gymrec schema. "
            "Appending to legacy or mismatched schemas is unsupported; use --replace with a "
            "fully new-schema local dataset."
        )


def _contract_upload_operations(
    dataset,
    *,
    local_root,
    hf_repo_id,
    remote_files,
    revision,
):
    artifacts = []
    for contract_id in sorted(set(dataset["environment_contract_id"])):
        repo_path = (
            f"{ENVIRONMENT_ARTIFACT_DIR}/{contract_id}/{ENVIRONMENT_DOCUMENT_FILENAME}"
        )
        local_path = os.path.join(local_root, repo_path)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Missing environment contract: {local_path}")
        artifacts.append(("environment", repo_path, local_path))

    collector_ids = {value for value in dataset["collector_contract_id"] if value is not None}
    for contract_id in sorted(collector_ids):
        relative_dir = f"{COLLECTOR_ARTIFACT_DIR}/{contract_id}"
        local_dir = os.path.join(local_root, relative_dir)
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Missing collector contract directory: {local_dir}")
        for filename in sorted(os.listdir(local_dir)):
            local_path = os.path.join(local_dir, filename)
            if not os.path.isfile(local_path):
                continue
            repo_path = f"{relative_dir}/{filename}"
            artifacts.append(("collector", repo_path, local_path))

    operations = []
    for contract_kind, repo_path, local_path in artifacts:
        if repo_path in remote_files:
            remote_path = hf_hub_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                revision=revision,
                filename=repo_path,
            )
            if _sha256_file(remote_path) != _sha256_file(local_path):
                raise ValueError(f"Remote {contract_kind} contract conflict at {repo_path}")
            continue
        operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path))
    return operations


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
    preview_fps=None,
    publish_data=True,
    remote_state=None,
    max_retries=5,
    base_wait=1.0,
):
    """Upload an already-materialized dataset shard and its artifacts to the Hub."""
    identity = _coerce_recording_identity(recording_identity)
    env_id = _recording_env_id(identity, dataset=dataset) or "unknown"
    hf_repo_id = _identity_hf_repo_id(identity)
    api = HfApi()
    episode_ids = set(episode_ids)
    upload_dataset = _strip_runtime_columns(dataset)
    _validate_canonical_dataset_schema(upload_dataset, label="Upload shard")
    _validate_environment_artifacts(upload_dataset, local_root, label="Upload shard")
    _validate_collector_artifacts(upload_dataset, local_root, label="Upload shard")
    local_metadata = load_local_metadata(identity) or {}
    preview_fps = preview_fps or local_metadata.get("fps")
    if preview_fps is None:
        raise ValueError("Upload requires recorded provider fps metadata")

    for attempt in range(1, max_retries + 1):
        try:
            repo_exists, parent_commit, remote_files = _hf_repo_state(api, hf_repo_id, create=False)
            if repo_exists and not replace:
                remote_format = _remote_storage_format_from_files(remote_files)
                conflict_message = _remote_storage_conflict_message(
                    identity.display_ref, hf_repo_id, storage_format, remote_format
                )
                if conflict_message:
                    console.print(f"[{STYLE_FAIL}]{conflict_message}[/]")
                    return False
                _validate_remote_parquet_schema(
                    hf_repo_id,
                    _remote_parquet_columns(hf_repo_id, remote_files),
                    storage_format,
                )

            if replace:
                current_remote_state = RemoteDatasetState(parent_commit, frozenset(), 0)
            elif not repo_exists:
                current_remote_state = RemoteDatasetState(parent_commit, frozenset(), 0)
            else:
                current_remote_state = remote_state
                if current_remote_state is None or current_remote_state.revision != parent_commit:
                    current_remote_state = _remote_dataset_state(
                        identity,
                        revision=parent_commit,
                        repo_exists=repo_exists,
                        remote_files=remote_files,
                    )
                if current_remote_state is None:
                    raise RuntimeError(
                        f"Could not read remote dataset rows at revision {parent_commit}"
                    )

            next_shard_idx = _next_hf_shard_index(api, hf_repo_id, repo_exists, replace)
            operations = []
            if replace:
                operations.extend(
                    CommitOperationDelete(path_in_repo=path)
                    for path in remote_files
                    if path != ".gitattributes"
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                if publish_data:
                    shard_path = os.path.join(tmpdir, "shard.parquet")
                    upload_dataset.to_parquet(shard_path)
                    shard_name = (
                        "data/train-00000-of-00001.parquet"
                        if replace
                        else f"data/train-{next_shard_idx:05d}-of-{next_shard_idx + 1:05d}.parquet"
                    )
                    operations.append(
                        CommitOperationAdd(path_in_repo=shard_name, path_or_fileobj=shard_path)
                    )
                    operations.extend(
                        _contract_upload_operations(
                            upload_dataset,
                            local_root=local_root,
                            hf_repo_id=hf_repo_id,
                            remote_files=[] if replace else remote_files,
                            revision=parent_commit,
                        )
                    )

                    sidecar_metadata = _playback_metadata(dataset, local_metadata)
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

                if replace or DATASET_REPLAY_FILENAME not in remote_files:
                    replay_path = os.path.join(tmpdir, DATASET_REPLAY_FILENAME)
                    _materialize_dataset_replay(
                        upload_dataset,
                        local_root,
                        replay_path,
                        preview_fps,
                    )
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=DATASET_REPLAY_FILENAME,
                            path_or_fileobj=replay_path,
                        )
                    )

                card_content = _build_dataset_card_content(
                    identity,
                    env_id,
                    hf_repo_id,
                    new_frames=len(upload_dataset) if publish_data else 0,
                    new_episodes=len(episode_ids) if publish_data else 0,
                    existing_frames=current_remote_state.frames,
                    existing_episodes=current_remote_state.episodes,
                    local_root=local_root,
                    remote_files=[] if replace else remote_files,
                    dataset=upload_dataset,
                    fps=preview_fps,
                )
                if card_content:
                    card_path = os.path.join(tmpdir, "README.md")
                    with open(card_path, "w") as f:
                        f.write(card_content)
                    operations.append(
                        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_path)
                    )

                if not repo_exists:
                    created, created_parent, created_files = _hf_repo_state(
                        api, hf_repo_id, create=True
                    )
                    if not created:
                        raise RuntimeError(f"Could not create dataset repository {hf_repo_id}")
                    if any(path != ".gitattributes" for path in created_files):
                        raise RuntimeError(
                            "Conflict detected: dataset repository was populated during upload"
                        )
                    parent_commit = created_parent

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
                        else (
                            f"Add recordings from {env_id}"
                            if publish_data
                            else f"Repair dataset preview for {env_id}"
                        )
                    ),
                    parent_commit=parent_commit,
                )

            if publish_data:
                uploaded_episode_ids = (
                    episode_ids if replace else _load_uploaded_episode_ids(identity) | episode_ids
                )
                _save_uploaded_episode_ids(identity, uploaded_episode_ids)
            action = "Dataset uploaded" if publish_data else "Dataset preview repaired"
            console.print(
                f"[{STYLE_SUCCESS}]{action}: https://huggingface.co/datasets/{hf_repo_id}[/]"
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
    rows = [_row_from_live_journal_record(record) for record in records]
    first = rows[0]
    last = records[-1]
    video_relpath = first["video_path"]
    video_path = _package_path(package_dir, video_relpath)
    width = int(first["frame_width"])
    height = int(first["frame_height"])
    step_hashes = [row["frame_sha256"] for row in rows]
    terminal_path = _package_path(package_dir, last["terminal_candidate_path"])
    terminal_frame = _observation_to_rgb_array(PILImage.open(terminal_path))
    terminal_hash = _sha256_rgb(terminal_frame)
    fps = last.get("fps")
    if fps is None:
        raise ValueError("Live recording journal is missing provider fps")

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


def _row_from_live_journal_record(record):
    """Read current canonical journals and the pre-canonical local journal format."""
    if isinstance(record.get("row"), dict):
        return dict(record["row"])

    row = _canonical_dataset_row(
        episode_id=record.get("episode_id"),
        step_index=record.get("step_index"),
        seed=record.get("seed"),
        actions=record.get("action"),
        policy_actions=record.get("policy_action"),
        rewards=record.get("reward"),
        terminations=record.get("termination"),
        truncations=record.get("truncation"),
        infos=record.get("info"),
        session_id=record.get("session_id"),
        **{
            name: record.get(name)
            for name in ROW_CONTEXT_FIELD_NAMES
            if name
            not in {
                "episode_id",
                "step_index",
                "seed",
                "session_id",
            }
        },
    )
    storage_format = _normalize_storage_format(row["storage_format"])
    if storage_format == STORAGE_FORMAT_IMAGES:
        row["observations"] = record.get("observation_path")
    else:
        row.update({field.name: record.get(field.name) for field in VIDEO_DATASET_FIELDS})
    return row


def _dataset_from_recovered_live_records(package_dir, episode_id, records):
    if not records:
        raise RuntimeError(f"No recoverable completed steps found for {episode_id}")
    records = sorted(records, key=lambda record: int(record.get("row_index", 0)))
    rows = [_row_from_live_journal_record(record) for record in records]
    storage_format = _normalize_storage_format(rows[0]["storage_format"])
    last_record = records[-1]

    episode_uuid = str(uuid.UUID(str(episode_id)))
    session_uuid = str(uuid.UUID(str(rows[0]["session_id"])))
    for row in rows:
        row["episode_id"] = episode_uuid
        row["step_index"] = int(row["step_index"])
        row["session_id"] = session_uuid
        row["storage_format"] = storage_format
        row["collector_terminated"] = False

    terminal_row = _canonical_dataset_row(
        **{name: rows[0].get(name) for name in ROW_CONTEXT_FIELD_NAMES},
    )
    terminal_row.update(
        {
            "episode_id": episode_uuid,
            "step_index": len(rows),
            "session_id": session_uuid,
            "storage_format": storage_format,
            "collector_terminated": True,
        }
    )

    if storage_format == STORAGE_FORMAT_IMAGES:
        for row in rows:
            row["observations"] = _package_path(package_dir, row["observations"])
        terminal_row["observations"] = _package_path(
            package_dir, last_record["terminal_candidate_path"]
        )
    else:
        video_relpath, terminal_hash, _ = _recover_live_video_artifact(package_dir, records)
        for row in rows:
            row["frame_width"] = int(row["frame_width"])
            row["frame_height"] = int(row["frame_height"])
        terminal_row.update(
            {
                "video_path": video_relpath,
                "frame_sha256": terminal_hash,
                "frame_width": int(last_record["terminal_candidate_width"]),
                "frame_height": int(last_record["terminal_candidate_height"]),
            }
        )

    rows.append(terminal_row)
    data = {
        field.name: [row.get(field.name) for row in rows]
        for field in _dataset_fields(storage_format)
    }
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
    remote_state = _remote_dataset_state(identity)
    if remote_state is None:
        already_uploaded = _load_uploaded_episode_ids(identity)
    else:
        already_uploaded = set(remote_state.episode_ids)
    for episode_id, entry in entries:
        package_dir = entry["package_dir"]
        if episode_id in already_uploaded:
            _set_live_upload_manifest_entry(
                identity,
                episode_id,
                state="uploaded",
                package_dir=package_dir,
                storage_format=_normalize_storage_format(entry.get("storage_format")),
                frames=entry.get("frames", 0),
                fps=entry.get("fps"),
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
                fps=entry.get("fps"),
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
                storage_format=_normalize_storage_format(entry.get("storage_format")),
                frames=entry.get("frames", 0),
                fps=entry.get("fps"),
                error=e,
            )
            console.print(f"[{STYLE_FAIL}]Pending live upload failed for {episode_id}: {e}[/]")
    return ok


def preflight_live_upload(recording_identity, storage_format):
    """Validate live upload can reach the target Hub dataset before gameplay starts."""
    identity = _coerce_recording_identity(recording_identity)
    if not ensure_hf_login():
        return False
    try:
        _require_dataset_replay_tools()
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
    try:
        _validate_remote_parquet_schema(
            hf_repo_id,
            _remote_parquet_columns(hf_repo_id, remote_files),
            storage_format,
        )
    except ValueError as exc:
        console.print(f"[{STYLE_FAIL}]{exc}[/]")
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

    remote_state = _remote_dataset_state(identity)
    if remote_state is None:
        already_uploaded = _load_uploaded_episode_ids(identity)
    else:
        already_uploaded = set(remote_state.episode_ids)
    new_indices = []
    new_episode_ids = set()
    for i, row in enumerate(local_dataset):
        eid = _normalize_episode_id(row["episode_id"])
        if replace or eid not in already_uploaded:
            new_indices.append(i)
            new_episode_ids.add(eid)

    if not new_indices:
        try:
            hf_repo_id = _identity_hf_repo_id(identity)
            repo_exists, revision, remote_files = _hf_repo_state(HfApi(), hf_repo_id, create=False)
            remote_format = _remote_storage_format_from_files(remote_files)
            conflict_message = _remote_storage_conflict_message(
                identity.display_ref, hf_repo_id, storage_format, remote_format
            )
            if conflict_message:
                console.print(f"[{STYLE_FAIL}]{conflict_message}[/]")
                return False
            needs_repair = repo_exists and _remote_dataset_publication_needs_repair(
                hf_repo_id,
                revision,
                remote_files,
            )
        except Exception as exc:
            console.print(f"[{STYLE_FAIL}]Could not inspect remote dataset publication: {exc}[/]")
            return False
        if needs_repair:
            console.print(f"[{STYLE_INFO}]Repairing missing dataset preview artifacts...[/]")
            repaired = _upload_dataset_shard_to_hub(
                identity,
                local_dataset,
                storage_format=storage_format,
                local_root=get_local_dataset_path(identity),
                episode_ids=set(),
                include_previews=False,
                publish_data=False,
                remote_state=remote_state,
                max_retries=max_retries,
                base_wait=base_wait,
            )
            return pending_ok and repaired
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
        remote_state=remote_state,
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

    documents = _load_dataset_environment_documents(dataset, label="Local dataset")
    if len(documents) != 1:
        raise ValueError("Minari export requires one environment contract")
    _contract, session = _session_from_environment_document(
        next(iter(documents.values())), render_mode="rgb_array"
    )
    action_space = session.env.action_space
    observation_space = session.env.observation_space
    session.env.close()

    # Build EpisodeBuffers
    buffers = []
    total_steps = 0
    for ep_idx, episode_key in enumerate(sorted(episode_keys)):
        rows = [dataset[row_index] for row_index in row_indices_by_episode[episode_key]]
        if "step_index" in rows[0]:
            rows.sort(key=lambda row: row["step_index"])
        if bool(rows[-1].get("collector_terminated")):
            console.print(
                f"[{STYLE_FAIL}]Episode {ep_idx} ended at a collector boundary. Minari cannot "
                "represent it without fabricating an environment truncation.[/]"
            )
            return False
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

            # Detect the terminal observation by its null action.
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
    create_kwargs["action_space"] = action_space
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


def _dataset_card_intro(env_id, collectors):
    if collectors and collectors != ["human"]:
        collector_str = ", ".join(f"`{c}`" for c in collectors)
        return (
            f"Gameplay recordings (collected by: {collector_str}) from the Gymnasium "
            f"environment `{env_id}`,"
        )
    return f"Human gameplay recordings from the Gymnasium environment `{env_id}`,"


def _dataset_card_environment_lines(metadata):
    if not metadata:
        return []
    lines = [
        "## Environment Contract",
        "",
        "| Setting | Value |",
        "|---------|-------|",
    ]
    for key, label in (
        ("provider_id", "Provider"),
        ("environment_contract_id", "Contract ID"),
        ("fps", "Target FPS"),
    ):
        if key in metadata:
            value = f"`{metadata[key]}`" if key == "environment_contract_id" else metadata[key]
            lines.append(f"| {label} | {value} |")
    lines.append("")
    return lines


_DATASET_CARD_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "gymrec_templates")
    ),
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


def _render_dataset_card_template(context):
    return _DATASET_CARD_TEMPLATE_ENV.get_template("dataset_card.md.j2").render(**context)


def render_dataset_card_content(
    env_id,
    repo_id,
    frames,
    episodes,
    metadata=None,
    collectors=None,
    gymrec_versions=None,
    collector_contracts=None,
    curator=None,
):
    """Render the shared Hugging Face dataset card Markdown."""
    metadata = dict(metadata or {})
    provider_id = metadata.get("provider_id")
    if provider_id not in SUPPORTED_PROVIDER_IDS:
        raise ValueError("Dataset card requires a supported provider_id")
    collectors = collectors or []
    gymrec_versions = gymrec_versions or []
    collector_contracts = collector_contracts or []
    curator = curator or _current_hf_username()
    storage_format = _configured_storage_format(metadata.get("storage_format"))
    card_data = DatasetCardData(
        language="en",
        license=CONFIG["dataset"]["license"],
        task_categories=CONFIG["dataset"]["task_categories"],
        tags=["gymnasium", provider_id, env_id],
        size_categories=[_size_category(frames)],
        pretty_name=f"{env_id} Gameplay Dataset",
    )

    summary_optional_rows = []
    if collectors:
        summary_optional_rows.append(f"| Collector(s) | {', '.join(collectors)} |")
    if gymrec_versions:
        summary_optional_rows.append(f"| gymrec version(s) | {', '.join(gymrec_versions)} |")
    summary_optional_rows = "\n" + "\n".join(summary_optional_rows) if summary_optional_rows else ""

    collector_contract_lines = []
    if collector_contracts:
        collector_contract_lines.extend(
            (
                "## Collector Contracts",
                "",
                "| Contract ID | Immutable model source | Mode | Files |",
                "|-------------|------------------------|------|-------|",
            )
        )
        for contract in collector_contracts:
            contract_id = contract["contract_id"]
            source = f"hf://{contract.get('repo_id')}@{contract.get('revision')}"
            file_names = ["model.json", "recipe.json"]
            if contract.get("has_release_manifest"):
                file_names.append("release_manifest.json")
            file_names.append("collection.json")
            files = ", ".join(
                f"[`{filename}`](collectors/{contract_id}/{filename})" for filename in file_names
            )
            collector_contract_lines.append(
                f"| `{contract_id}` | `{source}` | `{contract.get('policy_mode')}` | {files} |"
            )
        collector_contract_lines.append("")
    collector_contracts_block = (
        "\n".join(collector_contract_lines) + "\n" if collector_contract_lines else ""
    )

    field_lines = [field.card_line() for field in _dataset_fields(storage_format)]
    if storage_format == STORAGE_FORMAT_LOSSLESS_VIDEO:
        field_lines.append(
            f"- Browser-friendly `*{PREVIEW_VIDEO_SUFFIX}` files under "
            f"`{VIDEO_ARTIFACT_DIR}/` are lossy previews only and are not used "
            "for trajectory replay/training"
        )

    environment_lines = _dataset_card_environment_lines(metadata)
    environment_block = "\n".join(environment_lines) + "\n" if environment_lines else ""
    return _render_dataset_card_template(
        {
            "card_yaml": card_data.to_yaml(),
            "env_id": env_id,
            "intro": _dataset_card_intro(env_id, collectors),
            "replay_url": _dataset_replay_url(repo_id),
            "replay_filename": DATASET_REPLAY_FILENAME,
            "frames": f"{frames:,}",
            "episodes": f"{episodes:,}",
            "provider_label": PROVIDER_LABELS.get(provider_id, provider_id),
            "storage_format": storage_format,
            "summary_optional_rows": summary_optional_rows,
            "collector_contracts_block": collector_contracts_block,
            "environment_block": environment_block,
            "field_lines": "\n".join(field_lines),
            "repo_id": repo_id,
            "curator": curator,
        }
    ).rstrip("\n")


def _collector_contract_summaries(local_root, *, repo_id=None, remote_files=()):
    documents = {}
    if local_root:
        collector_root = os.path.join(local_root, COLLECTOR_ARTIFACT_DIR)
        if os.path.isdir(collector_root):
            for contract_id in sorted(os.listdir(collector_root)):
                path = os.path.join(collector_root, contract_id, "collection.json")
                if os.path.isfile(path):
                    documents[contract_id] = _load_json_document(
                        path, label=f"collector {contract_id}"
                    )
    if repo_id:
        for remote_path in sorted(remote_files):
            parts = remote_path.split("/")
            if len(parts) != 3 or parts[0] != COLLECTOR_ARTIFACT_DIR:
                continue
            contract_id, filename = parts[1:]
            if filename != "collection.json" or contract_id in documents:
                continue
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=remote_path,
                )
                documents[contract_id] = _load_json_document(path, label=f"collector {contract_id}")
            except Exception:
                continue
    summaries = []
    for contract_id, document in sorted(documents.items()):
        source = document.get("source") or {}
        summaries.append(
            {
                "contract_id": contract_id,
                "repo_id": source.get("repo_id"),
                "revision": source.get("revision"),
                "policy_mode": _metadata_value(document, ("policy", "mode")),
                "has_release_manifest": "release_manifest" in source,
            }
        )
    return summaries


def _build_dataset_card_content(
    recording_identity,
    env_id,
    repo_id,
    new_frames,
    new_episodes,
    dataset,
    existing_frames=0,
    existing_episodes=0,
    local_root=None,
    remote_files=None,
    fps=None,
):
    """Build a dataset card using row-derived existing and new statistics."""
    total_frames = existing_frames + new_frames
    total_episodes = existing_episodes + new_episodes

    local_metadata = load_local_metadata(recording_identity)
    metadata = _playback_metadata(dataset, local_metadata)
    if fps is not None:
        metadata["fps"] = fps
    columns = set(getattr(dataset, "column_names", None) or [])
    collectors = (
        sorted({value for value in dataset["collector"] if value})
        if "collector" in columns
        else []
    )
    gymrec_versions = (
        sorted({value for value in dataset["gymrec_version"] if value})
        if "gymrec_version" in columns
        else []
    )
    collector_contracts = _collector_contract_summaries(
        local_root,
        repo_id=repo_id,
        remote_files=remote_files or [],
    )
    return render_dataset_card_content(
        env_id,
        repo_id,
        frames=total_frames,
        episodes=total_episodes,
        metadata=metadata,
        collectors=collectors,
        gymrec_versions=gymrec_versions,
        collector_contracts=collector_contracts,
    )


def create_environment_session(contract, *, render_mode):
    """Create one internally adapted environment and its immutable recording artifact."""
    contract = (
        contract
        if isinstance(contract, EnvironmentContract)
        else EnvironmentContract.parse(contract)
    )
    session = create_session(contract, render_mode=render_mode)
    artifact = _environment_artifact(contract, session)
    return session, artifact


def _provider_catalog(*, progress=None, task_id=None):
    providers = discover_providers()
    rows = []
    for provider_id, provider in sorted(providers.items()):
        if progress is not None:
            label = PROVIDER_LABELS.get(provider_id, provider_id)
            progress.update(
                task_id,
                description=f"[bold]Scanning {label} environments[/]",
                refresh=True,
            )
        for environment_id in provider.catalog():
            rows.append((provider_id, str(environment_id)))
        if progress is not None:
            progress.advance(task_id)
    return rows


def _load_provider_catalog_with_progress():
    """Initialize Gymrec and visibly discover the installed environment catalog."""
    with _episode_progress(transient=True) as progress:
        task_id = progress.add_task(
            "[bold]Initializing Gymrec[/]",
            total=1 + len(SUPPORTED_PROVIDER_IDS),
        )
        # Paint the initial state before imports or provider discovery can block.
        progress.refresh()
        _lazy_init()
        progress.update(task_id, advance=1, refresh=True)
        rows = _provider_catalog(progress=progress, task_id=task_id)
        progress.update(
            task_id,
            description=f"[bold]Found {len(rows)} environments[/]",
            refresh=True,
        )
    return rows


def _load_recording_refs_with_progress():
    """Discover local and Hub recordings with visible phase feedback."""
    with _episode_progress(transient=True) as progress:
        task_id = progress.add_task("[bold]Searching local recordings[/]", total=2)
        progress.refresh()
        local_refs = _get_available_recording_refs_from_local()
        progress.update(
            task_id,
            advance=1,
            description="[bold]Searching Hugging Face Hub recordings[/]",
            refresh=True,
        )
        hub_refs = _get_available_recording_refs_from_hf()
        progress.update(task_id, advance=1, refresh=True)
    return local_refs, hub_refs


def _manual_environment_contract(provider_id, environment_id, config=None):
    if provider_id is None:
        matches = {
            candidate
            for candidate, candidate_env_id in _provider_catalog()
            if candidate_env_id == environment_id
        }
        if len(matches) != 1:
            supported = ", ".join(sorted(SUPPORTED_PROVIDER_IDS))
            raise ValueError(
                f"--provider is required for {environment_id!r}; choose one of: {supported}"
            )
        provider_id = matches.pop()
    return EnvironmentContract.parse(
        {
            "contract_version": 1,
            "provider_id": provider_id,
            "environment_id": environment_id,
            "config": config or {},
        }
    )


@dataclass(frozen=True)
class SelectionChoice:
    """One presentation-neutral item and its domain value."""

    key: str
    category: str
    label: str
    search_text: str
    exact_value: str
    value: object


def _environment_selection_choices(rows):
    choices = []
    for index, (provider_id, environment_id) in enumerate(rows):
        category = PROVIDER_LABELS.get(provider_id, provider_id)
        choices.append(
            SelectionChoice(
                key=f"environment:{index}",
                category=category,
                label=environment_id,
                search_text=f"{category} {provider_id} {environment_id}",
                exact_value=f"{provider_id}:{environment_id}",
                value=(provider_id, environment_id),
            )
        )
    return choices


def _recording_selection_choices(local_refs, hub_refs):
    discovered = {}
    for origin, refs in (("Local", local_refs), ("Hub", hub_refs)):
        for ref in refs:
            identity = _coerce_recording_identity(ref)
            entry = discovered.setdefault(
                identity.display_ref,
                {"value": identity.display_ref, "origins": set()},
            )
            entry["origins"].add(origin)

    choices = []
    for index, display_ref in enumerate(sorted(discovered)):
        entry = discovered[display_ref]
        origins = entry["origins"]
        category = "Local + Hub" if len(origins) > 1 else next(iter(origins))
        choices.append(
            SelectionChoice(
                key=f"recording:{index}",
                category=category,
                label=display_ref,
                search_text=f"{category} {display_ref}",
                exact_value=display_ref,
                value=entry["value"],
            )
        )
    return choices


def _terminal_tui_supported():
    if os.environ.get("GYMREC_TEXT_MENU", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("TERM") in {None, "", "dumb"}:
        return False
    size = shutil.get_terminal_size(fallback=(0, 0))
    return (
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and size.columns >= 60
        and size.lines >= 16
    )


def _choice_label(choice):
    return f"{choice.category}: {choice.label}"


def _select_choice_text_fallback(choices, *, title, argument_name):
    if not sys.stdin.isatty():
        raise ValueError(
            f"Interactive {title.lower()} selection requires a terminal; "
            f"pass {argument_name} explicitly"
        )

    console.print(
        f"[{STYLE_INFO}]{title}[/]: {len(choices)} available\n"
        "[dim]Type part of a name to search, a provider-qualified ID, or a shown number. "
        "Press Enter to cancel.[/]"
    )
    matches = list(choices[:25])
    while True:
        for index, choice in enumerate(matches, start=1):
            console.print(f"  [{STYLE_PATH}]{index:>2}[/]  {_choice_label(choice)}")
        try:
            query = Prompt.ask("Search or select", default="").strip()
        except EOFError as exc:
            raise ValueError(
                f"Interactive selection ended before a choice was made; "
                f"pass {argument_name} explicitly"
            ) from exc
        if not query:
            return None
        if query.isdigit():
            selected = int(query)
            if 1 <= selected <= len(matches):
                return matches[selected - 1].value

        query_lower = query.lower()
        qualified = [
            choice for choice in choices if choice.exact_value.lower() == query_lower
        ]
        if len(qualified) == 1:
            return qualified[0].value
        bare = [choice for choice in choices if choice.label.lower() == query_lower]
        if len(bare) == 1:
            return bare[0].value

        matches = [
            choice
            for choice in choices
            if query_lower in choice.search_text.lower()
        ][:25]
        if not matches:
            console.print(f"[{STYLE_FAIL}]No matches. Try a shorter search.[/]")


async def _select_choice(choices, *, title, placeholder, argument_name):
    if not choices:
        raise ValueError(f"No choices are available for {title.lower()}")
    if not _terminal_tui_supported():
        return _select_choice_text_fallback(
            choices,
            title=title,
            argument_name=argument_name,
        )

    from gymrec_tui import select_item

    selected_key = await select_item(
        choices,
        title=title,
        placeholder=placeholder,
    )
    if selected_key is None:
        return None
    values = {choice.key: choice.value for choice in choices}
    try:
        return values[selected_key]
    except KeyError as exc:
        raise RuntimeError("The interactive selector returned an unknown choice") from exc


async def select_environment_interactive(available_recordings_only=False):
    if available_recordings_only:
        local_refs, hub_refs = _load_recording_refs_with_progress()
        return await _select_choice(
            _recording_selection_choices(local_refs, hub_refs),
            title="Select a recording",
            placeholder="Search local and Hub recordings",
            argument_name="the recording ID",
        )

    rows = _load_provider_catalog_with_progress()
    return await _select_choice(
        _environment_selection_choices(rows),
        title="Select an environment",
        placeholder="Search environments or providers",
        argument_name="the environment ID and --provider",
    )


def list_environments():
    rows = _load_provider_catalog_with_progress()
    table = Table(title="Supported Environment Providers")
    table.add_column("Provider", style=STYLE_ENV)
    table.add_column("Environment", style=STYLE_PATH)
    for provider_id in sorted(SUPPORTED_PROVIDER_IDS):
        environments = tuple(
            environment_id
            for candidate_provider_id, environment_id in rows
            if candidate_provider_id == provider_id
        )
        if not environments:
            table.add_row(provider_id, "[dim]no environments available[/]")
        for environment_id in environments:
            table.add_row(provider_id, str(environment_id))
    console.print(table)


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


def _add_provider_arg(parser, *, default=None):
    parser.add_argument(
        "--provider",
        choices=sorted(SUPPORTED_PROVIDER_IDS),
        default=default,
        help="Environment provider for manual recording; policy bundles select their provider",
    )


@dataclass(frozen=True)
class RecordPlan:
    agent: str
    headless: bool
    max_episodes: int | None
    upload_live: bool
    seed: int

    @property
    def human(self) -> bool:
        return self.agent == "human"


def _make_record_plan(args):
    """Normalize record-mode flags into one validated execution plan."""
    agent = getattr(args, "agent", "human")
    headless = bool(getattr(args, "headless", False))
    episodes = getattr(args, "episodes", None)
    upload_live = bool(getattr(args, "upload_live", False))
    dry_run = bool(getattr(args, "dry_run", False))
    seed = getattr(args, "seed", None)

    if episodes is not None and episodes < 1:
        return None, "--episodes must be >= 1"
    if upload_live and dry_run:
        return None, "--upload-live cannot be combined with --dry-run"
    if seed is not None and seed < 0:
        return None, "--seed must be >= 0"

    effective_episodes = episodes if episodes is not None else 1
    episode_seed_increments = effective_episodes - 1
    max_base_seed = MAX_COMPATIBLE_SEED - episode_seed_increments
    if max_base_seed < 0:
        return None, f"--episodes must be <= {MAX_COMPATIBLE_SEED + 1}"
    if seed is not None and seed > max_base_seed:
        return None, (
            f"--seed must be <= {max_base_seed} for {effective_episodes} episode(s) "
            f"so every episode seed fits the supported 32-bit range"
        )
    if seed is None:
        seed = DEFAULT_RECORD_SEED

    if agent == "human":
        if headless:
            return None, "--headless can only be used with --agent (not human mode)"
        return (
            RecordPlan(
                agent=agent,
                headless=False,
                max_episodes=episodes,
                upload_live=upload_live,
                seed=seed,
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
            upload_live=upload_live,
            seed=seed,
        ),
        None,
    )


def _finish_recording_publication(recording_identity, *, dry_run):
    """Offer upload first, then print the playback command as the final recording hint."""
    if not dry_run:
        try:
            do_upload = Confirm.ask("Upload to Hugging Face Hub?", default=True, console=console)
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

    console.print(
        f"To play back: [{STYLE_CMD}]{_gymrec_cmd('playback', recording_identity.display_ref)}[/]"
    )


def _parse_cli_args(parser, argv=None):
    """Parse CLI args, routing a missing subcommand through the record parser."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args([*argv, "record"])
    return args


def _parse_provider_config(value):
    try:
        config = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("provider config must be a JSON object")
    return config


async def main():
    parser = argparse.ArgumentParser(description="Record and replay native-runtime environments")
    subparsers = parser.add_subparsers(dest="command")

    parser_record = subparsers.add_parser("record", help="Record gameplay")
    _add_env_id_arg(parser_record)
    _add_provider_arg(parser_record)
    parser_record.add_argument(
        "--env-config",
        type=_parse_provider_config,
        default=None,
        help="Opaque provider configuration as a JSON object (manual recording only)",
    )
    _add_fps_arg(parser_record, "Display/capture frames per second (default: provider value)")
    _add_scale_arg(parser_record)
    parser_record.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Record without uploading to Hugging Face",
    )
    parser_record.add_argument(
        "--upload-live",
        action="store_true",
        default=False,
        help="Upload each completed episode during recording; incompatible with --dry-run",
    )
    parser_record.add_argument(
        "--agent",
        default="human",
        choices=["human", "random"],
        help="Input source: human or seeded random policy (default: human)",
    )
    parser_record.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without display (agent mode only)",
    )
    parser_record.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episodes to collect (default: unlimited for human, 1 for an agent)",
    )
    parser_record.add_argument(
        "--storage",
        choices=STORAGE_FORMATS,
        default=None,
        help="Observation storage: images or lossless-video (default: config.toml)",
    )
    parser_record.add_argument(
        "--dataset-repo",
        default=None,
        help="Dataset target for policy recording (default: source policy repository name)",
    )
    parser_record.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Base environment and policy seed (default: {DEFAULT_RECORD_SEED})",
    )
    parser_record.add_argument("--hf-revision", default=None)
    parser_record.add_argument("--device", default="auto")
    policy_mode = parser_record.add_mutually_exclusive_group()
    policy_mode.add_argument("--deterministic", action="store_true", default=None)
    policy_mode.add_argument("--stochastic", action="store_false", dest="deterministic")

    parser_playback = subparsers.add_parser("playback", help="Replay a dataset")
    _add_env_id_arg(parser_playback)
    _add_fps_arg(parser_playback, "Playback frames per second (default: recorded provider value)")
    _add_scale_arg(parser_playback)
    parser_playback.add_argument("--verify", action="store_true", default=False)

    parser_video = subparsers.add_parser("video", help="Render dataset episodes to video")
    _add_env_id_arg(parser_video)
    _add_fps_arg(parser_video, "Export frames per second (default: recorded provider value)")
    parser_video.add_argument("--output", default=None)
    parser_video.add_argument("--range", dest="episode_range", default=None)
    parser_video.add_argument("--first", type=int, default=None)
    parser_video.add_argument("--last", type=int, default=None)

    parser_upload = subparsers.add_parser("upload", help="Upload a local dataset")
    _add_env_id_arg(parser_upload)
    parser_upload.add_argument("--replace", action="store_true")

    subparsers.add_parser("login", help="Log in to Hugging Face Hub")
    subparsers.add_parser("list_environments", help="List supported native environments")

    parser_minari = subparsers.add_parser("minari-export", help="Export to Minari")
    _add_env_id_arg(parser_minari)
    parser_minari.add_argument("--name", default=None)
    parser_minari.add_argument("--author", default=None)

    args = _parse_cli_args(parser)
    if args.command == "login":
        _lazy_init()
        ensure_hf_login(force=True)
        return
    if args.command == "list_environments":
        list_environments()
        return

    selected_provider = None
    selected = getattr(args, "env_id", None)
    if selected is None:
        available_recordings_only = args.command in {"playback", "video"}
        if available_recordings_only:
            _lazy_init()
        selected = await select_environment_interactive(available_recordings_only)
        if selected is None:
            return
        if isinstance(selected, tuple):
            selected_provider, selected = selected

    _lazy_init()
    hf_policy_source = None
    environment_contract = None
    recording_identity = None

    if args.command == "record" and is_huggingface_model_ref(selected):
        if args.agent != "human":
            raise ValueError("Hugging Face policy refs cannot be combined with --agent")
        if args.provider is not None or args.env_config is not None:
            raise ValueError("Policy bundles own their provider and environment config")
        hf_policy_source = resolve_huggingface_policy_source(
            selected,
            revision=args.hf_revision,
            device=args.device,
            deterministic=args.deterministic,
        )
        environment_contract = hf_policy_source.environment_contract
        env_id = environment_contract.environment_id
        dataset_repo_id = normalize_dataset_repo_id(
            args.dataset_repo or hf_policy_source.repo_id
        )
        recording_identity = _policy_recording_identity(
            hf_policy_source, dataset_repo=dataset_repo_id
        )
        args.agent = "hf"
        console.print(
            f"[{STYLE_INFO}]Resolved policy {hf_policy_source.collector} -> "
            f"{environment_contract.provider_id}:{env_id}, "
            f"mode={'deterministic' if hf_policy_source.deterministic else 'stochastic'}, "
            f"dataset={recording_identity.display_ref}[/]"
        )
    else:
        env_id = selected
        recording_identity = _coerce_recording_identity(env_id)
        if args.command == "record":
            if args.dataset_repo:
                raise ValueError("--dataset-repo is only valid for Hugging Face policy refs")
            environment_contract = _manual_environment_contract(
                args.provider or selected_provider,
                env_id,
                args.env_config,
            )

    if args.command == "upload":
        upload_local_dataset(recording_identity, replace=args.replace)
        return
    if args.command == "minari-export":
        minari_export(recording_identity, dataset_name=args.name, author=args.author)
        return

    if getattr(args, "scale", None) is not None:
        CONFIG["display"]["scale_factor"] = args.scale

    loaded_dataset = None
    playback_source = None
    environment_documents = {}
    if args.command in {"playback", "video"}:
        loaded_dataset, playback_source = load_recorded_dataset(recording_identity)
        if loaded_dataset is None:
            _print_missing_dataset(recording_identity)
            return
        environment_documents = _load_dataset_environment_documents(
            loaded_dataset, label=f"Dataset {recording_identity.display_ref}"
        )
        env_id = _recording_env_id(recording_identity, dataset=loaded_dataset)
        recording_identity = recording_identity.with_env_id(env_id)

    if args.command == "video":
        fps = args.fps
        if fps is None:
            values = {int(round(float(document["fps"]))) for document in environment_documents.values()}
            if len(values) != 1:
                raise ValueError("Datasets with different provider FPS values require --fps")
            fps = values.pop()
        export_dataset_video(
            env_id,
            loaded_dataset,
            output_path=args.output,
            fps=fps,
            episode_range=args.episode_range,
            first=args.first,
            last=args.last,
        )
        return

    if args.command == "playback":
        episodes, playback_total = _dataset_playback_episodes(
            loaded_dataset, verify=args.verify
        )
        groups = []
        for episode in episodes:
            contract_id = episode["environment_contract_id"]
            if not groups or groups[-1][0] != contract_id:
                groups.append((contract_id, []))
            groups[-1][1].append(episode)
        for contract_id, grouped_episodes in groups:
            document = environment_documents[contract_id]
            _contract, provider_session = _session_from_environment_document(
                document, render_mode="rgb_array"
            )
            artifact = EnvironmentArtifact(contract_id=contract_id, document=document)
            fps = args.fps or max(int(round(float(document["fps"]))), 1)
            recorder = DatasetRecorderWrapper(
                storage_format=STORAGE_FORMAT_IMAGES,
                provider_session=provider_session,
                environment_artifact=artifact,
            )
            group_total = sum(episode["step_count"] for episode in grouped_episodes)
            await recorder.replay(
                fps=fps,
                total=group_total,
                verify=args.verify,
                episodes=grouped_episodes,
            )
            recorder.close()
        return

    storage_format = _configured_storage_format(args.storage)
    record_plan, plan_error = _make_record_plan(args)
    if plan_error:
        raise ValueError(plan_error)
    if args.seed is None:
        console.print(f"[{STYLE_INFO}]Base seed: {record_plan.seed} (default)[/]")
    if record_plan.upload_live and not preflight_live_upload(recording_identity, storage_format):
        return

    provider_session, environment_artifact = create_environment_session(
        environment_contract,
        render_mode="human" if record_plan.human else "rgb_array",
    )
    env = provider_session.env
    fps = args.fps or _provider_fps(provider_session)
    policy = None
    collector_contract = None
    if not record_plan.human:
        if hf_policy_source is not None:
            policy = StableBaselines3Policy(hf_policy_source, provider_session)
            collector_contract = build_collector_contract(
                hf_policy_source,
                provider_session,
                environment_contract_id=environment_artifact.contract_id,
                inference_device=getattr(policy.model, "device", hf_policy_source.device),
            )
        else:
            policy = RandomPolicy(env.action_space)

    live_upload_manager = (
        LiveEpisodeUploadManager(
            recording_identity,
            storage_format,
            collector_contract=collector_contract,
            environment_artifact=environment_artifact,
            fps=fps,
        )
        if record_plan.upload_live
        else None
    )
    recorder = DatasetRecorderWrapper(
        input_source=None if record_plan.human else AgentInputSource(policy),
        headless=record_plan.headless,
        collector=(
            "human"
            if record_plan.human
            else hf_policy_source.collector
            if hf_policy_source
            else record_plan.agent
        ),
        storage_format=storage_format,
        live_upload_manager=live_upload_manager,
        initial_seed=record_plan.seed,
        collector_contract=collector_contract,
        provider_session=provider_session,
        environment_artifact=environment_artifact,
    )

    progress_callback = None
    progress_context = None
    if not record_plan.human:
        total_steps = [0]
        progress_context = _episode_progress(transient=False)
        progress = progress_context.__enter__()
        task_id = progress.add_task("[bold]Episodes[/]", total=record_plan.max_episodes)

        def progress_callback(_episode_number, steps_in_episode):
            total_steps[0] += steps_in_episode
            progress.update(
                task_id,
                advance=1,
                description=f"[bold]Episodes[/] [dim]({total_steps[0]} steps total)[/]",
            )

    try:
        recorded_dataset = await recorder.record(
            fps=fps,
            max_episodes=record_plan.max_episodes,
            progress_callback=progress_callback,
        )
    finally:
        if progress_context is not None:
            progress_context.__exit__(None, None, None)

    if recorded_dataset is None or record_plan.upload_live:
        recorder.close()
        return
    save_dataset_locally(
        recorded_dataset,
        recording_identity,
        artifact_root=recorder.temp_dir,
        metadata=recorder._env_metadata,
    )
    recorder.close()
    _finish_recording_publication(recording_identity, dry_run=args.dry_run)


def cli():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
    except (ValueError, RuntimeError) as exc:
        console.print(f"[{STYLE_FAIL}]Error: {exc}[/]")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    cli()
