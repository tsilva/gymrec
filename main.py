import argparse
import asyncio
import json
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from dotenv import load_dotenv
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

load_dotenv(override=True)  # Load environment variables from .env file

_initialized = False


def _get_gymrec_version():
    """Return version string like '0.1.0+abc1234' (or just '0.1.0' if git unavailable)."""
    import subprocess

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
    """Load keymappings from keymappings.toml, falling back to defaults."""
    key_map = _build_key_name_map(pygame)

    # Build defaults (same values as previously hardcoded)
    default_start_key = pygame.K_SPACE

    default_atari = {
        pygame.K_SPACE: "FIRE",
        pygame.K_UP: "UP",
        pygame.K_RIGHT: "RIGHT",
        pygame.K_LEFT: "LEFT",
        pygame.K_DOWN: "DOWN",
    }

    default_vizdoom = {
        pygame.K_UP: "MOVE_FORWARD",
        pygame.K_DOWN: "MOVE_BACKWARD",
        pygame.K_LEFT: "TURN_LEFT",
        pygame.K_RIGHT: "TURN_RIGHT",
        pygame.K_LSHIFT: "SPEED",
        pygame.K_RSHIFT: "SPEED",
        pygame.K_LCTRL: "ATTACK",
        pygame.K_RCTRL: "ATTACK",
        pygame.K_SPACE: "USE",
    }
    for i in range(1, 8):
        default_vizdoom[getattr(pygame, f"K_{i}")] = f"SELECT_WEAPON{i}"

    default_retro = {
        "Nes": {
            pygame.K_z: 0,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "Atari2600": {
            pygame.K_z: 0,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
        },
        "Snes": {
            pygame.K_z: 0,
            pygame.K_a: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
            pygame.K_s: 9,
            pygame.K_q: 10,
            pygame.K_w: 11,
        },
        "GbAdvance": {
            pygame.K_z: 0,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
            pygame.K_a: 10,
            pygame.K_s: 11,
        },
        "GameBoy": {
            pygame.K_z: 0,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "GbColor": {
            pygame.K_z: 0,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "PCEngine": {
            pygame.K_x: 0,
            pygame.K_c: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_z: 8,
            pygame.K_a: 9,
            pygame.K_s: 10,
            pygame.K_d: 11,
        },
        "Saturn": {
            pygame.K_x: 0,
            pygame.K_z: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_c: 8,
            pygame.K_a: 9,
            pygame.K_s: 10,
            pygame.K_d: 11,
        },
        "32x": {
            pygame.K_x: 0,
            pygame.K_z: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_c: 8,
            pygame.K_a: 9,
            pygame.K_s: 10,
            pygame.K_d: 11,
        },
        "Genesis": {
            pygame.K_x: 0,
            pygame.K_z: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_c: 8,
            pygame.K_a: 9,
            pygame.K_s: 10,
            pygame.K_d: 11,
        },
        "Sms": {
            pygame.K_z: 0,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "GameGear": {
            pygame.K_z: 0,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_x: 8,
        },
        "SCD": {
            pygame.K_x: 0,
            pygame.K_z: 1,
            pygame.K_n: 2,
            pygame.K_m: 3,
            pygame.K_UP: 4,
            pygame.K_DOWN: 5,
            pygame.K_LEFT: 6,
            pygame.K_RIGHT: 7,
            pygame.K_c: 8,
            pygame.K_a: 9,
            pygame.K_s: 10,
            pygame.K_d: 11,
        },
    }

    # Try loading config file
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keymappings.toml")
    if not os.path.exists(config_path):
        return default_start_key, default_atari, default_vizdoom, default_retro

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # General section
    start_key = default_start_key
    if "general" in config and "start_key" in config["general"]:
        start_key = _resolve_key(config["general"]["start_key"], key_map)

    # Atari section
    atari = default_atari
    if "atari" in config:
        atari = {}
        for key_name, action in config["atari"].items():
            atari[_resolve_key(key_name, key_map)] = action

    # VizDoom section
    vizdoom = default_vizdoom
    if "vizdoom" in config:
        vizdoom = {}
        for key_name, action in config["vizdoom"].items():
            vizdoom[_resolve_key(key_name, key_map)] = action

    # Stable-Retro section
    retro = default_retro
    if "stable_retro" in config:
        retro = {}
        for console, bindings in config["stable_retro"].items():
            retro[console] = {}
            for key_name, action in bindings.items():
                retro[console][_resolve_key(key_name, key_map)] = action

    return start_key, atari, vizdoom, retro


DEFAULT_CONFIG = {
    "display": {"scale_factor": 2},
    "fps_defaults": {"atari": 90, "vizdoom": 45, "retro": 90},
    "dataset": {
        "repo_prefix": "gymrec__",
        "license": "mit",
        "task_categories": ["reinforcement-learning"],
        "commit_message": "Update dataset card",
    },
    "storage": {
        "local_dir": os.path.join(os.path.expanduser("~"), ".gymrec", "datasets"),
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


def _load_config():
    """Load configuration from config.toml, falling back to defaults."""
    import copy

    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")
    if not os.path.exists(config_path):
        return config
    with open(config_path, "rb") as f:
        user_config = tomllib.load(f)
    for section in config:
        if section in user_config:
            config[section].update(user_config[section])
    return config


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
        DatasetCard, \
        DatasetCardData, \
        HfApi, \
        login, \
        get_token, \
        CommitOperationAdd, \
        hf_hub_download
    global Dataset, HFImage, Value, load_dataset, load_from_disk, load_dataset_builder
    global START_KEY, ATARI_KEY_BINDINGS, VIZDOOM_KEY_BINDINGS, STABLE_RETRO_KEY_BINDINGS

    import numpy as np
    import pygame
    from datasets import (
        Dataset,
        Value,
        load_dataset,
        load_dataset_builder,
        load_from_disk,
    )
    from datasets import (
        Image as HFImage,
    )
    from huggingface_hub import (
        CommitOperationAdd,
        DatasetCard,
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
        f"[{STYLE_FAIL}]Could not authenticate. Try:[/] [{STYLE_CMD}]uv run python main.py login[/]"
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

    @abstractmethod
    def handle_events(self) -> bool:
        """Process any pending events. Returns False to quit."""
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

    def _get_stable_retro_action(self):
        """Return the MultiBinary action vector for stable-retro environments."""
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
            if hasattr(self.env, "_stable_retro") and self.env._stable_retro:
                return self._get_stable_retro_action()
            return self._get_atari_action()

    def handle_events(self) -> bool:
        """Handle pygame input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                with self.key_lock:
                    self.current_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                with self.key_lock:
                    self.current_keys.discard(event.key)
        return True


class AgentInputSource(InputSource):
    """Agent input via policy function."""

    def __init__(self, policy, headless=False):
        self.policy = policy
        self.headless = headless

    def reset(self):
        """Reset any episode-local policy state."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def get_action(self, observation):
        """Get action from policy."""
        return self.policy(observation)

    def observe_step(self, reward, terminated, truncated, info):
        """Forward step results to the policy when it needs feedback."""
        if hasattr(self.policy, "observe_step"):
            self.policy.observe_step(reward, terminated, truncated, info)

    def handle_events(self) -> bool:
        """Minimal event handling - just check for ESC if not headless."""
        if not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
        return True


# =============================================================================
# Policies
# =============================================================================


class BasePolicy(ABC):
    """Abstract base class for agent policies."""

    def __init__(self, action_space, env=None):
        self.action_space = action_space
        self.env = env

    def reset(self):
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


class DatasetRecorderWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for recording and replaying Atari gameplay as Hugging Face datasets.
    """

    def __init__(self, env, input_source=None, headless=False, collector="human"):
        _lazy_init()
        super().__init__(env)

        self.recording = False
        self.frame_shape = None  # Delay initialization
        self.screen = None  # Delay initialization
        self.headless = headless
        self.input_source = input_source
        self.collector = collector
        self._gymrec_version = _get_gymrec_version()

        if not headless:
            pygame.init()
            # pygame.display.set_caption will be set after env_id is available

        self.current_keys = set()
        self.key_lock = threading.Lock()
        self.key_to_action = None  # Resolved lazily in _resolve_atari_key_mapping
        self._atari_key_bindings_raw = ATARI_KEY_BINDINGS
        self._vizdoom_buttons = None
        self._vizdoom_vector_map = None
        self.noop_action = 0

        self.episode_ids = []
        self.seeds = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.terminations = []
        self.truncations = []
        self.infos = []
        self.session_ids = []
        self._current_episode_uuid = None
        self._current_episode_seed = None
        self._session_uuid = None

        self.temp_dir = tempfile.mkdtemp()

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

    def _ensure_screen(self, frame):
        """
        Ensure pygame screen is initialized with the correct shape.
        """
        # If frame is a dict (e.g., VizDoom), extract the image
        if isinstance(frame, dict):
            # Try common keys for image observation
            for k in ["obs", "image", "screen"]:
                if k in frame:
                    frame = frame[k]
                    break
        if self.screen is None or self.frame_shape is None:
            self.frame_shape = frame.shape
            scale = CONFIG["display"]["scale_factor"]
            self.screen = pygame.display.set_mode(
                (self.frame_shape[1] * scale, self.frame_shape[0] * scale)
            )
            pygame.display.set_caption(getattr(self.env, "_env_id", "Gymnasium Recorder"))

    def _save_frame_image(self, frame):
        """Save a frame as lossless WebP and return the file path."""
        if isinstance(frame, dict):
            for k in ["obs", "image", "screen"]:
                if k in frame:
                    frame = frame[k]
                    break
        frame_uint8 = frame.astype(np.uint8)
        path = os.path.join(self.temp_dir, f"frame_{len(self.frames):05d}.webp")
        img = PILImage.fromarray(frame_uint8)
        img.save(path, format="WEBP", lossless=True, method=6)
        return path

    @staticmethod
    def _normalize_action(action):
        """Normalize action format for dataset storage."""
        if isinstance(action, np.ndarray):
            return action.tolist()
        elif isinstance(action, dict):
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in action.items()}
        else:
            return [int(action)]

    def _record_frame(self, episode_uuid, seed, step, frame, action):
        """Save a frame and action to temporary storage."""
        if not self.recording:
            return

        path = self._save_frame_image(frame)
        self.episode_ids.append(episode_uuid.bytes)
        self.seeds.append(self._current_episode_seed)
        self.frames.append(path)
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

        path = self._save_frame_image(frame)
        self.episode_ids.append(episode_uuid.bytes)
        self.seeds.append(self._current_episode_seed)
        self.frames.append(path)
        self.actions.append([])
        self.rewards.append(None)
        self.terminations.append(None)
        self.truncations.append(None)
        self.infos.append(None)
        self.session_ids.append(self._session_uuid.bytes)

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

        # Precompute vector -> discrete action mapping for faster lookup
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

    def _resolve_atari_key_mapping(self):
        """Resolve meaning-based Atari key bindings to action indices using the env's action meanings."""
        # Standard fallback: meaning -> index for the default Atari action order
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

        # Try to get actual meanings from the environment
        meaning_to_idx = None
        try:
            meanings = self.env.unwrapped.get_action_meanings()
            if meanings:
                meaning_to_idx = {}
                for idx, m in enumerate(meanings):
                    meaning_to_idx[m.upper()] = idx
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
        # Reverse map: pygame key -> meaning string (for composite action lookup)
        self._atari_key_to_meaning = {}
        for key, value in self._atari_key_bindings_raw.items():
            if isinstance(value, str):
                self._atari_key_to_meaning[key] = value.upper()

    def _get_atari_action(self):
        """Return the Discrete action index for Atari environments."""
        if self.key_to_action is None:
            self._resolve_atari_key_mapping()

        # Collect all pressed meaning strings
        pressed_meanings = set()
        for key in self.current_keys:
            if key in self._atari_key_to_meaning:
                pressed_meanings.add(self._atari_key_to_meaning[key])

        if not pressed_meanings:
            return self.noop_action

        # Build composite name following ALE convention: [UP|DOWN][RIGHT|LEFT][FIRE]
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

        # Try composite first, fall back to single key
        if composite in self._atari_meaning_to_idx:
            return self._atari_meaning_to_idx[composite]

        # Fallback: return first matching single key
        for key in self.current_keys:
            if key in self.key_to_action:
                return self.key_to_action[key]
        return self.noop_action

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
        # When the action space contains both binary and continuous components
        # (e.g. Dict["binary", "continuous"]), build the appropriate dict
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

    def _get_stable_retro_action(self):
        """Return the MultiBinary action vector for stable-retro environments."""
        action = np.zeros(self.env.action_space.n, dtype=np.int32)
        platform = getattr(self.env.unwrapped, "system", None)
        mapping = STABLE_RETRO_KEY_BINDINGS.get(platform, {})
        for key in self.current_keys:
            idx = mapping.get(key)
            if idx is not None and idx < action.shape[0]:
                action[idx] = 1
        return action

    def _get_user_action(self):
        """Map pressed keys to actions for the current environment."""
        with self.key_lock:
            if hasattr(self.env, "_vizdoom") and self.env._vizdoom:
                return self._get_vizdoom_action()
            if hasattr(self.env, "_stable_retro") and self.env._stable_retro:
                return self._get_stable_retro_action()
            return self._get_atari_action()

    def _render_frame(self, frame):
        """
        Render a frame using pygame, scaled by the configured scale factor.
        Skip rendering in headless mode.
        """
        if self.headless:
            return

        # If frame is a dict (e.g., VizDoom), extract the image
        if isinstance(frame, dict):
            for k in ["obs", "image", "screen"]:
                if k in frame:
                    frame = frame[k]
                    break
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

        if hasattr(self.env, "_vizdoom") and self.env._vizdoom:
            env_type = "VizDoom"
            if self._vizdoom_buttons is None:
                self._vizdoom_buttons = self._init_vizdoom_key_mapping()
            for key, action_name in VIZDOOM_KEY_BINDINGS.items():
                btn_idx = self._vizdoom_buttons.get(action_name)
                idx_str = f"btn {btn_idx}" if btn_idx is not None else ""
                table.add_row(pygame.key.name(key), action_name, idx_str)
            ml_idx = self._vizdoom_buttons.get("MOVE_LEFT")
            mr_idx = self._vizdoom_buttons.get("MOVE_RIGHT")
            table.add_row("alt+left", "MOVE_LEFT", f"btn {ml_idx}" if ml_idx is not None else "")
            table.add_row("alt+right", "MOVE_RIGHT", f"btn {mr_idx}" if mr_idx is not None else "")
        elif hasattr(self.env, "_stable_retro") and self.env._stable_retro:
            platform = getattr(self.env.unwrapped, "system", None)
            env_type = f"Stable-Retro ({platform})"
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
            if self.key_to_action is None:
                self._resolve_atari_key_mapping()
            try:
                meanings = self.env.unwrapped.get_action_meanings()
            except (AttributeError, TypeError):
                meanings = None
            for key, action_idx in self.key_to_action.items():
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
            return await self._record(fps=fps)
        finally:
            self.recording = False

    async def _record(self, fps=None):
        try:
            await self._play(fps)  # bypass play() to avoid premature close()
            return self._recorded_dataset
        finally:
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

        self.episode_ids.clear()
        self.seeds.clear()
        self.frames.clear()
        self.actions.clear()
        self.rewards.clear()
        self.terminations.clear()
        self.truncations.clear()
        self.infos.clear()
        self.session_ids.clear()

        # Capture environment metadata for dataset card
        self._env_metadata = _capture_env_metadata(self.env)

        self._session_uuid = uuid.uuid4()
        self._current_episode_uuid = uuid.uuid4()
        self._current_episode_seed = int(time.time())
        seed = self._current_episode_seed
        self._episode_count = 1
        self._cumulative_reward = 0.0
        obs, _ = self.env.reset(seed=seed)

        # Setup input source
        if self.input_source is None:
            # Default to human input
            self.input_source = HumanInputSource(self.env, self.key_lock, self.current_keys)
        elif hasattr(self.input_source, "reset"):
            self.input_source.reset()

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
            if self.headless:
                if not self.input_source.handle_events():
                    break
            else:
                if not self._input_loop():
                    break

            # Get action from input source
            action = self.input_source.get_action(obs)

            self._record_frame(self._current_episode_uuid, seed, step, obs, action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if hasattr(self.input_source, "observe_step"):
                self.input_source.observe_step(reward, terminated, truncated, info)
            self._cumulative_reward += float(reward)
            if self.recording:
                self.rewards.append(float(reward))
                self.terminations.append(bool(terminated))
                self.truncations.append(bool(truncated))
                self.infos.append(json.dumps(info, default=_json_default))

            self._render_frame(obs)

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

            if terminated or truncated:
                self._record_terminal_observation(self._current_episode_uuid, obs)

                if self._progress_callback is not None:
                    self._progress_callback(self._episode_count, step)

                # Check if we've reached the max episode count
                if self._max_episodes is not None and self._episode_count >= self._max_episodes:
                    break

                self._current_episode_uuid = uuid.uuid4()
                self._current_episode_seed = int(time.time())
                seed = self._current_episode_seed
                obs, _ = self.env.reset(seed=seed)
                if hasattr(self.input_source, "reset"):
                    self.input_source.reset()
                self._episode_count += 1
                self._cumulative_reward = 0.0
                step = 0
                self._render_frame(obs)

        # Record terminal observation on user exit (ESC) or when max_episodes reached
        if self.recording and self.frames and self.actions[-1] != []:
            # Mark last real step as truncated: user exited mid-episode.
            # Minari requires at least one True in terminations or truncations per episode.
            self.truncations[-1] = True
            self._record_terminal_observation(self._current_episode_uuid, obs)

        if self.recording and self.frames:
            data = {
                "episode_id": self.episode_ids,
                "seed": self.seeds,
                "observations": self.frames,
                "actions": self.actions,
                "rewards": self.rewards,
                "terminations": self.terminations,
                "truncations": self.truncations,
                "infos": self.infos,
                "session_id": self.session_ids,
                "collector": [self.collector] * len(self.frames),
                "gymrec_version": [self._gymrec_version] * len(self.frames),
            }
            self._recorded_dataset = Dataset.from_dict(data)
            self._recorded_dataset = self._recorded_dataset.cast_column("observations", HFImage())
            # Cast episode_id and session_id to binary for efficient UUID storage
            self._recorded_dataset = self._recorded_dataset.cast_column(
                "episode_id", Value("binary")
            )
            self._recorded_dataset = self._recorded_dataset.cast_column(
                "session_id", Value("binary")
            )

    def _extract_obs_image(self, obs):
        """Extract image array from observation, handling dict obs (e.g. VizDoom)."""
        if isinstance(obs, dict):
            for k in ["obs", "image", "screen"]:
                if k in obs:
                    return obs[k]
        return obs

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

    async def replay(self, actions, fps=None, total=None, verify=False):
        if fps is None:
            fps = get_default_fps(self.env)
        self._fps = fps
        self._playback_frame_index = 0
        self._playback_total = total
        obs, _ = self.env.reset()
        self._ensure_screen(obs)
        self._render_frame(obs)
        self._print_keymappings()

        mse_threshold = 5.0
        verify_metrics = [] if verify else None
        reward_mismatches = 0
        terminal_mismatches = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task("Replaying", total=total)
            try:
                for frame_number, item in enumerate(actions, start=1):
                    frame_start = time.monotonic()
                    if not self._input_loop():
                        break

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
                        obs_image = self._extract_obs_image(obs)
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
                        if bool(terminated) != bool(recorded_terminated) or bool(truncated) != bool(
                            recorded_truncated
                        ):
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
    user_info = whoami()
    username = user_info.get("name") or user_info.get("user") or user_info.get("username")
    encoded_env_id = _encode_env_id_for_hf(env_id)
    hf_repo_id = f"{username}/{CONFIG['dataset']['repo_prefix']}{encoded_env_id}"
    return hf_repo_id


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


def get_local_dataset_path(env_id):
    """Return the local storage path for a given environment's dataset."""
    encoded_env_id = _encode_env_id_for_hf(env_id)
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
                # Skip old-format directories that don't use the encoding scheme
                # (directories created before encoding was implemented won't have _dash_ or _underscore_)
                if "_dash_" not in entry and "_underscore_" not in entry:
                    continue
                # Decode the directory name back to env_id
                env_id = _decode_hf_repo_name(entry)
                available.append(env_id)
    return sorted(set(available))


def _get_available_envs_from_hf():
    """Get list of env_ids that have HF Hub recordings."""
    try:
        user_info = whoami()
        username = user_info.get("name") or user_info.get("user") or user_info.get("username")
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


def _get_metadata_path(env_id):
    """Return the path to the metadata JSON file for a given environment."""
    encoded_env_id = _encode_env_id_for_hf(env_id)
    return os.path.join(CONFIG["storage"]["local_dir"], f"{encoded_env_id}_metadata.json")


def _get_uploaded_episodes_path(env_id):
    """Return the path to the uploaded episodes tracking file for a given environment."""
    encoded_env_id = _encode_env_id_for_hf(env_id)
    return os.path.join(CONFIG["storage"]["local_dir"], f"{encoded_env_id}_uploaded.json")


def _load_uploaded_episode_ids(env_id):
    """Load the set of already-uploaded episode IDs from local tracking file."""
    path = _get_uploaded_episodes_path(env_id)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f))


def _save_uploaded_episode_ids(env_id, episode_ids: set):
    """Save the set of uploaded episode IDs to local tracking file."""
    path = _get_uploaded_episodes_path(env_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(episode_ids), f)


def save_dataset_locally(dataset, env_id, metadata=None):
    """Save dataset to local disk, appending to any existing data."""
    path = get_local_dataset_path(env_id)
    metadata_path = _get_metadata_path(env_id)

    if os.path.exists(path):
        # Load existing dataset - UUIDs are already unique, no offsetting needed
        existing_dataset = load_from_disk(path, keep_in_memory=True)

        # Backward-compatible: add missing provenance columns with sentinel values
        _SENTINEL_STR = "unknown"
        _SENTINEL_BYTES = b"\x00" * 16
        n_existing = len(existing_dataset)
        if "session_id" not in existing_dataset.column_names:
            existing_dataset = existing_dataset.add_column(
                "session_id", [_SENTINEL_BYTES] * n_existing
            )
            existing_dataset = existing_dataset.cast_column("session_id", Value("binary"))
        if "collector" not in existing_dataset.column_names:
            existing_dataset = existing_dataset.add_column(
                "collector", [_SENTINEL_STR] * n_existing
            )
        if "gymrec_version" not in existing_dataset.column_names:
            existing_dataset = existing_dataset.add_column(
                "gymrec_version", [_SENTINEL_STR] * n_existing
            )

        # Concatenate datasets
        from datasets import concatenate_datasets

        dataset = concatenate_datasets([existing_dataset, dataset])

        # Remove old dataset directory
        shutil.rmtree(path)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.save_to_disk(path)

    # Save/update metadata
    if metadata is not None:
        existing_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                existing_metadata = json.load(f)
        # Update with new metadata (newer values take precedence)
        existing_metadata.update(metadata)
        # Add recording timestamp
        if "recordings" not in existing_metadata:
            existing_metadata["recordings"] = []
        # Extract provenance info from dataset columns if available
        _collectors = set(dataset["collector"]) if "collector" in dataset.column_names else set()
        _versions = (
            set(dataset["gymrec_version"]) if "gymrec_version" in dataset.column_names else set()
        )
        recording_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "episodes": len(set(dataset["episode_id"])),
            "frames": len(dataset),
        }
        if _collectors:
            recording_entry["collectors"] = sorted(_collectors)
        if _versions:
            recording_entry["gymrec_versions"] = sorted(_versions)
        existing_metadata["recordings"].append(recording_entry)
        with open(metadata_path, "w") as f:
            json.dump(existing_metadata, f, indent=2, default=_json_default)

    console.print(f"Dataset saved locally ([{STYLE_PATH}]{path}[/])")
    return path


def load_local_metadata(env_id):
    """Load metadata from local disk. Returns None if not found."""
    metadata_path = _get_metadata_path(env_id)
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_local_dataset(env_id):
    """Load dataset from local disk. Returns None if not found."""
    path = get_local_dataset_path(env_id)
    if not os.path.exists(path):
        return None
    return load_from_disk(path)


def _get_default_fps_for_env_id(env_id, metadata=None):
    """Infer a sensible FPS without instantiating the environment."""
    if metadata and metadata.get("fps") is not None:
        try:
            return max(int(round(float(metadata["fps"]))), 1)
        except (TypeError, ValueError):
            pass

    retro_platforms = {
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
    }

    if any(env_id.endswith(f"-{plat}") for plat in retro_platforms) or "Atari2600" in env_id:
        return CONFIG["fps_defaults"]["retro"]
    if "Vizdoom" in env_id or "vizdoom" in env_id:
        return CONFIG["fps_defaults"]["vizdoom"]
    return CONFIG["fps_defaults"]["atari"]


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
    observation = row.get("observations", row.get("observation"))
    if observation is None:
        raise ValueError("Dataset row is missing observations")
    return observation


def _frame_to_rgb_array(frame):
    """Normalize a dataset frame to an HxWx3 uint8 RGB array."""
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

    first_frame = _frame_to_rgb_array(_get_row_observation(dataset[selected_episodes[0][1][0]]))
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
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_id = progress.add_task("[bold]Encoding[/]", total=total_frames)
            for episode_number, row_indices in selected_episodes:
                for row_index in row_indices:
                    row = dataset[row_index]
                    frame = _frame_to_rgb_array(_get_row_observation(row))
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


def upload_local_dataset(env_id, max_retries=5, base_wait=1.0):
    """Upload new episodes to HF Hub using append-only shard uploads.

    Only uploads episodes that have not been uploaded before (tracked in a local
    JSON file). Uploads new data as a parquet shard alongside existing shards —
    no remote data is downloaded or replaced. Uses optimistic locking via
    parent_commit in create_commit() to handle concurrent uploads safely.

    Args:
        env_id: The environment ID to upload
        max_retries: Maximum number of retry attempts on conflict (default: 5)
        base_wait: Base wait time between retries in seconds (default: 1.0)
    """
    if not ensure_hf_login():
        return False

    local_dataset = load_local_dataset(env_id)
    if local_dataset is None:
        console.print(f"[{STYLE_FAIL}]No local dataset found for {env_id}[/]")
        console.print(f"  Expected at: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]")
        return False

    # Filter to episodes not yet uploaded
    already_uploaded = _load_uploaded_episode_ids(env_id)
    new_indices = []
    new_episode_ids = set()
    for i, row in enumerate(local_dataset):
        eid = row["episode_id"]
        if isinstance(eid, bytes):
            eid = uuid.UUID(bytes=eid).hex
        elif isinstance(eid, uuid.UUID):
            eid = eid.hex
        else:
            eid = str(eid)
        if eid not in already_uploaded:
            new_indices.append(i)
            new_episode_ids.add(eid)

    if not new_indices:
        console.print(f"[{STYLE_INFO}]All episodes already uploaded, nothing to do[/]")
        return True

    new_dataset = local_dataset.select(new_indices)
    n_new_episodes = len(new_episode_ids)
    console.print(
        f"[{STYLE_INFO}]Uploading {n_new_episodes} new episodes ({len(new_indices)} frames)...[/]"
    )

    hf_repo_id = env_id_to_hf_repo_id(env_id)
    api = HfApi()

    for attempt in range(1, max_retries + 1):
        try:
            # 1. Check repo existence and get parent commit
            repo_exists = False
            parent_commit = None
            try:
                repo_info = api.repo_info(repo_id=hf_repo_id, repo_type="dataset")
                repo_exists = True
                parent_commit = repo_info.sha
            except Exception:
                pass

            if not repo_exists:
                api.create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True)
                repo_info = api.repo_info(repo_id=hf_repo_id, repo_type="dataset")
                parent_commit = repo_info.sha

            # 2. Count existing shards to determine next shard index
            next_shard_idx = 0
            if repo_exists:
                try:
                    for item in api.list_repo_tree(
                        hf_repo_id, repo_type="dataset", path_in_repo="data"
                    ):
                        if hasattr(item, "rfilename"):
                            name = item.rfilename
                        else:
                            name = str(item)
                        if name.startswith("data/train-") and name.endswith(".parquet"):
                            next_shard_idx += 1
                except Exception:
                    pass  # No data/ dir yet, start at 0

            # 3. Write new episodes as parquet shard to temp dir
            import tempfile

            operations = []
            with tempfile.TemporaryDirectory() as tmpdir:
                shard_path = os.path.join(tmpdir, "shard.parquet")
                new_dataset.to_parquet(shard_path)
                shard_name = f"data/train-{next_shard_idx:05d}-of-{next_shard_idx + 1:05d}.parquet"
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=shard_name,
                        path_or_fileobj=shard_path,
                    )
                )

                # 4. Build updated dataset card
                card_content = _build_dataset_card_content(
                    env_id,
                    hf_repo_id,
                    api,
                    new_frames=len(new_indices),
                    new_episodes=n_new_episodes,
                    repo_exists=repo_exists,
                )
                if card_content:
                    card_path = os.path.join(tmpdir, "README.md")
                    with open(card_path, "w") as f:
                        f.write(card_content)
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo="README.md",
                            path_or_fileobj=card_path,
                        )
                    )

                # 5. Pre-upload LFS files then atomic commit
                api.preupload_lfs_files(
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                    additions=[op for op in operations if isinstance(op, CommitOperationAdd)],
                )
                api.create_commit(
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Add recordings from {env_id}",
                    parent_commit=parent_commit,
                )

            # 6. Track uploaded episodes locally
            _save_uploaded_episode_ids(env_id, already_uploaded | new_episode_ids)

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
                        f"[{STYLE_INFO}]Conflict detected, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...[/]"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    console.print(f"[{STYLE_FAIL}]Max retries ({max_retries}) exceeded.[/]")
                    console.print(
                        f"[{STYLE_INFO}]Another client may be uploading. Try again later.[/]"
                    )
                    return False
            else:
                console.print(f"[{STYLE_FAIL}]Upload failed: {e}[/]")
                return False

    return False


def minari_export(env_id, dataset_name=None, author=None):
    """Export a local HF dataset to Minari format for offline RL."""
    try:
        import minari
        from minari.data_collector import EpisodeBuffer
    except ImportError:
        console.print(f"[{STYLE_FAIL}]Minari is not installed.[/]")
        console.print(f"Install with: [{STYLE_CMD}]uv pip install 'minari>=0.5.0'[/]")
        return False

    dataset = load_local_dataset(env_id)
    if dataset is None:
        console.print(f"[{STYLE_FAIL}]No local dataset found for {env_id}[/]")
        console.print(f"  Expected at: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]")
        return False

    # Group rows by episode
    episodes = {}
    for row in dataset:
        eid = row["episode_id"]
        # Handle both UUID bytes and legacy integer episode IDs
        if isinstance(eid, bytes):
            eid = uuid.UUID(bytes=eid)
        if eid not in episodes:
            episodes[eid] = []
        episodes[eid].append(row)
    for eid in episodes:
        if "step" in episodes[eid][0]:
            episodes[eid].sort(key=lambda r: r["step"])

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
    for ep_idx, (eid, rows) in enumerate(sorted(episodes.items())):
        observations = []
        actions = []
        rewards = []
        terminations = []
        truncations = []
        ep_seed = rows[0].get("seed", 0)

        for row in rows:
            img = row.get("observations", row.get("observation"))
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            observations.append(img)

            # Detect terminal observation by empty/missing actions
            action = row.get("actions", row.get("action"))
            if (isinstance(action, list) and len(action) == 0) or action is None:
                continue

            if isinstance(action, list) and len(action) == 1:
                action = action[0]
            actions.append(action)

            reward = row.get("rewards", row.get("reward"))
            rewards.append(float(reward) if reward is not None else 0.0)
            term = row.get("terminations", row.get("termination"))
            terminations.append(bool(term) if term is not None else False)
            trunc = row.get("truncations", row.get("truncation"))
            truncations.append(bool(trunc) if trunc is not None else False)

        # Fallback for old datasets without terminal observation:
        # duplicate last obs to satisfy Minari N+1 requirement
        if len(observations) == len(actions):
            observations.append(observations[-1])

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


def _detect_backend(env_id):
    """Return backend name for metadata tags."""
    if env_id in set(_get_stableretro_envs()):
        return "stable-retro"
    elif "Vizdoom" in env_id:
        return "vizdoom"
    else:
        return "atari"


def _capture_env_metadata(env):
    """Capture environment configuration metadata for dataset card."""
    metadata = {
        "env_id": getattr(env, "_env_id", "unknown"),
        "backend": _detect_backend(getattr(env, "_env_id", "")),
        "frameskip": get_frameskip(env),
        "fps": get_default_fps(env),
    }

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
    if hasattr(obs_space, "shape"):
        metadata["observation_shape"] = list(obs_space.shape)
    if hasattr(obs_space, "dtype"):
        metadata["observation_dtype"] = str(obs_space.dtype)

    # Spec-based metadata
    spec = getattr(env, "spec", None)
    if spec is not None:
        if hasattr(spec, "max_episode_steps") and spec.max_episode_steps is not None:
            metadata["max_episode_steps"] = spec.max_episode_steps
        kwargs = getattr(spec, "kwargs", {}) or {}
        # Sticky actions (ALE)
        if "repeat_action_probability" in kwargs:
            metadata["sticky_actions"] = kwargs["repeat_action_probability"]

    # ALE-specific: check unwrapped for sticky actions
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None:
        sticky = getattr(unwrapped, "_repeat_action_probability", None)
        if sticky is not None:
            metadata["sticky_actions"] = sticky

    # Reward range
    if hasattr(env, "reward_range"):
        metadata["reward_range"] = list(env.reward_range)

    # Stable-Retro specific
    if hasattr(env, "_stable_retro") and env._stable_retro:
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


_BACKEND_LABELS = {
    "atari": "Atari (ALE-py)",
    "vizdoom": "VizDoom",
    "stable-retro": "Stable-Retro",
}


def generate_dataset_card(dataset, env_id, repo_id, metadata=None):
    """Generate or update the dataset card for a given dataset repo."""

    frames = len(dataset)
    episodes = len(set(dataset["episode_id"]))
    backend = _detect_backend(env_id)

    user_info = whoami()
    curator = user_info.get("name") or user_info.get("user") or "unknown"

    # Extract provenance info from dataset columns
    collectors = sorted(set(dataset["collector"])) if "collector" in dataset.column_names else []
    gymrec_versions = (
        sorted(set(dataset["gymrec_version"])) if "gymrec_version" in dataset.column_names else []
    )

    # Build dynamic intro based on collectors
    if collectors and collectors != ["human"]:
        collector_str = ", ".join(f"`{c}`" for c in collectors)
        intro = f"Gameplay recordings (collected by: {collector_str}) from the Gymnasium environment `{env_id}`,"
    else:
        intro = f"Human gameplay recordings from the Gymnasium environment `{env_id}`,"

    card_data = DatasetCardData(
        language="en",
        license=CONFIG["dataset"]["license"],
        task_categories=CONFIG["dataset"]["task_categories"],
        tags=["gymnasium", backend, env_id],
        size_categories=[_size_category(frames)],
        pretty_name=f"{env_id} Gameplay Dataset",
    )

    header = card_data.to_yaml()
    content_lines = [
        "---",
        header,
        "---",
        "",
        f"# {env_id} Gameplay Dataset",
        "",
        intro,
        "captured using [gymrec](https://github.com/tsilva/gymrec).",
        "",
        "## Dataset Summary",
        "",
        "| Stat | Value |",
        "|------|-------|",
        f"| Total frames | {frames:,} |",
        f"| Episodes | {episodes:,} |",
        f"| Environment | `{env_id}` |",
        f"| Backend | {_BACKEND_LABELS.get(backend, backend)} |",
    ]
    if collectors:
        content_lines.append(f"| Collector(s) | {', '.join(collectors)} |")
    if gymrec_versions:
        content_lines.append(f"| gymrec version(s) | {', '.join(gymrec_versions)} |")
    content_lines.append("")

    # Add Environment Configuration section if metadata is available
    if metadata:
        content_lines.extend(
            [
                "## Environment Configuration",
                "",
                "| Setting | Value |",
                "|---------|-------|",
            ]
        )

        # Core settings
        if "frameskip" in metadata:
            content_lines.append(f"| Frameskip | {metadata['frameskip']} |")
        if "fps" in metadata:
            content_lines.append(f"| Target FPS | {metadata['fps']} |")
        if "sticky_actions" in metadata:
            content_lines.append(f"| Sticky Actions | {metadata['sticky_actions']} |")
        if "max_episode_steps" in metadata:
            content_lines.append(f"| Max Episode Steps | {metadata['max_episode_steps']} |")

        # Observation space
        if "observation_shape" in metadata:
            shape = metadata["observation_shape"]
            content_lines.append(f"| Observation Shape | {' × '.join(str(s) for s in shape)} |")
        if "observation_dtype" in metadata:
            content_lines.append(f"| Observation Dtype | {metadata['observation_dtype']} |")

        # Action space
        if "action_space_type" in metadata:
            content_lines.append(f"| Action Space | {metadata['action_space_type']} |")
        if "n_actions" in metadata:
            content_lines.append(f"| Number of Actions | {metadata['n_actions']} |")

        # Reward range
        if "reward_range" in metadata:
            rmin, rmax = metadata["reward_range"]
            content_lines.append(f"| Reward Range | [{rmin}, {rmax}] |")

        # Backend-specific info
        if backend == "stable-retro":
            if "retro_platform" in metadata and metadata["retro_platform"]:
                content_lines.append(f"| Platform | {metadata['retro_platform']} |")
            if "retro_game" in metadata and metadata["retro_game"]:
                content_lines.append(f"| Game | {metadata['retro_game']} |")
            if "retro_buttons" in metadata and metadata["retro_buttons"]:
                named = [b for b in metadata["retro_buttons"][:8] if b is not None]
                buttons = ", ".join(named)
                total_named = sum(1 for b in metadata["retro_buttons"] if b is not None)
                if total_named > 8:
                    buttons += f" (+{total_named - 8} more)"
                content_lines.append(f"| Buttons | {buttons} |")

        elif backend == "vizdoom":
            if "vizdoom_scenario" in metadata and metadata["vizdoom_scenario"]:
                content_lines.append(f"| Scenario | {metadata['vizdoom_scenario']} |")
            if "vizdoom_num_binary_buttons" in metadata:
                content_lines.append(
                    f"| Binary Buttons | {metadata['vizdoom_num_binary_buttons']} |"
                )
            if "vizdoom_num_delta_buttons" in metadata:
                content_lines.append(f"| Delta Buttons | {metadata['vizdoom_num_delta_buttons']} |")

        content_lines.append("")

    content_lines.extend(
        [
            "## Dataset Structure",
            "",
            "Minari-compatible flat table format. Use `minari-export` for native [Minari](https://minari.farama.org/) HDF5 format.",
            "",
            "Each episode has N step rows plus one terminal observation row (N+1 pattern).",
            "The terminal observation is the final state after the last step — it has an empty action",
            "and null values for rewards/terminations/truncations/infos.",
            "",
            "- **episode_id** (`binary(16)`): Unique UUID identifier for each episode (16 bytes, universally unique across all recordings)",
            "- **seed** (`int` or `null`): RNG seed used for `env.reset()` (set on first row of each episode, `null` on other rows)",
            "- **observations** (`Image`): RGB frame from the environment",
            "- **actions** (`list`): Action taken at this step (`[]` for terminal observations)",
            "- **rewards** (`float` or `null`): Reward received (`null` on terminal observation rows)",
            "- **terminations** (`bool` or `null`): Whether the episode terminated naturally (`null` on terminal observation rows)",
            "- **truncations** (`bool` or `null`): Whether the episode was truncated (`null` on terminal observation rows)",
            "- **infos** (`str` or `null`): Additional environment info as JSON (`null` on terminal observation rows)",
            "- **session_id** (`binary(16)`): UUID grouping all episodes from one `gymrec record` run",
            '- **collector** (`string`): Who collected the data (`"human"`, `"random"`, or future agent names)',
            '- **gymrec_version** (`string`): Version of gymrec used to record (e.g. `"0.1.0+abc1234"`)',
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
    card = DatasetCard("\n".join(content_lines))

    card.push_to_hub(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=CONFIG["dataset"]["commit_message"],
    )


def _build_dataset_card_content(env_id, repo_id, api, new_frames, new_episodes, repo_exists):
    """Build dataset card content string for an append-only upload.

    If the repo exists, downloads the current README and parses existing frame/episode
    counts to compute running totals. Otherwise starts from new_frames/new_episodes.
    Returns the full card content string (does not push to Hub).
    """
    import re

    total_frames = new_frames
    total_episodes = new_episodes

    if repo_exists:
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
            if frames_match:
                total_frames += int(frames_match.group(1).replace(",", ""))
            if episodes_match:
                total_episodes += int(episodes_match.group(1).replace(",", ""))
        except Exception:
            pass

    backend = _detect_backend(env_id)
    user_info = whoami()
    curator = user_info.get("name") or user_info.get("user") or "unknown"
    metadata = load_local_metadata(env_id)

    # Extract provenance info from local metadata recordings
    all_collectors = set()
    all_versions = set()
    if metadata and "recordings" in metadata:
        for rec in metadata["recordings"]:
            for c in rec.get("collectors", []):
                all_collectors.add(c)
            for v in rec.get("gymrec_versions", []):
                all_versions.add(v)
    collectors = sorted(all_collectors)
    gymrec_versions = sorted(all_versions)

    # Build dynamic intro based on collectors
    if collectors and collectors != ["human"]:
        collector_str = ", ".join(f"`{c}`" for c in collectors)
        intro = f"Gameplay recordings (collected by: {collector_str}) from the Gymnasium environment `{env_id}`,"
    else:
        intro = f"Human gameplay recordings from the Gymnasium environment `{env_id}`,"

    card_data = DatasetCardData(
        language="en",
        license=CONFIG["dataset"]["license"],
        task_categories=CONFIG["dataset"]["task_categories"],
        tags=["gymnasium", backend, env_id],
        size_categories=[_size_category(total_frames)],
        pretty_name=f"{env_id} Gameplay Dataset",
    )

    header = card_data.to_yaml()
    content_lines = [
        "---",
        header,
        "---",
        "",
        f"# {env_id} Gameplay Dataset",
        "",
        intro,
        "captured using [gymrec](https://github.com/tsilva/gymrec).",
        "",
        "## Dataset Summary",
        "",
        "| Stat | Value |",
        "|------|-------|",
        f"| Total frames | {total_frames:,} |",
        f"| Episodes | {total_episodes:,} |",
        f"| Environment | `{env_id}` |",
        f"| Backend | {_BACKEND_LABELS.get(backend, backend)} |",
    ]
    if collectors:
        content_lines.append(f"| Collector(s) | {', '.join(collectors)} |")
    if gymrec_versions:
        content_lines.append(f"| gymrec version(s) | {', '.join(gymrec_versions)} |")
    content_lines.append("")

    if metadata:
        content_lines.extend(
            [
                "## Environment Configuration",
                "",
                "| Setting | Value |",
                "|---------|-------|",
            ]
        )
        if "frameskip" in metadata:
            content_lines.append(f"| Frameskip | {metadata['frameskip']} |")
        if "fps" in metadata:
            content_lines.append(f"| Target FPS | {metadata['fps']} |")
        if "sticky_actions" in metadata:
            content_lines.append(f"| Sticky Actions | {metadata['sticky_actions']} |")
        if "max_episode_steps" in metadata:
            content_lines.append(f"| Max Episode Steps | {metadata['max_episode_steps']} |")
        if "observation_shape" in metadata:
            shape = metadata["observation_shape"]
            content_lines.append(f"| Observation Shape | {' × '.join(str(s) for s in shape)} |")
        if "observation_dtype" in metadata:
            content_lines.append(f"| Observation Dtype | {metadata['observation_dtype']} |")
        if "action_space_type" in metadata:
            content_lines.append(f"| Action Space | {metadata['action_space_type']} |")
        if "n_actions" in metadata:
            content_lines.append(f"| Number of Actions | {metadata['n_actions']} |")
        if "reward_range" in metadata:
            rmin, rmax = metadata["reward_range"]
            content_lines.append(f"| Reward Range | [{rmin}, {rmax}] |")
        if backend == "stable-retro":
            if "retro_platform" in metadata and metadata["retro_platform"]:
                content_lines.append(f"| Platform | {metadata['retro_platform']} |")
            if "retro_game" in metadata and metadata["retro_game"]:
                content_lines.append(f"| Game | {metadata['retro_game']} |")
            if "retro_buttons" in metadata and metadata["retro_buttons"]:
                named = [b for b in metadata["retro_buttons"][:8] if b is not None]
                buttons = ", ".join(named)
                total_named = sum(1 for b in metadata["retro_buttons"] if b is not None)
                if total_named > 8:
                    buttons += f" (+{total_named - 8} more)"
                content_lines.append(f"| Buttons | {buttons} |")
        elif backend == "vizdoom":
            if "vizdoom_scenario" in metadata and metadata["vizdoom_scenario"]:
                content_lines.append(f"| Scenario | {metadata['vizdoom_scenario']} |")
            if "vizdoom_num_binary_buttons" in metadata:
                content_lines.append(
                    f"| Binary Buttons | {metadata['vizdoom_num_binary_buttons']} |"
                )
            if "vizdoom_num_delta_buttons" in metadata:
                content_lines.append(f"| Delta Buttons | {metadata['vizdoom_num_delta_buttons']} |")
        content_lines.append("")

    content_lines.extend(
        [
            "## Dataset Structure",
            "",
            "Minari-compatible flat table format. Use `minari-export` for native [Minari](https://minari.farama.org/) HDF5 format.",
            "",
            "Each episode has N step rows plus one terminal observation row (N+1 pattern).",
            "The terminal observation is the final state after the last step — it has an empty action",
            "and null values for rewards/terminations/truncations/infos.",
            "",
            "- **episode_id** (`binary(16)`): Unique UUID identifier for each episode (16 bytes, universally unique across all recordings)",
            "- **seed** (`int` or `null`): RNG seed used for `env.reset()` (set on first row of each episode, `null` on other rows)",
            "- **observations** (`Image`): RGB frame from the environment",
            "- **actions** (`list`): Action taken at this step (`[]` for terminal observations)",
            "- **rewards** (`float` or `null`): Reward received (`null` on terminal observation rows)",
            "- **terminations** (`bool` or `null`): Whether the episode terminated naturally (`null` on terminal observation rows)",
            "- **truncations** (`bool` or `null`): Whether the episode was truncated (`null` on terminal observation rows)",
            "- **infos** (`str` or `null`): Additional environment info as JSON (`null` on terminal observation rows)",
            "- **session_id** (`binary(16)`): UUID grouping all episodes from one `gymrec record` run",
            '- **collector** (`string`): Who collected the data (`"human"`, `"random"`, or future agent names)',
            '- **gymrec_version** (`string`): Version of gymrec used to record (e.g. `"0.1.0+abc1234"`)',
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


def _create_env__stableretro(env_id):
    import stable_retro as retro

    try:
        env = retro.make(env_id, render_mode="rgb_array")
    except FileNotFoundError:
        console.print(f"\n[{STYLE_FAIL}]Error: ROM not found for '{env_id}'.[/]")
        console.print("\nStable-retro requires ROM files to be imported separately.")
        console.print(
            f"Import ROMs with:  [{STYLE_CMD}]python -m stable_retro.import /path/to/your/roms/[/]"
        )
        console.print(
            f"\nUse [{STYLE_CMD}]list_environments[/] to see which games have ROMs imported."
        )
        sys.exit(1)
    env._stable_retro = True
    return env


def _create_env__vizdoom(env_id):
    import vizdoom.gymnasium_wrapper  # noqa: F401

    env = gym.make(env_id, render_mode="rgb_array", max_buttons_pressed=0)
    env._vizdoom = True
    return env


def _create_env__alepy(env_id):
    import ale_py

    gym.register_envs(ale_py)
    return gym.make(env_id, render_mode="rgb_array")


def create_env(env_id):
    """Create a Gymnasium environment with the appropriate backend."""

    if env_id in set(_get_stableretro_envs()):
        env = _create_env__stableretro(env_id)
    elif "Vizdoom" in env_id:
        env = _create_env__vizdoom(env_id)
    else:
        env = _create_env__alepy(env_id)

    env._env_id = env_id.replace("-", "_")
    return env


def get_frameskip(env) -> int:
    """Detect the frameskip value for an environment.

    Returns the number of internal frames per env.step() call.
    For stochastic frameskip tuples like (2, 5), returns the average.
    """
    env_id = getattr(env, "_env_id", "")

    # VizDoom and Retro have different frameskip semantics; skip detection
    if hasattr(env, "_vizdoom") and env._vizdoom:
        return 1
    if hasattr(env, "_stable_retro") and env._stable_retro:
        return 1

    # Check spec kwargs (works for gym.make() environments)
    spec = getattr(env, "spec", None)
    if spec is not None:
        kwargs = getattr(spec, "kwargs", {}) or {}
        fs = kwargs.get("frameskip")
        if fs is not None:
            if isinstance(fs, (tuple, list)) and len(fs) == 2:
                return max(int(round((fs[0] + fs[1]) / 2)), 1)
            try:
                return max(int(fs), 1)
            except (TypeError, ValueError):
                pass

    # ALE-specific attribute fallback
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None:
        fs = getattr(unwrapped, "_frameskip", None)
        if fs is not None:
            if isinstance(fs, (tuple, list)) and len(fs) == 2:
                return max(int(round((fs[0] + fs[1]) / 2)), 1)
            try:
                return max(int(fs), 1)
            except (TypeError, ValueError):
                pass

    # Name-based: NoFrameskip means frameskip=1
    if "NoFrameskip" in env_id:
        return 1

    return 1


def get_default_fps(env):
    """Determine a sensible default FPS for an environment."""

    base_fps = None

    for key in ("render_fps", "video.frames_per_second"):
        fps = env.metadata.get(key)
        if fps:
            try:
                base_fps = int(round(float(fps)))
                break
            except (TypeError, ValueError):
                pass

    if base_fps is None:
        env_id = getattr(env, "_env_id", "")

        retro_platforms = {
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
        }

        if any(env_id.endswith(f"-{plat}") for plat in retro_platforms) or "Atari2600" in env_id:
            base_fps = CONFIG["fps_defaults"]["retro"]
        elif "Vizdoom" in env_id or "vizdoom" in env_id:
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


def _get_atari_envs() -> list[str]:
    try:
        import ale_py

        gym.register_envs(ale_py)
        return sorted(
            env_id
            for env_id in gym.envs.registry.keys()
            if str(gym.spec(env_id).entry_point) == "ale_py.env:AtariEnv"
            and env_id.startswith("ALE/")
        )
    except Exception:
        return []


def _get_stableretro_envs(imported_only: bool = False) -> list[str]:
    try:
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
    retro_platforms = {
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

    if env_id.startswith("ALE/"):
        return "Atari"
    elif env_id.startswith("Vizdoom") or env_id.startswith("vizdoom"):
        return "VizDoom"
    elif any(env_id.endswith(f"-{plat}") for plat in retro_platforms):
        return "Stable-Retro"
    else:
        return "Atari"  # Default fallback


def select_environment_interactive(available_recordings_only: bool = False) -> str:
    from simple_term_menu import TerminalMenu

    if available_recordings_only:
        # Get envs with available recordings
        local_envs = _get_available_envs_from_local()
        hf_envs = _get_available_envs_from_hf()
        all_recorded_envs = sorted(set(local_envs + hf_envs))

        if not all_recorded_envs:
            console.print(
                f"[{STYLE_FAIL}]No recordings found.[/]\n"
                f"  Local path: [{STYLE_PATH}]{CONFIG['storage']['local_dir']}[/]\n"
                f"  Record first: [{STYLE_CMD}]uv run python main.py record <env_id>[/]"
            )
            raise SystemExit(1)

        # Group by platform
        atari_envs = [e for e in all_recorded_envs if _get_env_platform(e) == "Atari"]
        retro_envs = [e for e in all_recorded_envs if _get_env_platform(e) == "Stable-Retro"]
        vizdoom_envs = [e for e in all_recorded_envs if _get_env_platform(e) == "VizDoom"]

        entries = []
        env_id_map = []

        for env_id in atari_envs:
            entries.append(f"[Atari]  {env_id}")
            env_id_map.append(env_id)
        if atari_envs and (retro_envs or vizdoom_envs):
            entries.append("")
            env_id_map.append(None)

        for env_id in retro_envs:
            entries.append(f"[Stable-Retro]  {env_id}")
            env_id_map.append(env_id)
        if retro_envs and vizdoom_envs:
            entries.append("")
            env_id_map.append(None)

        for env_id in vizdoom_envs:
            entries.append(f"[VizDoom]  {env_id}")
            env_id_map.append(env_id)

        title = "  Select Recording\n"
        status_bar = "  ↑↓ navigate · / search · Enter select · Esc cancel"
    else:
        # Original behavior: list all available environments
        atari_envs = _get_atari_envs()
        retro_envs = _get_stableretro_envs(imported_only=True)
        vizdoom_envs = _get_vizdoom_envs()

        entries = []
        env_id_map = []

        for env_id in atari_envs:
            entries.append(f"[Atari]  {env_id}")
            env_id_map.append(env_id)
        if atari_envs and (retro_envs or vizdoom_envs):
            entries.append("")
            env_id_map.append(None)

        for env_id in retro_envs:
            entries.append(f"[Stable-Retro]  {env_id}")
            env_id_map.append(env_id)
        if retro_envs and vizdoom_envs:
            entries.append("")
            env_id_map.append(None)

        for env_id in vizdoom_envs:
            entries.append(f"[VizDoom]  {env_id}")
            env_id_map.append(env_id)

        if not entries:
            console.print(
                "[dim]No environments found. Install ale-py, stable-retro, or vizdoom.[/]"
            )
            raise SystemExit(1)

        title = "  Select Environment\n"
        status_bar = "  ↑↓ navigate · / search · Enter select · Esc cancel"

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
        lines = "\n".join(atari_ids)
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


def _import_roms(path: str):
    """Import ROMs into stable-retro from a directory or file."""
    import zipfile

    import stable_retro.data

    if not os.path.exists(path):
        console.print(f"[{STYLE_FAIL}]Error: Path not found: {path}[/]")
        return

    known_hashes = stable_retro.data.get_known_hashes()
    imported_games = 0

    def save_if_matches(filename, f):
        nonlocal imported_games
        try:
            data, hash = stable_retro.data.groom_rom(filename, f)
        except (OSError, ValueError):
            return
        if hash in known_hashes:
            game, ext, curpath = known_hashes[hash]
            game_path = os.path.join(curpath, game)
            rom_path = os.path.join(game_path, "rom%s" % ext)
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

    console.print(f"[{STYLE_SUCCESS}]Imported {imported_games} ROM(s)[/]")


def _distribute_episodes(total, num_workers):
    """Divide total episodes evenly among workers, distributing remainder to first workers."""
    base = total // num_workers
    remainder = total % num_workers
    return [base + (1 if i < remainder else 0) for i in range(num_workers)]


def _worker_collect_episodes(
    env_id,
    worker_id,
    num_episodes,
    max_steps,
    agent_type,
    fps,
    progress_queue,
    output_dir,
):
    """Top-level worker function for parallel episode collection (must be picklable)."""
    try:
        _lazy_init()
        env = create_env(env_id)
        if fps is None:
            fps = get_default_fps(env)

        if agent_type == "random":
            policy = RandomPolicy(env.action_space)
        elif agent_type == "mario":
            policy = MarioRightJumpPolicy(env.action_space)
        elif agent_type == "breakout":
            policy = BreakoutCatcherPolicy(env.action_space, env=env)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        input_source = AgentInputSource(policy, headless=True)
        recorder = DatasetRecorderWrapper(
            env, input_source=input_source, headless=True, collector=agent_type
        )

        def progress_callback(episode_number, steps_in_episode):
            try:
                progress_queue.put(
                    ("progress", worker_id, episode_number, steps_in_episode),
                    block=False,
                )
            except Exception:
                pass

        def step_callback(episode_number, step_number):
            try:
                progress_queue.put(
                    ("step", worker_id, episode_number, step_number),
                    block=False,
                )
            except Exception:
                pass

        import asyncio as _asyncio

        recorded_dataset = _asyncio.run(
            recorder.record(
                fps=fps,
                max_episodes=num_episodes,
                max_steps=max_steps,
                progress_callback=progress_callback,
                step_callback=step_callback,
            )
        )

        env_metadata = recorder._env_metadata

        if recorded_dataset is not None:
            worker_path = os.path.join(output_dir, f"worker_{worker_id}")
            recorded_dataset.save_to_disk(worker_path)
            progress_queue.put(("done", worker_id, worker_path, env_metadata))
        else:
            progress_queue.put(("done", worker_id, None, None))
        recorder.close()  # cleanup temp files after dataset is saved

    except Exception as e:
        import traceback

        progress_queue.put(("error", worker_id, str(e), traceback.format_exc()))


def _parallel_record(env_id, num_workers, total_episodes, max_steps, agent_type, fps):
    """Run parallel episode collection across multiple worker processes."""
    from datasets import concatenate_datasets
    from datasets import load_from_disk as _load_from_disk

    episode_counts = _distribute_episodes(total_episodes, num_workers)
    # Filter out workers with 0 episodes
    active_counts = [(i, count) for i, count in enumerate(episode_counts) if count > 0]
    actual_workers = len(active_counts)

    output_dir = tempfile.mkdtemp(prefix="gymrec_parallel_")

    ctx = multiprocessing.get_context("spawn")
    progress_queue = ctx.Queue()

    console.print(
        f"[{STYLE_INFO}]Starting {actual_workers} worker(s) to collect {total_episodes} episode(s)[/]"
    )

    processes = []
    for worker_id, num_eps in active_counts:
        p = ctx.Process(
            target=_worker_collect_episodes,
            args=(
                env_id,
                worker_id,
                num_eps,
                max_steps,
                agent_type,
                fps,
                progress_queue,
                output_dir,
            ),
            daemon=True,
        )
        p.start()
        processes.append((worker_id, p))

    worker_paths = {}
    worker_metadata = {}
    completed_workers = 0
    total_steps = 0
    episodes_done = 0

    # Track per-worker state
    worker_states = {wid: {"episode": 0, "step": 0} for wid, _ in active_counts}

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        # Create per-worker progress bars
        worker_task_ids = {}
        for wid, num_eps in active_counts:
            worker_task_ids[wid] = progress.add_task(
                f"[cyan]Worker {wid}[/] [dim]waiting...[/]",
                total=num_eps,
            )

        # Main overall progress bar
        main_task_id = progress.add_task("[bold]Episodes[/]", total=total_episodes)

        try:
            while completed_workers < actual_workers:
                # Check for dead workers
                for worker_id, p in processes:
                    if not p.is_alive() and p.exitcode != 0 and worker_id not in worker_paths:
                        console.print(
                            f"[{STYLE_FAIL}]Worker {worker_id} crashed (exit code {p.exitcode})[/]"
                        )
                        worker_paths[worker_id] = None
                        completed_workers += 1

                try:
                    msg = progress_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                msg_type = msg[0]
                if msg_type == "step":
                    # Live step update from a worker
                    _, wid, episode_number, step_number = msg
                    worker_states[wid]["episode"] = episode_number
                    worker_states[wid]["step"] = step_number
                    # Calculate total steps across all workers
                    current_total_steps = sum(s["step"] for s in worker_states.values())
                    progress.update(
                        worker_task_ids[wid],
                        description=f"[cyan]Worker {wid}[/] [dim]Ep {episode_number}, Step {step_number}[/]",
                    )
                    # Update main progress bar with live step count (don't advance, just update description)
                    progress.update(
                        main_task_id,
                        description=f"[bold]Episodes[/] [dim]({current_total_steps} steps)[/]",
                    )
                elif msg_type == "progress":
                    # Episode completed
                    _, wid, episode_number, steps_in_episode = msg
                    total_steps += steps_in_episode
                    episodes_done += 1
                    worker_states[wid]["episode"] = episode_number
                    worker_states[wid]["step"] = 0
                    progress.update(
                        worker_task_ids[wid],
                        advance=1,
                        description=f"[cyan]Worker {wid}[/] [dim]Ep {episode_number} done ({steps_in_episode} steps)[/]",
                    )
                    progress.update(
                        main_task_id,
                        advance=1,
                        description=f"[bold]Episodes[/] [dim]({total_steps} steps total)[/]",
                    )
                elif msg_type == "done":
                    _, wid, path, metadata = msg
                    worker_paths[wid] = path
                    if metadata is not None:
                        worker_metadata[wid] = metadata
                    # Mark worker task as complete
                    if wid in worker_task_ids:
                        progress.update(
                            worker_task_ids[wid],
                            description=f"[green]Worker {wid}[/] [dim]complete[/]",
                        )
                    completed_workers += 1
                elif msg_type == "error":
                    _, wid, err_msg, tb = msg
                    console.print(f"[{STYLE_FAIL}]Worker {wid} error: {err_msg}[/]")
                    if wid in worker_task_ids:
                        progress.update(
                            worker_task_ids[wid],
                            description=f"[red]Worker {wid}[/] [dim]error[/]",
                        )
                    worker_paths[wid] = None
                    completed_workers += 1

        except KeyboardInterrupt:
            console.print(f"\n[{STYLE_INFO}]Interrupted — terminating workers...[/]")
            for _, p in processes:
                p.terminate()
            for _, p in processes:
                p.join(timeout=5)
            # Drain remaining messages
            while True:
                try:
                    msg = progress_queue.get_nowait()
                    if msg[0] == "done" and msg[2] is not None:
                        worker_paths[msg[1]] = msg[2]
                        if msg[3] is not None:
                            worker_metadata[msg[1]] = msg[3]
                except queue.Empty:
                    break

    for _, p in processes:
        p.join(timeout=10)

    # Collect valid datasets
    valid_paths = [p for p in worker_paths.values() if p is not None and os.path.exists(p)]
    if not valid_paths:
        shutil.rmtree(output_dir, ignore_errors=True)
        console.print(f"[{STYLE_FAIL}]No data collected from any worker.[/]")
        return None, None

    datasets = [_load_from_disk(p, keep_in_memory=True) for p in valid_paths]
    merged = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

    # Use metadata from first successful worker
    merged_metadata = next(iter(worker_metadata.values()), None) if worker_metadata else None

    shutil.rmtree(output_dir, ignore_errors=True)
    return merged, merged_metadata


async def main():
    parser = argparse.ArgumentParser(description="Gymnasium Recorder/Playback")
    subparsers = parser.add_subparsers(dest="command")

    parser_record = subparsers.add_parser("record", help="Record gameplay")
    parser_record.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )
    parser_record.add_argument(
        "--fps", type=int, default=None, help="Frames per second for playback/recording"
    )
    parser_record.add_argument(
        "--scale", type=int, default=None, help="Display scale factor (default: 2)"
    )
    parser_record.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Record without uploading to Hugging Face (no HF account required)",
    )
    parser_record.add_argument(
        "--agent",
        type=str,
        default="human",
        choices=["human", "random", "mario", "breakout"],
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
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for agent data collection (default: 1)",
    )

    parser_playback = subparsers.add_parser("playback", help="Replay a dataset")
    parser_playback.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )
    parser_playback.add_argument(
        "--fps", type=int, default=None, help="Frames per second for playback/recording"
    )
    parser_playback.add_argument(
        "--scale", type=int, default=None, help="Display scale factor (default: 2)"
    )
    parser_playback.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify determinism by comparing replayed frames against recorded frames (pixel MSE)",
    )

    parser_video = subparsers.add_parser(
        "video", help="Render recorded dataset episodes to a video file"
    )
    parser_video.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )
    parser_video.add_argument(
        "--fps", type=int, default=None, help="Frames per second for exported video"
    )
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
    parser_upload.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )

    subparsers.add_parser("login", help="Log in to Hugging Face Hub")
    subparsers.add_parser("list_environments", help="List available environments")

    parser_import = subparsers.add_parser(
        "import_roms", help="Import ROMs into stable-retro from a directory or file"
    )
    parser_import.add_argument(
        "path",
        type=str,
        help="Path to directory or file containing ROMs",
    )

    parser_minari = subparsers.add_parser(
        "minari-export", help="Export local dataset to Minari format"
    )
    parser_minari.add_argument(
        "env_id",
        type=str,
        nargs="?",
        default=None,
        help="Gymnasium environment id (e.g. BreakoutNoFrameskip-v4)",
    )
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
    if args.command is None:
        args.command = "record"
        for attr, default in [
            ("env_id", None),
            ("fps", None),
            ("scale", None),
            ("dry_run", False),
            ("agent", "human"),
            ("headless", False),
            ("episodes", None),
            ("max_steps", None),
            ("workers", 1),
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

    if args.command == "import_roms":
        _import_roms(args.path)
        return

    env_id = args.env_id

    _lazy_init()

    if env_id is None:
        # For playback, only show environments with available recordings
        is_recording_command = args.command in ("playback", "video")
        env_id = select_environment_interactive(available_recordings_only=is_recording_command)

    if args.command == "upload":
        upload_local_dataset(env_id)
        return

    if args.command == "minari-export":
        minari_export(env_id, dataset_name=args.name, author=args.author)
        return

    if hasattr(args, "scale") and args.scale is not None:
        CONFIG["display"]["scale_factor"] = args.scale

    env = None
    fps = args.fps
    if args.command in ("record", "playback"):
        env = create_env(env_id)
        if fps is None:
            fps = get_default_fps(env)
    elif args.command == "video" and fps is None:
        fps = _get_default_fps_for_env_id(env_id, metadata=load_local_metadata(env_id))

    if args.command == "record":
        recorder = None
        # Setup input source based on --agent flag
        if args.agent == "human":
            if args.headless:
                console.print(
                    f"[{STYLE_FAIL}]Error: --headless can only be used with --agent (not human mode)[/]"
                )
                return
            input_source = None  # Will default to HumanInputSource in _play
            max_episodes = args.episodes  # None = unlimited for human
            max_steps = getattr(args, "max_steps", None)
            recorder = DatasetRecorderWrapper(
                env, input_source=input_source, headless=False, collector="human"
            )
            recorded_dataset = await recorder.record(
                fps=fps, max_episodes=max_episodes, max_steps=max_steps
            )
        else:
            # Agent mode
            if args.agent not in ("random", "mario", "breakout"):
                console.print(f"[{STYLE_FAIL}]Error: Unknown agent type '{args.agent}'[/]")
                return

            # Validate --workers usage
            workers = getattr(args, "workers", 1)
            if workers > 1 and args.agent == "human":
                console.print(
                    f"[{STYLE_FAIL}]Error: --workers requires --agent (not human mode)[/]"
                )
                return
            if workers > 1:
                args.headless = True  # Force headless for parallel workers

            # Don't allow headless mode without specifying episodes
            if args.headless and args.episodes is None:
                console.print(
                    f"[{STYLE_FAIL}]Error: --headless requires --episodes to be specified[/]"
                )
                return

            # Default to 1 episode for agent if not specified
            max_episodes = args.episodes if args.episodes is not None else 1
            max_steps = getattr(args, "max_steps", None)

            # Don't allow more workers than episodes
            if workers > max_episodes:
                console.print(
                    f"[{STYLE_FAIL}]Error: --workers ({workers}) cannot be greater than --episodes ({max_episodes})[/]"
                )
                return

            mode_str = "headless" if args.headless else "with display"
            console.print(
                f"[{STYLE_INFO}]Recording with {args.agent} agent ({mode_str}), {max_episodes} episode(s)[/]"
            )

            if workers > 1:
                # Parallel collection path
                recorded_dataset, env_metadata = _parallel_record(
                    env_id=env_id,
                    num_workers=workers,
                    total_episodes=max_episodes,
                    max_steps=max_steps,
                    agent_type=args.agent,
                    fps=fps,
                )
                if recorded_dataset is None:
                    env.close()
                    return
                save_dataset_locally(recorded_dataset, env_id, metadata=env_metadata)
                env.close()

            else:
                # Single-worker agent path with progress bar
                if args.agent == "random":
                    policy = RandomPolicy(env.action_space)
                elif args.agent == "mario":
                    policy = MarioRightJumpPolicy(env.action_space)
                elif args.agent == "breakout":
                    policy = BreakoutCatcherPolicy(env.action_space, env=env)
                input_source = AgentInputSource(policy, headless=args.headless)

                recorder = DatasetRecorderWrapper(
                    env,
                    input_source=input_source,
                    headless=args.headless,
                    collector=args.agent,
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

                with Progress(
                    SpinnerColumn(),
                    TextColumn("{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=False,
                ) as progress:
                    task_id = progress.add_task("[bold]Episodes[/]", total=max_episodes)
                    recorded_dataset = await recorder.record(
                        fps=fps,
                        max_episodes=max_episodes,
                        max_steps=max_steps,
                        progress_callback=_make_progress_callback(task_id, progress),
                    )

        if recorded_dataset is None:
            if recorder is not None:
                recorder.close()
            return

        if recorder is not None:
            save_dataset_locally(recorded_dataset, env_id, metadata=recorder._env_metadata)
            recorder.close()  # cleanup temp files after dataset is saved
        console.print(f"To play back: [{STYLE_CMD}]uv run python main.py playback {env_id}[/]")

        if not args.dry_run:
            try:
                do_upload = Confirm.ask(
                    "Upload to Hugging Face Hub?", default=True, console=console
                )
            except EOFError:
                do_upload = False
            if do_upload:
                if not upload_local_dataset(env_id):
                    console.print(
                        f"To retry: [{STYLE_CMD}]uv run python main.py upload {env_id}[/]"
                    )
            else:
                console.print(
                    f"To upload later: [{STYLE_CMD}]uv run python main.py upload {env_id}[/]"
                )
    elif args.command == "playback":
        loaded_dataset = load_local_dataset(env_id)
        if loaded_dataset is not None:
            console.print(
                f"[{STYLE_INFO}]Playing back from local dataset ({len(loaded_dataset)} frames)[/]"
            )
            total = len(loaded_dataset)
        else:
            console.print("[dim]No local dataset found, trying Hugging Face Hub...[/]")
            try:
                hf_repo_id = env_id_to_hf_repo_id(env_id)
                api = HfApi()
                api.dataset_info(hf_repo_id)
                loaded_dataset = load_dataset(hf_repo_id, split="train", streaming=True)
            except Exception:
                loaded_dataset = None
            if loaded_dataset is None:
                console.print(f"[{STYLE_FAIL}]No dataset found for {env_id}.[/]")
                console.print(f"  Local path: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]")
                console.print(
                    f"  Record a session first: [{STYLE_CMD}]uv run python main.py record {env_id}[/]"
                )
                return
            try:
                builder = load_dataset_builder(hf_repo_id)
                if builder.info.splits and "train" in builder.info.splits:
                    total = builder.info.splits["train"].num_examples
                else:
                    total = None
            except Exception:
                total = None
            console.print(f"[{STYLE_INFO}]Playing back streaming from Hugging Face Hub[/]")
        recorder = DatasetRecorderWrapper(env)

        def _is_step_row(row):
            """Filter out terminal observation rows."""
            action = row.get("actions", row.get("action"))
            if action is None or (isinstance(action, list) and len(action) == 0):
                return False
            return True

        if args.verify:
            recorded_data = (
                (
                    row.get("actions", row.get("action")),
                    row.get("observations", row.get("observation")),
                    row.get("rewards", row.get("reward")),
                    row.get("terminations", row.get("termination")),
                    row.get("truncations", row.get("truncation")),
                )
                for row in loaded_dataset
                if _is_step_row(row)
            )
            await recorder.replay(recorded_data, fps=fps, total=total, verify=True)
        else:
            actions = (
                row.get("actions", row.get("action")) for row in loaded_dataset if _is_step_row(row)
            )
            await recorder.replay(actions, fps=fps, total=total)
    elif args.command == "video":
        loaded_dataset = load_local_dataset(env_id)
        if loaded_dataset is not None:
            console.print(
                f"[{STYLE_INFO}]Loading local dataset for video export "
                f"({len(loaded_dataset)} frames)[/]"
            )
        else:
            console.print("[dim]No local dataset found, downloading from Hugging Face Hub...[/]")
            try:
                hf_repo_id = env_id_to_hf_repo_id(env_id)
                api = HfApi()
                api.dataset_info(hf_repo_id)
                loaded_dataset = load_dataset(hf_repo_id, split="train")
                console.print(
                    f"[{STYLE_INFO}]Loaded dataset from Hugging Face Hub "
                    f"({len(loaded_dataset)} frames)[/]"
                )
            except Exception:
                loaded_dataset = None

        if loaded_dataset is None:
            console.print(f"[{STYLE_FAIL}]No dataset found for {env_id}.[/]")
            console.print(f"  Local path: [{STYLE_PATH}]{get_local_dataset_path(env_id)}[/]")
            console.print(
                f"  Record a session first: [{STYLE_CMD}]uv run python main.py record {env_id}[/]"
            )
            return

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
    asyncio.run(main())


if __name__ == "__main__":
    cli()
