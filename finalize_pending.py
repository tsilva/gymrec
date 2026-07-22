"""Offline, non-destructive Gymrec v3 pending-package finalizer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import tomllib
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from datasets import Dataset, Image, Value, concatenate_datasets
from PIL import Image as PILImage

COMMON_COLUMNS = (
    "episode_id",
    "step_index",
    "seed",
    "actions",
    "policy_actions",
    "rewards",
    "terminations",
    "truncations",
    "infos",
    "session_id",
    "dataset_format_version",
    "collector",
    "gymrec_version",
    "storage_format",
    "provider_id",
    "env_id",
    "environment_contract_id",
    "collector_contract_id",
    "policy_mode",
    "policy_seed",
    "collector_terminated",
)
IMAGE_COLUMNS = ("observations",)
VIDEO_COLUMNS = ("video_path", "frame_sha256", "frame_width", "frame_height")
TRANSITION_COLUMNS = (
    "actions",
    "policy_actions",
    "rewards",
    "terminations",
    "truncations",
    "infos",
)
CASTS = {
    "episode_id": "string",
    "step_index": "int64",
    "seed": "int64",
    "rewards": "float64",
    "terminations": "bool",
    "truncations": "bool",
    "infos": "string",
    "session_id": "string",
    "dataset_format_version": "int64",
    "collector": "string",
    "gymrec_version": "string",
    "storage_format": "string",
    "provider_id": "string",
    "env_id": "string",
    "environment_contract_id": "string",
    "collector_contract_id": "string",
    "policy_mode": "string",
    "policy_seed": "int64",
    "collector_terminated": "bool",
    "video_path": "string",
    "frame_sha256": "string",
    "frame_width": "int64",
    "frame_height": "int64",
}
MAX_DECODED_BYTES = 32 * 1024**3


@dataclass
class RecoveredPackage:
    episode_id: str
    dataset: Dataset
    source_root: Path
    video_overrides: Mapping[str, Path]
    identity: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symbolic link in pending package: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"non-regular pending package entry: {path}")
        result[path.relative_to(root).as_posix()] = _sha256(path)
    return result


def _safe_path(root: Path, value: Any) -> Path:
    relative = PurePosixPath(str(value or ""))
    if (
        not str(value or "")
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in str(value)
    ):
        raise ValueError(f"unsafe package-relative path {value!r}")
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=True).relative_to(root.resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"package path escapes or is missing: {value!r}") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"package path is not a regular file: {value!r}")
    return candidate


def _canonical_episode_id(name: str) -> str:
    parsed = uuid.UUID(name)
    canonical = str(parsed)
    if name not in {canonical, parsed.hex} or name.lower() != name:
        raise ValueError(f"noncanonical pending package directory {name!r}")
    return canonical


def _read_journal(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    raw_lines = path.read_bytes().splitlines()
    records = []
    for index, raw in enumerate(raw_lines):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except UnicodeDecodeError, json.JSONDecodeError:
            if index == len(raw_lines) - 1:
                break
            raise ValueError(f"torn journal record before the final line at {path}")
        if not isinstance(value, Mapping):
            raise ValueError(f"journal record is not an object at {path}")
        records.append(value)
    return records


def _journal_row(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record.get("row"), Mapping):
        raise ValueError("pending journal is not current Gymrec v3")
    return dict(record["row"])


def _validate_dataset(dataset: Dataset, episode_id: str) -> str:
    if len(dataset) < 2:
        raise ValueError(f"episode {episode_id} has no transitions")
    storage = str(dataset[0].get("storage_format"))
    expected = (*COMMON_COLUMNS, *(IMAGE_COLUMNS if storage == "images" else VIDEO_COLUMNS))
    if tuple(dataset.column_names) != expected or storage not in {"images", "lossless-video"}:
        raise ValueError(f"episode {episode_id} is not canonical Gymrec v3")
    rows = [dataset[index] for index in range(len(dataset))]
    if {str(row["episode_id"]) for row in rows} != {episode_id}:
        raise ValueError(f"episode UUID does not match package {episode_id}")
    if [int(row["step_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"episode {episode_id} has noncontiguous steps")
    context = {
        key: rows[0].get(key)
        for key in COMMON_COLUMNS
        if key not in {*TRANSITION_COLUMNS, "step_index", "collector_terminated"}
    }
    for row in rows:
        if row.get("dataset_format_version") != 3:
            raise ValueError(f"episode {episode_id} is not v3")
        if any(row.get(key) != value for key, value in context.items()):
            raise ValueError(f"episode {episode_id} changes context")
    for row in rows[:-1]:
        if any(row.get(key) is None for key in TRANSITION_COLUMNS if key != "policy_actions"):
            raise ValueError(f"episode {episode_id} has an incomplete transition")
    if any(rows[-1].get(key) is not None for key in TRANSITION_COLUMNS):
        raise ValueError(f"episode {episode_id} has a malformed terminal row")
    return storage


def _decode_video(path: Path, *, width: int, height: int) -> list[np.ndarray]:
    frame_bytes = width * height * 3
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-f",
            "matroska",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if completed.returncode:
        raise ValueError(completed.stderr.decode(errors="replace")[-1000:])
    if len(completed.stdout) > MAX_DECODED_BYTES or len(completed.stdout) % frame_bytes:
        raise ValueError(f"invalid decoded video size at {path}")
    return [
        np.frombuffer(completed.stdout[index : index + frame_bytes], dtype=np.uint8)
        .reshape(height, width, 3)
        .copy()
        for index in range(0, len(completed.stdout), frame_bytes)
    ]


def _encode_video(frames: Sequence[np.ndarray], output: Path, fps: float) -> None:
    height, width = frames[0].shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(float(fps)),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "veryslow",
            "-f",
            "matroska",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames)
    _stdout, stderr = process.communicate(payload, timeout=300)
    if process.returncode:
        output.unlink(missing_ok=True)
        raise ValueError(stderr.decode(errors="replace")[-1000:])


def _verify_media(dataset: Dataset, root: Path) -> None:
    rows = [dataset[index] for index in range(len(dataset))]
    storage = str(rows[0]["storage_format"])
    if storage == "images":
        for row in rows:
            value = row["observations"]
            if isinstance(value, Mapping) and value.get("path"):
                _safe_path(root, value["path"])
        return
    relative = str(rows[0]["video_path"])
    if not relative.endswith(".rgb.mkv.bin") or {row["video_path"] for row in rows} != {relative}:
        raise ValueError("pending video paths are not canonical")
    path = _safe_path(root, relative)
    width, height = int(rows[0]["frame_width"]), int(rows[0]["frame_height"])
    frames = _decode_video(path, width=width, height=height)
    if len(frames) != len(rows):
        raise ValueError("pending video frame count differs from Parquet")
    for row, frame in zip(rows, frames, strict=True):
        if hashlib.sha256(frame.tobytes()).hexdigest() != row["frame_sha256"]:
            raise ValueError("pending video frame hash differs from Parquet")


def _journal_dataset(
    package: Path,
    episode_id: str,
    records: Sequence[Mapping[str, Any]],
    work: Path,
) -> tuple[Dataset, Mapping[str, Path]]:
    if not records:
        raise ValueError("pending package has no recoverable journal prefix")
    records = sorted(records, key=lambda value: int(value.get("row_index", -1)))
    rows = [_journal_row(record) for record in records]
    if [int(row["step_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("pending journal steps are not contiguous")
    last = records[-1]
    terminal_path = _safe_path(package, last.get("terminal_candidate_path"))
    with PILImage.open(terminal_path) as image:
        terminal_frame = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if hashlib.sha256(terminal_frame.tobytes()).hexdigest() != last.get(
        "terminal_candidate_sha256"
    ):
        raise ValueError("terminal candidate hash mismatch")
    context = {
        key: rows[0].get(key)
        for key in COMMON_COLUMNS
        if key not in {*TRANSITION_COLUMNS, "step_index", "collector_terminated"}
    }
    terminal = {key: context.get(key) for key in COMMON_COLUMNS}
    terminal.update(
        {
            "episode_id": episode_id,
            "step_index": len(rows),
            **{key: None for key in TRANSITION_COLUMNS},
            "collector_terminated": not (
                bool(rows[-1]["terminations"]) or bool(rows[-1]["truncations"])
            ),
        }
    )
    storage = str(rows[0]["storage_format"])
    overrides: dict[str, Path] = {}
    if storage == "images":
        for row in rows:
            row["observations"] = str(_safe_path(package, row["observations"]))
        terminal["observations"] = str(terminal_path)
    elif storage == "lossless-video":
        relative = str(rows[0]["video_path"])
        video = _safe_path(package, relative)
        width, height = int(rows[0]["frame_width"]), int(rows[0]["frame_height"])
        frames = _decode_video(video, width=width, height=height)
        expected = [str(row["frame_sha256"]) for row in rows]
        if len(frames) not in {len(rows), len(rows) + 1}:
            raise ValueError("video does not prove the journal prefix")
        if [
            hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames[: len(rows)]
        ] != expected:
            raise ValueError("video prefix does not match the journal")
        terminal_hash = hashlib.sha256(terminal_frame.tobytes()).hexdigest()
        if len(frames) == len(rows):
            frames.append(terminal_frame)
            recovered = work / relative
            _encode_video(frames, recovered, float(last["fps"]))
            overrides[relative] = recovered
        elif hashlib.sha256(frames[-1].tobytes()).hexdigest() != terminal_hash:
            raise ValueError("video terminal frame does not match the candidate")
        terminal.update(
            {
                "video_path": relative,
                "frame_sha256": terminal_hash,
                "frame_width": width,
                "frame_height": height,
            }
        )
    else:
        raise ValueError(f"unsupported journal storage {storage!r}")
    rows.append(terminal)
    columns = (*COMMON_COLUMNS, *(IMAGE_COLUMNS if storage == "images" else VIDEO_COLUMNS))
    dataset = Dataset.from_dict({name: [row.get(name) for row in rows] for name in columns})
    for name, dtype in CASTS.items():
        if name in dataset.column_names:
            dataset = dataset.cast_column(name, Value(dtype))
    if storage == "images":
        dataset = dataset.cast_column("observations", Image())
    _validate_dataset(dataset, episode_id)
    return dataset, overrides


def _journal_prefix_matches(dataset: Dataset, records: Sequence[Mapping[str, Any]]) -> bool:
    if len(records) > len(dataset) - 1:
        return False
    for index, record in enumerate(records):
        journal = _journal_row(record)
        row = dataset[index]
        for key in COMMON_COLUMNS:
            if journal.get(key) != row.get(key):
                return False
    return True


def _copy_artifacts(package: RecoveredPackage, target: Path) -> None:
    for name in ("environments", "collectors"):
        source = package.source_root / name
        if source.exists():
            for source_file in source.rglob("*"):
                if source_file.is_dir():
                    continue
                relative = source_file.relative_to(package.source_root)
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and _sha256(destination) != _sha256(source_file):
                    raise ValueError(f"immutable group artifact conflict at {relative}")
                if not destination.exists():
                    shutil.copy2(source_file, destination)
    for relative in (
        set(package.dataset["video_path"]) if "video_path" in package.dataset.column_names else ()
    ):
        source = package.video_overrides.get(str(relative)) or _safe_path(
            package.source_root, relative
        )
        destination = target.joinpath(*PurePosixPath(str(relative)).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and _sha256(destination) != _sha256(source):
            raise ValueError(f"group media conflict at {relative}")
        if not destination.exists():
            shutil.copy2(source, destination)


def _feature_document(dataset: Dataset) -> Mapping[str, Any]:
    return dataset.features.to_dict()


def _group_key(package: RecoveredPackage) -> str:
    row = package.dataset[0]
    context = {
        key: row.get(key)
        for key in (
            "storage_format",
            "provider_id",
            "env_id",
            "environment_contract_id",
            "collector_contract_id",
            "policy_mode",
        )
    }
    return hashlib.sha256(
        _canonical_json(
            {
                "identity": package.identity,
                "features": _feature_document(package.dataset),
                "context": context,
            }
        )
    ).hexdigest()


def _local_root() -> Path:
    config_path = Path(__file__).with_name("config.toml")
    root = Path.home() / ".gymrec" / "datasets"
    if config_path.is_file():
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        configured = config.get("storage", {}).get("local_dir")
        if configured:
            root = Path(str(configured)).expanduser()
    return root.resolve()


def _encode_env_id(value: str) -> str:
    return value.replace("_", "_underscore_").replace("-", "_dash_").replace("/", "_slash_")


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _queue_roots(reference: str | None) -> list[tuple[str, Path]]:
    local = _local_root()
    if reference:
        explicit = Path(reference).expanduser()
        if explicit.is_dir():
            return [(str(explicit.resolve()), explicit.resolve())]
        text = reference.removeprefix("hf://").strip("/")
        if reference.startswith("hf://"):
            parts = text.split("/")
            if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
                raise ValueError("invalid Gymrec repository recording reference")
            return [(reference, local / "repos" / parts[0] / parts[1] / "live_pending")]
        return [(reference, local / f"{_encode_env_id(reference)}_live_pending")]
    roots = []
    for path in local.glob("*_live_pending"):
        roots.append((path.name.removesuffix("_live_pending"), path))
    for path in local.glob("repos/*/*/live_pending"):
        roots.append((f"hf://{path.parts[-3]}/{path.parts[-2]}", path))
    return sorted(roots)


def finalize_pending(
    reference: str | None,
    output: Path,
    *,
    assume_stopped: bool,
) -> Mapping[str, Any]:
    output = output.expanduser()
    if output.exists():
        raise FileExistsError(f"output container already exists: {output}")
    queues = [(identity, path) for identity, path in _queue_roots(reference) if path.is_dir()]
    if not queues:
        raise FileNotFoundError("no matching Gymrec pending package root")
    resolved_output = output.resolve()
    for _identity, queue in queues:
        if resolved_output == queue or resolved_output.is_relative_to(queue):
            raise ValueError("bridge output must be outside every Gymrec pending source root")
    initial = {str(path): _safe_inventory(path) for _identity, path in queues}
    manifests = {}
    package_inventory = []
    aliases: dict[tuple[str, str], str] = {}
    for identity, queue in queues:
        manifest_path = queue / "manifest.json"
        manifest = {}
        if manifest_path.is_file():
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                manifest = loaded.get("episodes", {}) or {}
        manifests[str(queue)] = manifest
        for package in sorted(path for path in queue.iterdir() if path.is_dir()):
            try:
                episode_id = _canonical_episode_id(package.name)
            except ValueError, AttributeError:
                continue
            key = (identity, episode_id)
            if key in aliases:
                raise ValueError(f"duplicate UUID aliases {aliases[key]!r} and {package.name!r}")
            aliases[key] = package.name
            package_inventory.append((identity, queue, package, episode_id))
    needs_stability = []
    for identity, queue, package, episode_id in package_inventory:
        state = str(manifests[str(queue)].get(episode_id, {}).get("state") or "unknown")
        if not (package / "episode.parquet").is_file() and state not in {"pending", "failed"}:
            needs_stability.append((identity, queue, package, episode_id, state))
    if needs_stability and not assume_stopped:
        raise ValueError(
            "recording/journal-only packages require --assume-stopped after all writers stop"
        )
    if needs_stability:
        time.sleep(2.0)
        current = {str(path): _safe_inventory(path) for _identity, path in queues}
        changed = {name for name in initial if initial[name] != current[name]}
        trusted_inventory = current
    else:
        changed = set()
        trusted_inventory = initial

    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    recovered: list[RecoveredPackage] = []
    statuses = []
    try:
        for identity, queue, package, episode_id in package_inventory:
            status = {"identity": identity, "episode_id": episode_id, "status": "deferred"}
            try:
                if str(queue) in changed:
                    raise ValueError("source inventory changed during --assume-stopped check")
                records = _read_journal(package / "journal.jsonl")
                shard = package / "episode.parquet"
                if shard.is_file():
                    try:
                        dataset = Dataset.from_parquet(str(shard))
                    except Exception as parquet_exc:
                        if not records:
                            raise ValueError(
                                f"finalized Parquet is unreadable and no journal can recover it: "
                                f"{parquet_exc}"
                            ) from parquet_exc
                        dataset, overrides = _journal_dataset(
                            package, episode_id, records, temporary / "recovered" / episode_id
                        )
                        status["recovery"] = "journal_fallback_from_unreadable_parquet"
                    else:
                        _validate_dataset(dataset, episode_id)
                        if records and not _journal_prefix_matches(dataset, records):
                            raise ValueError("journal is not an exact prefix of finalized Parquet")
                        _verify_media(dataset, package)
                        overrides = {}
                else:
                    dataset, overrides = _journal_dataset(
                        package, episode_id, records, temporary / "recovered" / episode_id
                    )
                recovered.append(
                    RecoveredPackage(episode_id, dataset, package, overrides, identity)
                )
                status["status"] = "recovered"
                status["storage_format"] = dataset[0]["storage_format"]
            except Exception as exc:
                status["reason"] = str(exc)
            statuses.append(status)
        groups = defaultdict(list)
        for package in recovered:
            groups[_group_key(package)].append(package)
        for digest, packages in sorted(groups.items()):
            group = temporary / digest
            datasets = [package.dataset for package in packages]
            combined = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
            combined.save_to_disk(str(group))
            for package in packages:
                _copy_artifacts(package, group)
            (group / "bridge-group.json").write_bytes(
                _canonical_json(
                    {
                        "version": 1,
                        "group": digest,
                        "episodes": [package.episode_id for package in packages],
                        "source_commit": "08b0328c990d0955a0df12636117ce08458afb5b",
                    }
                )
            )
        final_inventory = {str(path): _safe_inventory(path) for _identity, path in queues}
        if final_inventory != trusted_inventory:
            raise RuntimeError("Gymrec pending source bytes changed during finalization")
        report = {
            "version": 1,
            "source_commit": "08b0328c990d0955a0df12636117ce08458afb5b",
            "assume_stopped": bool(assume_stopped),
            "groups": sorted(groups),
            "packages": statuses,
            "source_inventory_sha256": hashlib.sha256(
                _canonical_json(trusted_inventory)
            ).hexdigest(),
        }
        (temporary / "report.json").write_bytes(_canonical_json(report))
        _fsync_tree(temporary)
        os.replace(temporary, output)
        parent_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
