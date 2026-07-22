from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, Image, Value, load_from_disk

from finalize_pending import (
    CASTS,
    COMMON_COLUMNS,
    IMAGE_COLUMNS,
    _safe_inventory,
    finalize_pending,
)


def _write_finalized_package(queue: Path, episode_uuid: uuid.UUID) -> Path:
    package = queue / episode_uuid.hex
    package.mkdir(parents=True)
    environment = {
        "document_type": "gymrec.environment",
        "format_version": 1,
        "provider_id": "stable-retro-turbo",
        "provider_contract_version": 1,
        "environment_id": "fixture-v0",
        "declared_config": {},
        "effective_config": {},
        "provenance": {
            "distribution": "stable-retro-turbo",
            "version": "1.0.1.post33",
            "assets": {},
        },
        "action_space": {"type": "Discrete", "n": 2},
        "observation_space": {
            "type": "Box",
            "shape": [3, 4, 3],
            "dtype": "uint8",
        },
        "control_profile": "stable_retro.Nes",
        "fps": 30.0,
    }
    environment_id = hashlib.sha256(
        json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    context = {
        "episode_id": str(episode_uuid),
        "seed": 4,
        "session_id": str(uuid.uuid4()),
        "dataset_format_version": 3,
        "collector": "random",
        "gymrec_version": "0.1.1",
        "storage_format": "images",
        "provider_id": "stable-retro-turbo",
        "env_id": "fixture-v0",
        "environment_contract_id": environment_id,
        "collector_contract_id": None,
        "policy_mode": None,
        "policy_seed": None,
    }
    rows = [
        {
            **context,
            "step_index": 0,
            "actions": 1,
            "policy_actions": None,
            "rewards": 1.0,
            "terminations": True,
            "truncations": False,
            "infos": "{}",
            "collector_terminated": False,
            "observations": np.zeros((3, 4, 3), dtype=np.uint8),
        },
        {
            **context,
            "step_index": 1,
            "actions": None,
            "policy_actions": None,
            "rewards": None,
            "terminations": None,
            "truncations": None,
            "infos": None,
            "collector_terminated": False,
            "observations": np.ones((3, 4, 3), dtype=np.uint8),
        },
    ]
    columns = (*COMMON_COLUMNS, *IMAGE_COLUMNS)
    dataset = Dataset.from_dict({name: [row[name] for row in rows] for name in columns})
    for name, dtype in CASTS.items():
        if name in dataset.column_names:
            dataset = dataset.cast_column(name, Value(dtype))
    dataset = dataset.cast_column("observations", Image())
    dataset.to_parquet(package / "episode.parquet")
    artifact = package / "environments" / environment_id / "environment.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(environment, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return package


def test_finalize_pending_is_non_destructive_and_writes_adoptable_groups(tmp_path):
    queue = tmp_path / "live_pending"
    package = _write_finalized_package(queue, uuid.uuid4())
    before = _safe_inventory(package)
    output = tmp_path / "migration"

    report = finalize_pending(str(queue), output, assume_stopped=False)

    assert _safe_inventory(package) == before
    assert len(report["groups"]) == 1
    group = output / report["groups"][0]
    dataset = load_from_disk(group)
    assert len(dataset) == 2
    assert (output / "report.json").is_file()


def test_finalize_pending_rejects_duplicate_uuid_aliases(tmp_path):
    queue = tmp_path / "live_pending"
    episode_uuid = uuid.uuid4()
    _write_finalized_package(queue, episode_uuid)
    (queue / str(episode_uuid)).mkdir()

    with pytest.raises(ValueError, match="duplicate UUID aliases"):
        finalize_pending(str(queue), tmp_path / "migration", assume_stopped=False)


def test_finalize_pending_rejects_source_symlink(tmp_path):
    queue = tmp_path / "live_pending"
    package = _write_finalized_package(queue, uuid.uuid4())
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (package / "unsafe").symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link"):
        finalize_pending(str(queue), tmp_path / "migration", assume_stopped=False)


def test_finalize_pending_rejects_output_inside_source(tmp_path):
    queue = tmp_path / "live_pending"
    _write_finalized_package(queue, uuid.uuid4())

    with pytest.raises(ValueError, match="outside every Gymrec pending source"):
        finalize_pending(str(queue), queue / "migration", assume_stopped=False)
