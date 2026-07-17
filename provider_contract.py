"""Environment-provider boundary used by gymrec.

Provider packages own environment construction and every environment semantic.
gymrec only validates this small envelope, records the provider's final Gymnasium
transitions, and persists the provider's effective contract for playback.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PROVIDER_ENTRY_POINT_GROUP = "gymrec.environment_providers"
SUPPORTED_PROVIDER_IDS = frozenset(
    {
        "stable-retro-turbo",
        "supermariobrosnes-turbo",
    }
)
ENVIRONMENT_CONTRACT_VERSION = 1
ENVIRONMENT_DOCUMENT_TYPE = "gymrec.environment"
ENVIRONMENT_DOCUMENT_FORMAT_VERSION = 1


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class EnvironmentContract:
    contract_version: int
    provider_id: str
    environment_id: str
    config: dict[str, Any]

    @classmethod
    def parse(cls, value: Mapping[str, Any], *, label: str = "environment"):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")
        expected = {"contract_version", "provider_id", "environment_id", "config"}
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing or extra:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("unsupported " + ", ".join(extra))
            raise ValueError(f"{label} has an invalid envelope ({'; '.join(details)})")
        version = value["contract_version"]
        if version != ENVIRONMENT_CONTRACT_VERSION:
            raise ValueError(
                f"{label}.contract_version is {version!r}; "
                f"expected {ENVIRONMENT_CONTRACT_VERSION}"
            )
        provider_id = value["provider_id"]
        if provider_id not in SUPPORTED_PROVIDER_IDS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDER_IDS))
            raise ValueError(
                f"Unsupported environment provider {provider_id!r}; supported providers: "
                f"{supported}"
            )
        environment_id = value["environment_id"]
        if not isinstance(environment_id, str) or not environment_id.strip():
            raise ValueError(f"{label}.environment_id must be a non-empty string")
        config = value["config"]
        if not isinstance(config, Mapping):
            raise ValueError(f"{label}.config must be an object")
        return cls(
            contract_version=version,
            provider_id=str(provider_id),
            environment_id=environment_id.strip(),
            config=copy.deepcopy(dict(config)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "provider_id": self.provider_id,
            "environment_id": self.environment_id,
            "config": copy.deepcopy(self.config),
        }


def _entry_points():
    points = importlib.metadata.entry_points()
    if hasattr(points, "select"):
        return tuple(points.select(group=PROVIDER_ENTRY_POINT_GROUP))
    return tuple(points.get(PROVIDER_ENTRY_POINT_GROUP, ()))


def discover_providers(*, entry_points=None) -> dict[str, Any]:
    """Load the two explicitly supported providers from package entry points."""
    discovered: dict[str, Any] = {}
    for entry_point in _entry_points() if entry_points is None else entry_points:
        provider_id = str(entry_point.name)
        if provider_id not in SUPPORTED_PROVIDER_IDS:
            continue
        if provider_id in discovered:
            raise RuntimeError(f"Multiple environment providers registered as {provider_id!r}")
        provider = entry_point.load()
        provider = provider() if isinstance(provider, type) else provider
        if getattr(provider, "provider_id", None) != provider_id:
            raise RuntimeError(
                f"Environment provider entry point {provider_id!r} loaded an object with "
                f"provider_id={getattr(provider, 'provider_id', None)!r}"
            )
        if getattr(provider, "contract_version", None) != ENVIRONMENT_CONTRACT_VERSION:
            raise RuntimeError(
                f"Environment provider {provider_id!r} does not implement gymrec provider "
                f"contract version {ENVIRONMENT_CONTRACT_VERSION}"
            )
        for method in ("create", "catalog"):
            if not callable(getattr(provider, method, None)):
                raise RuntimeError(
                    f"Environment provider {provider_id!r} is missing required method {method}()"
                )
        discovered[provider_id] = provider
    return discovered


def load_provider(provider_id: str, *, entry_points=None):
    if provider_id not in SUPPORTED_PROVIDER_IDS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDER_IDS))
        raise ValueError(
            f"Unsupported environment provider {provider_id!r}; supported providers: {supported}"
        )
    provider = discover_providers(entry_points=entry_points).get(provider_id)
    if provider is None:
        raise RuntimeError(
            f"Environment provider {provider_id!r} is not installed with a "
            f"{PROVIDER_ENTRY_POINT_GROUP!r} entry point. Install a provider release that "
            f"implements contract version {ENVIRONMENT_CONTRACT_VERSION}."
        )
    return provider


def create_session(contract: EnvironmentContract, *, render_mode: str, entry_points=None):
    provider = load_provider(contract.provider_id, entry_points=entry_points)
    session = provider.create(
        environment_id=contract.environment_id,
        config=copy.deepcopy(contract.config),
        render_mode=render_mode,
    )
    if getattr(session, "provider_id", None) != contract.provider_id:
        raise RuntimeError("Provider session returned a different provider_id")
    if getattr(session, "environment_id", None) != contract.environment_id:
        raise RuntimeError("Provider session returned a different environment_id")
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("Provider session is missing its Gymnasium env")
    for method in (
        "recording_observation",
        "policy_observation",
        "adapt_policy_action",
        "validate_policy",
        "action_from_labels",
    ):
        if not callable(getattr(session, method, None)):
            raise RuntimeError(
                f"Environment provider {contract.provider_id!r} session is missing {method}()"
            )
    return session


def build_environment_document(contract: EnvironmentContract, session) -> tuple[str, dict]:
    effective_config = getattr(session, "effective_config", None)
    provenance = getattr(session, "provenance", None)
    if not isinstance(effective_config, Mapping):
        raise RuntimeError("Provider session effective_config must be an object")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("Provider session provenance must be an object")
    if provenance.get("distribution") != contract.provider_id:
        raise RuntimeError("Provider provenance must identify its distribution")
    if not isinstance(provenance.get("version"), str) or not provenance["version"]:
        raise RuntimeError("Provider provenance must include a non-empty version")
    if not isinstance(provenance.get("assets"), Mapping):
        raise RuntimeError("Provider provenance must include an assets object")
    document = {
        "document_type": ENVIRONMENT_DOCUMENT_TYPE,
        "format_version": ENVIRONMENT_DOCUMENT_FORMAT_VERSION,
        "provider_id": contract.provider_id,
        "provider_contract_version": contract.contract_version,
        "environment_id": contract.environment_id,
        "declared_config": copy.deepcopy(contract.config),
        "effective_config": copy.deepcopy(dict(effective_config)),
        "provenance": copy.deepcopy(dict(provenance)),
        "action_space": space_contract(session.env.action_space),
        "observation_space": space_contract(session.env.observation_space),
        "control_profile": getattr(session, "control_profile", None),
        "fps": float(getattr(session, "fps", 0)),
    }
    if not math.isfinite(document["fps"]) or document["fps"] <= 0:
        raise RuntimeError("Provider session fps must be positive")
    document_id = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    return document_id, document


def validate_environment_document(document: Mapping[str, Any], expected_id: str | None = None):
    if not isinstance(document, Mapping):
        raise ValueError("Environment document must be an object")
    if document.get("document_type") != ENVIRONMENT_DOCUMENT_TYPE:
        raise ValueError("Unsupported environment document type")
    if document.get("format_version") != ENVIRONMENT_DOCUMENT_FORMAT_VERSION:
        raise ValueError("Unsupported environment document format version")
    expected_fields = {
        "document_type",
        "format_version",
        "provider_id",
        "provider_contract_version",
        "environment_id",
        "declared_config",
        "effective_config",
        "provenance",
        "action_space",
        "observation_space",
        "control_profile",
        "fps",
    }
    if set(document) != expected_fields:
        missing = sorted(expected_fields - set(document))
        extra = sorted(set(document) - expected_fields)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported " + ", ".join(extra))
        raise ValueError(f"Environment document has an invalid schema ({'; '.join(details)})")
    if not isinstance(document["declared_config"], Mapping):
        raise ValueError("Environment document declared_config must be an object")
    provenance = document["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("distribution") != document["provider_id"]
        or not isinstance(provenance.get("version"), str)
        or not provenance.get("version")
        or not isinstance(provenance.get("assets"), Mapping)
    ):
        raise ValueError("Environment document has invalid provider provenance")
    for field in ("action_space", "observation_space"):
        if not isinstance(document[field], Mapping):
            raise ValueError(f"Environment document {field} must be an object")
    control_profile = document["control_profile"]
    if control_profile is not None and (
        not isinstance(control_profile, str) or not control_profile.strip()
    ):
        raise ValueError("Environment document control_profile must be null or a string")
    fps = document["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Environment document fps must be a positive finite number")
    contract = EnvironmentContract.parse(
        {
            "contract_version": document.get("provider_contract_version"),
            "provider_id": document.get("provider_id"),
            "environment_id": document.get("environment_id"),
            "config": document.get("effective_config"),
        },
        label="environment document",
    )
    actual_id = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    if expected_id is not None and actual_id != expected_id:
        raise ValueError(f"Environment contract hash mismatch: {actual_id} != {expected_id}")
    return contract


def space_contract(space) -> dict[str, Any]:
    contract = {"type": type(space).__name__, "repr": str(space)}
    shape = getattr(space, "shape", None)
    if shape is not None:
        contract["shape"] = [int(value) for value in shape]
    dtype = getattr(space, "dtype", None)
    if dtype is not None:
        contract["dtype"] = str(dtype)
    count = getattr(space, "n", None)
    if count is not None:
        contract["n"] = int(count)
    return contract
