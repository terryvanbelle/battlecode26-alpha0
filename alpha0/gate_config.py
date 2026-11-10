"""Lightweight versioned operational gate configuration readers."""
from __future__ import annotations

import re


def training_excluded_maps(gate: dict) -> frozenset[str]:
    """Read gradient-data exclusions independently of evaluation stage names.

    Version 1 archives reserved maps under stages.holdout. Version 2 evaluates
    every map and keeps the original gradient exclusions in an explicit field.
    Unknown or malformed gate schemas must not silently permit training.
    """
    if not isinstance(gate, dict) or type(gate.get("schema_version")) is not int or gate["schema_version"] not in (1, 2):
        raise ValueError("unsupported victory gate schema for training exclusions")
    if "training_excluded_maps" in gate:
        names = gate["training_excluded_maps"]
    elif gate["schema_version"] == 1:
        try:
            names = gate["stages"]["holdout"]["maps"]
        except (KeyError, TypeError) as error:
            raise ValueError("legacy victory gate is missing training exclusions") from error
    else:
        raise ValueError("victory gate v2 requires training_excluded_maps")
    if (not isinstance(names, list) or not names or
            any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names) or
            len(set(names)) != len(names)):
        raise ValueError("training_excluded_maps must be a nonempty unique list of map names")
    if gate["schema_version"] == 1 and "training_excluded_maps" in gate:
        # A transitional explicit field cannot silently remove legacy exclusions.
        # Parse the legacy branch through the same validation rather than trusting
        # a mixed schema with conflicting or malformed fields.
        legacy = dict(gate)
        legacy.pop("training_excluded_maps")
        if frozenset(names) != training_excluded_maps(legacy):
            raise ValueError("conflicting legacy and explicit training exclusions")
    return frozenset(names)
