"""Consumables catalog indexer (AI-facing) – does NOT modify core_component.

This module builds a lightweight, LLM-friendly catalog for component
consumable specs by reusing the existing validator/spec compilation logic
from `core_component.validator`. We import (read-only) the spec library and
derive a flattened description of each spec (resource id, property paths,
required flags, enum values, documentation links).

Design Goals:
    * Zero writes / no pollution of core_component package
    * Pure read-only view via SpecLibrary (compilation already resolves
      custom type references and list item specs)
    * Deterministic output ordering for stable embeddings / caching
    * Minimal transformation – keep field names intuitive for downstream
      prompt assembly (avoid proprietary meta key naming)

Schema (per entry):
    {
        "id": "AWS::SNS::Topic",
        "category": "AWS/SNS/Topic",            # path derived from spec file tree
        "properties": [
            {
                "path": "Properties.DisplayName",
                "type": "string",
                "required": false,
                "enum": ["val1", ...],           # OPTIONAL
                "doc": "https://..."             # OPTIONAL (first doc link)
            },
            ...
        ],
        "enum_index": { "Properties.Protocol": ["http", "https"] },
        "doc_urls": ["https://..."],
        "source_spec_key": "AWS::SNS::Topic"      # original root key
    }

Limitations:
    * We rely on heuristic mapping of compiled spec meta keys (_Type,
      _Required, _StringEnum, _Documentation). If core_component changes
      naming, update META_KEYS.
    * Deep recursion limited to avoid pathological cycles – we trust
      SpecLibrary recursion limits already.

Future Extensions:
    * Add summary generation (one-line description) using small offline model
    * Emit hashed content signatures for change detection
    * Provide optional NDJSON streamer for large catalog diffs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import core_logging as log

try:
    # Importing validator library (read-only usage)
    from core_component.validator.spec_library import SpecLibrary  # type: ignore
except Exception as e:  # pragma: no cover - environment/import failure path
    raise ImportError(
        "Failed to import SpecLibrary from core_component.validator – ensure sck-core-component is installed."
    ) from e


META_KEYS = {
    "type": "_Type",
    "required": "_Required",
    "enum": "_StringEnum",
    "documentation": "_Documentation",
    "list_item_type": "_ListItemType",
    "list_item_spec": "_ListItemSpec",
}


@dataclass
class ConsumableProperty:
    path: str
    type: Optional[str] = None
    required: Optional[bool] = None
    enum: Optional[List[str]] = None
    doc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"path": self.path}
        if self.type is not None:
            data["type"] = self.type
        if self.required is not None:
            data["required"] = self.required
        if self.enum:
            data["enum"] = self.enum
        if self.doc:
            data["doc"] = self.doc
        return data


@dataclass
class ConsumableEntry:
    id: str
    category: str
    source_spec_key: str
    properties: List[ConsumableProperty] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        enum_index = {
            p.path: p.enum for p in self.properties if p.enum
        }  # quick lookup index for completion use-cases
        doc_urls: List[str] = []
        for p in self.properties:
            if p.doc:
                doc_urls.append(p.doc)
        return {
            "id": self.id,
            "category": self.category,
            "source_spec_key": self.source_spec_key,
            "properties": [p.to_dict() for p in self.properties],
            "enum_index": enum_index,
            "doc_urls": sorted(set(doc_urls)),
        }


class ConsumablesIndexer:
    """Builds an AI-oriented consumables catalog from compiled component specs.

    Public Methods:
        build_index(): Returns list of entry dicts (deterministic order)
    """

    def __init__(self, meta_prefix: str = "_"):
        self.meta_prefix = meta_prefix
        self.spec_library = SpecLibrary(meta_prefix=meta_prefix)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_index(self) -> List[Dict[str, Any]]:
        specs = self.spec_library.get_specs()
        entries: List[ConsumableEntry] = []
        for root_key, spec in specs.items():
            if "::" not in root_key:
                # Skip non-resource root keys (e.g., internal custom type definitions)
                continue
            category = root_key.replace("::", "/")
            entry = ConsumableEntry(
                id=root_key, category=category, source_spec_key=root_key
            )
            self._collect_properties(spec, prefix="", out=entry.properties)
            entries.append(entry)

        # Deterministic ordering by id
        entries.sort(key=lambda e: e.id)
        return [e.to_dict() for e in entries]

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _collect_properties(
        self, spec: Dict[str, Any], prefix: str, out: List[ConsumableProperty]
    ):  # noqa: C901
        # Recurse keys that are *not* meta
        for key in sorted(spec):
            if key.startswith(self.meta_prefix):
                continue
            sub_spec = spec[key]
            fq_path = f"{prefix}.{key}" if prefix else key

            if not isinstance(sub_spec, dict):
                continue

            # Determine property metadata if present
            meta = self._extract_meta(sub_spec)
            if meta:
                out.append(ConsumableProperty(path=fq_path, **meta))

            # Dive deeper
            self._collect_properties(sub_spec, fq_path, out)

            # List item spec: create synthetic path with [] for enumerating members
            list_item_spec_key = META_KEYS["list_item_spec"]
            if list_item_spec_key in sub_spec:
                item_spec = sub_spec[list_item_spec_key]
                if isinstance(item_spec, dict):
                    list_path = fq_path + "[]"
                    self._collect_properties(item_spec, list_path, out)

    def _extract_meta(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        type_key = META_KEYS["type"]
        if type_key in spec:
            meta["type"] = spec[type_key]
        req_key = META_KEYS["required"]
        if req_key in spec:
            raw = spec[req_key]
            if isinstance(raw, bool):
                meta["required"] = raw
            elif isinstance(raw, str):
                low = raw.lower()
                if low in {"true", "yes", "1"}:
                    meta["required"] = True
                elif low in {"false", "no", "0"}:
                    meta["required"] = False
        enum_key = META_KEYS["enum"]
        if enum_key in spec and isinstance(spec[enum_key], list):
            enum_vals: List[str] = [
                str(v) for v in spec[enum_key] if isinstance(v, (str, int))
            ]
            if enum_vals:
                meta["enum"] = enum_vals
        doc_key = META_KEYS["documentation"]
        if doc_key in spec:
            doc_val = spec[doc_key]
            if isinstance(doc_val, str):
                meta["doc"] = doc_val
            elif isinstance(doc_val, list) and doc_val:
                # only store first to keep payload light; rest are accessible via spec file if needed
                first = doc_val[0]
                if isinstance(first, str):
                    meta["doc"] = first
        return meta


def write_index_json(
    path: str | Path, index: List[Dict[str, Any]], pretty: bool = False
) -> None:
    import json

    p = Path(path)
    p.write_text(
        json.dumps(index, indent=2 if pretty else None, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote consumables index", details={"path": str(p), "entries": len(index)})


__all__ = ["ConsumablesIndexer", "write_index_json"]
