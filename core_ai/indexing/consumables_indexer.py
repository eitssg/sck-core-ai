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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
import re

import core_logging as log
import core_framework as util

from core_component.validator.spec_library import SpecLibrary

logger = log


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
        enum_index = {p.path: p.enum for p in self.properties if p.enum}  # quick lookup index for completion use-cases
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

        # Resolve consumables root directory (env override or inferred)
        env_root = os.environ.get("SCK_CONSUMABLES_DIR")
        if env_root:
            self.consumables_root = Path(env_root)
        else:
            # From this file: .../sck-core-ai/core_ai/indexing/consumables_indexer.py
            # parents[4] → monorepo root; then sibling sck-core-component
            here = Path(__file__).resolve()
            try:
                workspace_root = here.parents[4]
            except IndexError:
                workspace_root = here.parents[3]
            self.consumables_root = workspace_root / "sck-core-component" / "core_component" / "compiler" / "consumables"

        # Stats counters
        self._stats: Dict[str, int] = {
            "files_scanned": 0,
            "spec_entries": 0,
            "template_entries": 0,
            "action_entries": 0,
            "properties": 0,
            "errors": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_index(self) -> List[Dict[str, Any]]:
        """Build combined index for specs, templates, and actions.

        Returns:
            Deterministically ordered list of entry dicts.
        """
        output: List[Dict[str, Any]] = []

        # 1) Build spec entries from SpecLibrary
        spec_entries: List[ConsumableEntry] = []
        try:
            specs = self.spec_library.get_specs()
            for root_key, spec in specs.items():
                if "::" not in root_key:
                    continue
                category = root_key.replace("::", "/")
                entry = ConsumableEntry(id=root_key, category=category, source_spec_key=root_key)
                self._collect_properties(spec, prefix="", out=entry.properties)
                spec_entries.append(entry)
        except Exception as e:  # pragma: no cover
            logger.error("Failed to load compiled specs", details={"error": str(e)})
            self._stats["errors"] += 1

        # Optional: map spec id → file path (for traceability)
        spec_file_map = self._map_spec_files()

        # 2) Scan templates and actions from filesystem
        template_entries = self._scan_templates()
        action_entries = self._scan_actions()

        # 3) Assemble unified output
        for e in spec_entries:
            d = e.to_dict()
            d["kind"] = "spec"
            d["file_path"] = spec_file_map.get(e.id)
            output.append(d)
            self._stats["spec_entries"] += 1
            self._stats["properties"] += len(e.properties)

        for t in template_entries:
            output.append(t)
            self._stats["template_entries"] += 1
            self._stats["properties"] += len(t.get("properties", []))

        for a in action_entries:
            output.append(a)
            self._stats["action_entries"] += 1

        output.sort(key=lambda d: str(d.get("id", "")))
        return output

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _collect_properties(self, spec: Dict[str, Any], prefix: str, out: List[ConsumableProperty]) -> None:  # noqa: C901
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
            enum_vals: List[str] = [str(v) for v in spec[enum_key] if isinstance(v, (str, int))]
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

    # ---------------- Filesystem scanning helpers ----------------
    def _discover_consumable_roots(self) -> List[Path]:
        roots: List[Path] = []
        if not self.consumables_root.exists():
            logger.warning("Consumables root not found", details={"path": str(self.consumables_root)})
            return roots
        for dirpath, dirnames, _ in os.walk(self.consumables_root):
            p = Path(dirpath)
            if (p / "specs").is_dir() or (p / "files").is_dir() or (p / "actions").is_dir():
                roots.append(p)
        # Deduplicate
        unique = []
        seen = set()
        for r in sorted(roots):
            s = str(r)
            if s not in seen:
                seen.add(s)
                unique.append(r)
        return unique

    def _map_spec_files(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for root in self._discover_consumable_roots():
            specs_dir = root / "specs"
            if not specs_dir.is_dir():
                continue
            for f in specs_dir.rglob("*.y*ml"):
                self._stats["files_scanned"] += 1
                try:
                    data = util.load_yaml_file(str(f))
                except Exception:
                    self._stats["errors"] += 1
                    continue
                if isinstance(data, dict):
                    for k in data.keys():
                        if isinstance(k, str) and "::" in k and k not in mapping:
                            mapping[k] = str(f)
        return mapping

    def _scan_templates(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        var_pattern = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
        inc_pattern = re.compile(r"""\{%\s*(?:include|import|from)\s+['"][^'"]+['"][^%]*%\}""")

        for root in self._discover_consumable_roots():
            files_dir = root / "files"
            if not files_dir.is_dir():
                continue
            for f in files_dir.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in {".yml", ".yaml", ".j2", ".tmpl"}:
                    continue
                self._stats["files_scanned"] += 1
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    self._stats["errors"] += 1
                    continue
                vars_found = sorted({v.strip() for v in var_pattern.findall(text) if v.strip()})
                includes = inc_pattern.findall(text)

                consumable_slug = str(root.relative_to(self.consumables_root)).replace("\\", "/")
                rel_path = str(f.relative_to(self.consumables_root)).replace("\\", "/")
                entry: Dict[str, Any] = {
                    "id": f"template:{rel_path}",
                    "kind": "template",
                    "category": f"files/{consumable_slug}",
                    "source_spec_key": self._infer_spec_from_folder(root),
                    "file_path": str(f),
                    "properties": [{"path": v, "type": "jinja_var"} for v in vars_found],
                    "metadata": {"includes": includes},
                }
                entries.append(entry)
        return entries

    def _scan_actions(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for root in self._discover_consumable_roots():
            actions_dir = root / "actions"
            if not actions_dir.is_dir():
                continue
            tasks: Dict[str, Dict[str, Any]] = {}
            for f in actions_dir.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in {".yml", ".yaml", ".actions"}:
                    continue
                self._stats["files_scanned"] += 1
                parsed = self._safe_load_yaml(f)
                if not isinstance(parsed, dict):
                    continue

                # Top-level tasks by key
                for k in list(parsed.keys()):
                    kl = str(k).lower()
                    if kl in {"deploy", "release", "teardown"}:
                        tasks[kl] = self._normalize_task(parsed[k])

                # Fallback: infer from filename
                name_l = f.stem.lower()
                for t in ("deploy", "release", "teardown"):
                    if t in name_l and t not in tasks:
                        tasks[t] = self._normalize_task(parsed)

            if tasks:
                consumable_slug = str(root.relative_to(self.consumables_root)).replace("\\", "/")
                entry: Dict[str, Any] = {
                    "id": f"action:{consumable_slug}",
                    "kind": "action",
                    "category": f"actions/{consumable_slug}",
                    "source_spec_key": self._infer_spec_from_folder(root),
                    "file_path": str(actions_dir),
                    "actions": dict(sorted(tasks.items())),
                }
                entries.append(entry)
        return entries

    def _infer_spec_from_folder(self, root: Path) -> Optional[str]:
        specs_dir = root / "specs"
        if not specs_dir.is_dir():
            return None
        for f in specs_dir.rglob("*.y*ml"):
            data = self._safe_load_yaml(f)
            if isinstance(data, dict):
                for k in data.keys():
                    if isinstance(k, str) and "::" in k:
                        return k
        return None

    def _safe_load_yaml(self, path: Path) -> Any:
        try:
            return util.load_yaml_file(str(path))
        except Exception:
            self._stats["errors"] += 1
            return None

    def _normalize_task(self, task: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {"raw": task}
        if isinstance(task, dict):
            templates: List[str] = []
            for c in ("templates", "files", "file", "template"):
                v = task.get(c)
                if isinstance(v, str):
                    templates.append(v)
                elif isinstance(v, list):
                    templates.extend([str(x) for x in v])
            if templates:
                result["templates"] = sorted(set(templates))

            params = task.get("parameters") or task.get("params")
            if isinstance(params, dict):
                result["params"] = sorted(params.keys())

            stack = task.get("stack") or task.get("stacks") or task.get("stack_name")
            if isinstance(stack, str):
                result["stacks"] = [stack]
            elif isinstance(stack, list):
                result["stacks"] = [str(s) for s in stack]

            deps = task.get("depends_on") or task.get("requires")
            if isinstance(deps, list):
                result["depends_on"] = [str(d) for d in deps]

            note = task.get("description") or task.get("summary") or task.get("notes")
            if isinstance(note, str):
                result["notes"] = note
        return result

    # ---------------- Stats & VectorStore integration ---------------
    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def to_vector_documents(self, index: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for item in index:
            kind = item.get("kind")
            entry_id = str(item.get("id"))
            file_path = item.get("file_path")
            project = "sck-core-component"

            if kind == "spec":
                for prop in item.get("properties", []):
                    path = prop.get("path")
                    t = prop.get("type")
                    req = prop.get("required")
                    enum_vals = prop.get("enum")
                    line = f"{entry_id} {path} type={t} required={req}"
                    if enum_vals:
                        line += f" enum={enum_vals}"
                    docs.append(line)
                    metas.append(
                        {
                            "source": "consumables",
                            "kind": kind,
                            "project": project,
                            "file_path": file_path,
                            "element_type": "consumable_property",
                            "entry_id": entry_id,
                            "property_path": path,
                        }
                    )
                    ids.append(f"spec:{entry_id}#{path}")
            elif kind == "template":
                for prop in item.get("properties", []):
                    path = prop.get("path")
                    line = f"template var: {path} in {entry_id}"
                    docs.append(line)
                    metas.append(
                        {
                            "source": "consumables",
                            "kind": kind,
                            "project": project,
                            "file_path": file_path,
                            "element_type": "template_variable",
                            "entry_id": entry_id,
                            "property_path": path,
                        }
                    )
                    ids.append(f"template:{entry_id}#var:{path}")
            elif kind == "action":
                actions = item.get("actions", {}) or {}
                for task, details in actions.items():
                    line = f"action {task} for {entry_id}"
                    if isinstance(details, dict):
                        if "templates" in details:
                            line += f" templates={details['templates']}"
                        if "params" in details:
                            line += f" params={details['params']}"
                    docs.append(line)
                    metas.append(
                        {
                            "source": "consumables",
                            "kind": kind,
                            "project": project,
                            "file_path": file_path,
                            "element_type": "action",
                            "entry_id": entry_id,
                            "task": task,
                        }
                    )
                    ids.append(f"action:{entry_id}#{task}")
        return docs, metas, ids

    def index_to_vector(
        self,
        vector_store,
        index: Optional[List[Dict[str, Any]]] = None,
        collection_name: str = "consumables",
        clear_before: bool = False,
    ) -> Dict[str, Any]:
        if index is None:
            index = self.build_index()
        docs, metas, ids = self.to_vector_documents(index)
        if clear_before:
            try:
                vector_store.clear_collection(collection_name)
            except Exception:
                pass
        result = vector_store.upsert_documents(documents=docs, metadatas=metas, ids=ids, collection_name=collection_name)
        return {"status": "success", "upserted": len(docs), "collection": collection_name, "result": result}

    def search_consumables(
        self, vector_store, query: str, n_results: int = 10, collection_name: str = "consumables"
    ) -> List[Dict[str, Any]]:
        return vector_store.search_documents(query=query, n_results=n_results, collection_name=collection_name)


def write_index_json(path: str | Path, index: List[Dict[str, Any]], pretty: bool = False) -> None:
    import json

    p = Path(path)
    p.write_text(
        json.dumps(index, indent=2 if pretty else None, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote consumables index", details={"path": str(p), "entries": len(index)})


__all__ = ["ConsumablesIndexer", "write_index_json"]
