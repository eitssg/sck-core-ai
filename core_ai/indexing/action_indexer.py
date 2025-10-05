"""Action catalog indexer.

Discovers ActionResource models and their paired Action executors under
`core_execute.actionlib.actions`. For each action, builds an AI-friendly
catalog entry that includes:

- id (action name), module path, resource class, action class
- brief summary of what the action performs (from docstrings)
- resource schema (fields of the ActionResource model)
- spec params (fields of the ActionSpec model that configures execution)

This index is designed for MCP/LLM context so tools can present the
available actions, their purpose, and required/optional parameters.

Read-only: does not mutate core_execute.
"""

from typing import Any, Dict, List, Optional, Tuple, get_type_hints

from dataclasses import dataclass, field
import inspect
import pkgutil
import importlib

import core_logging as log

from core_execute.actionlib.action import BaseAction

from core_framework.models import ActionResource, ActionMetadata, ActionSpec


@dataclass
class ActionParam:
    name: str
    type: str
    required: bool
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
        }
        if self.default is not None:
            d["default"] = self.default
        return d


@dataclass
class ModelFieldDesc:
    name: str
    type: str
    required: bool
    default: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "type": self.type, "required": self.required}
        if self.default is not None:
            d["default"] = self.default
        return d


@dataclass
class ActionEntry:
    id: str
    module: str
    resource_class: str
    action_class: Optional[str]
    kind: Optional[str]
    summary: Optional[str]
    performs: Optional[str]
    resource_fields: List[ModelFieldDesc] = field(default_factory=list)
    spec_params: List[ModelFieldDesc] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "module": self.module,
            "resource_class": self.resource_class,
            "action_class": self.action_class,
            "kind": self.kind,
            "summary": self.summary,
            "performs": self.performs,
            "resource_fields": [f.to_dict() for f in self.resource_fields],
            "spec_params": [p.to_dict() for p in self.spec_params],
        }


class ActionIndexer:
    ACTIONS_ROOT_PKG = "core_execute.actionlib.actions"

    def __init__(self):
        # BaseAction, ActionResource, ActionSpec are expected to be available.
        if BaseAction is None or ActionResource is None or ActionSpec is None:  # pragma: no cover
            raise RuntimeError("core_execute/core_framework models not available")

    def build_index(self) -> List[Dict[str, Any]]:
        entries: List[ActionEntry] = []
        try:
            pkg = importlib.import_module(self.ACTIONS_ROOT_PKG)
        except Exception as e:  # pragma: no cover
            log.warning("Unable to import actions root", error=str(e))
            return []

        for mod in self._iter_modules(pkg):
            try:
                module = importlib.import_module(mod)
            except Exception:
                continue

            # Collect classes in this module
            classes = {name: obj for name, obj in inspect.getmembers(module, inspect.isclass)}

            # Identify ActionResource subclasses ending with 'ActionResource'
            for name, obj in classes.items():
                if not issubclass_safe(obj, ActionResource):
                    continue
                if name == ActionResource.__name__:
                    continue
                if not name.endswith("ActionResource"):
                    continue

                action_name = name[: -len("ActionResource")] or name

                # Try to find matching Action class
                action_cls_name = f"{action_name}Action"
                action_cls = classes.get(action_cls_name)
                if action_cls is not None and not issubclass_safe(action_cls, BaseAction):
                    action_cls = None

                # Determine kind from Action class or resource attribute
                kind = None
                if action_cls is not None:
                    kind = getattr(action_cls, "kind", None)
                if kind is None:
                    kind = getattr(obj, "kind", None)

                # Determine Spec model via resource -> action -> name heuristic
                spec_model = getattr(obj, "SpecModel", None) or getattr(obj, "spec_model", None)
                if spec_model is None and action_cls is not None:
                    spec_model = getattr(action_cls, "SpecModel", None) or getattr(action_cls, "spec_model", None)
                if spec_model is None:
                    # Fallback: find Any class named <ActionName>ActionSpec
                    candidate_name = f"{action_name}ActionSpec"
                    cand = classes.get(candidate_name)
                    if cand and issubclass_safe(cand, ActionSpec):
                        spec_model = cand

                # Extract schemas
                resource_fields = self._extract_model_fields(obj)
                spec_params = self._extract_model_fields(spec_model) if spec_model else []

                # Compose summary from docstrings (resource preferred, else action)
                rdoc_str = inspect.getdoc(obj) or ""
                adoc_str = (inspect.getdoc(action_cls) or "") if action_cls is not None else ""
                if rdoc_str or adoc_str:
                    summary = (rdoc_str or adoc_str).split("\n")[0][:200]
                    performs = (adoc_str or rdoc_str).split("\n\n")[0][:400]
                else:
                    summary = None
                    performs = None

                entries.append(
                    ActionEntry(
                        id=str(kind or action_name),
                        module=mod,
                        resource_class=name,
                        action_class=action_cls.__name__ if action_cls is not None else None,
                        kind=str(kind) if kind is not None else None,
                        summary=summary,
                        performs=performs,
                        resource_fields=resource_fields,
                        spec_params=spec_params,
                    )
                )

        entries.sort(key=lambda e: e.id)
        return [e.to_dict() for e in entries]

    def _iter_modules(self, pkg) -> List[str]:
        mods: List[str] = []
        if not hasattr(pkg, "__path__"):
            return mods
        for module_finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if name.endswith(".__init__"):
                continue
            mods.append(name)
        return mods

    def _extract_model_fields(self, model) -> List[ModelFieldDesc]:
        fields: List[ModelFieldDesc] = []
        if model is None:
            return fields
        try:
            hints = get_type_hints(model)
        except Exception:
            hints = {}
        for field_name, field_info in getattr(model, "model_fields", {}).items():  # pydantic v2
            required = field_info.is_required()
            default = None if required else field_info.default
            hint = hints.get(field_name)
            type_name = (
                getattr(hint, "__name__", None) or str(hint)
                if hint
                else getattr(field_info.annotation, "__name__", None) or str(field_info.annotation)
            )
            fields.append(
                ModelFieldDesc(
                    name=field_name,
                    type=type_name,
                    required=required,
                    default=default,
                )
            )
        return fields


def issubclass_safe(cls, parent) -> bool:
    try:
        return isinstance(cls, type) and isinstance(parent, type) and issubclass(cls, parent)
    except Exception:
        return False


__all__ = ["ActionIndexer"]
