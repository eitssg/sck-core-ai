"""Action catalog indexer.

Discovers action classes defined in `core_execute.actionlib.actions` and
introspects their associated Pydantic spec models (ActionSpec subclasses)
to build an AI-friendly catalog: action id/kind, module path, spec fields
(name, type, required, default), and docstring summary.

Read-only: does not mutate core_execute. Falls back gracefully if
core_execute not installed.
"""

from typing import Any, Dict, List, Optional, get_type_hints

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
class ActionEntry:
    id: str
    module: str
    class_name: str
    summary: Optional[str]
    params: List[ActionParam] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "module": self.module,
            "class_name": self.class_name,
            "summary": self.summary,
            "params": [p.to_dict() for p in self.params],
        }


class ActionIndexer:
    ACTIONS_ROOT_PKG = "core_execute.actionlib.actions"

    def __init__(self):
        if BaseAction is None or ActionSpec is None:  # pragma: no cover
            raise RuntimeError("core_execute not available")

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
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if not BaseAction or not issubclass(obj, BaseAction):
                    continue
                if obj is BaseAction:  # skip base
                    continue
                spec_model = getattr(obj, "spec_model", None) or getattr(obj, "SpecModel", None)
                params: List[ActionParam] = []
                if spec_model and ActionSpec and issubclass(spec_model, ActionSpec):
                    params = self._extract_spec_params(spec_model)
                action_id = getattr(obj, "kind", None) or name.replace("Action", "")
                summary = (inspect.getdoc(obj) or "").split("\n")[0][:200]
                entries.append(
                    ActionEntry(
                        id=str(action_id),
                        module=mod,
                        class_name=name,
                        summary=summary,
                        params=params,
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

    def _extract_spec_params(self, spec_model) -> List[ActionParam]:
        params: List[ActionParam] = []
        try:
            hints = get_type_hints(spec_model)
        except Exception:
            hints = {}
        # Pydantic v2: model_fields holds FieldInfo
        for field_name, field_info in getattr(spec_model, "model_fields", {}).items():  # type: ignore[attr-defined]
            required = field_info.is_required()
            default = None if required else field_info.default
            hint = hints.get(field_name)
            type_name = getattr(hint, "__name__", str(hint)) if hint else "Any"
            params.append(
                ActionParam(
                    name=field_name,
                    type=type_name,
                    required=required,
                    default=default,
                )
            )
        return params


__all__ = ["ActionIndexer"]
