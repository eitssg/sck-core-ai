"""
Context manager for coordinating documentation and codebase search.

Provides unified interface for retrieving relevant context from both
documentation and codebase indexes for AI assistant queries.
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os

import core_logging as log

from .vector_store import VectorStore
from .documentation_indexer import DocumentationIndexer
from .codebase_indexer import CodebaseIndexer
from .consumables_indexer import ConsumablesIndexer
from .action_indexer import ActionIndexer


class ContextManager:
    """
    Unified context retrieval system for AI assistant queries.

    Coordinates searches across documentation and codebase indexes to provide
    comprehensive context for development assistance and code generation.
    """

    def __init__(self):
        """
        Initialize context manager.

        Args:
            vector_store: Optional existing vector store (creates new if None)
        """
        # Auto-detect paths via environment overrides or fallback heuristic
        # Assume we're in sck-core-ai, go up to workspace root
        env_workspace = os.environ.get("SCK_WORKSPACE_ROOT")
        workspace_root = env_workspace or os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

        env_docs_build = os.environ.get("SCK_DOCS_BUILD_DIR")
        build_directory = env_docs_build or os.path.join(workspace_root, "sck-core-docs", "build")

        self.build_directory = Path(build_directory)
        self.workspace_root = Path(workspace_root)

        # Initialize vector store (fixed, not injectable per design guidance)
        self.vector_store = VectorStore()

        # Initialize indexers
        self.doc_indexer = DocumentationIndexer(build_directory=str(self.build_directory), vector_store=self.vector_store)
        self.code_indexer = CodebaseIndexer(workspace_root=str(self.workspace_root), vector_store=self.vector_store)

        # Additional indexes managed by ContextManager
        self.consumables_indexer = ConsumablesIndexer()
        self.action_indexer = ActionIndexer()

        # Context retrieval strategies
        # Four-source weighting model; keep back-compat via fallback if keys not present
        self.retrieval_strategies = {
            # weights: documentation, codebase, consumables, actions
            "balanced": {"weights": {"documentation": 0.4, "codebase": 0.3, "consumables": 0.15, "actions": 0.15}},
            "documentation_focused": {"weights": {"documentation": 0.7, "codebase": 0.15, "consumables": 0.1, "actions": 0.05}},
            "code_focused": {"weights": {"documentation": 0.2, "codebase": 0.6, "consumables": 0.1, "actions": 0.1}},
            "documentation_only": {"weights": {"documentation": 1.0, "codebase": 0.0, "consumables": 0.0, "actions": 0.0}},
            "code_only": {"weights": {"documentation": 0.0, "codebase": 1.0, "consumables": 0.0, "actions": 0.0}},
        }
        # Logging per sck-core-ai rules
        log.info(f"Indexing system initialized with workspace: {workspace_root}")

    def initialize_indexes(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Initialize or rebuild all indexes.

        Args:
            force_rebuild: Whether to force rebuilding existing indexes

        Returns:
            Dictionary with initialization results
        """
        results: Dict[str, Any] = {"documentation": {}, "codebase": {}, "consumables": {}, "actions": {}, "errors": []}

        try:
            # Check if indexes already exist and are populated
            stats = self.get_system_stats()
            # Safely determine if (re)indexing is needed; default to True if stats unavailable
            doc_total = (stats.get("documentation") or {}).get("total_chunks") if isinstance(stats, dict) else 0
            code_total = (stats.get("codebase") or {}).get("total_elements") if isinstance(stats, dict) else 0
            vector_stats = self.vector_store.get_collection_stats()
            consumables_total = (vector_stats.get("consumables") or {}).get("document_count", 0)
            actions_total = (vector_stats.get("actions") or {}).get("document_count", 0)
            needs_doc_index = force_rebuild or (not isinstance(doc_total, int)) or doc_total == 0
            needs_code_index = force_rebuild or (not isinstance(code_total, int)) or code_total == 0
            needs_consumables_index = force_rebuild or not isinstance(consumables_total, int) or consumables_total == 0
            needs_actions_index = force_rebuild or not isinstance(actions_total, int) or actions_total == 0

            # Index documentation if needed
            if needs_doc_index:
                log.info("Indexing documentation...")
                # Prefer indexer-controlled clear behavior; default is incremental
                try:
                    doc_results = self.doc_indexer.index_all_documentation(clear_before=force_rebuild)  # type: ignore[arg-type]
                except TypeError:
                    # Backward compatibility if indexer doesn't support clear_before
                    doc_results = self.doc_indexer.index_all_documentation()
                results["documentation"] = doc_results
                if isinstance(doc_results, dict):
                    try:
                        log.info(f"Documentation indexing completed: {sum(doc_results.values())} chunks")
                    except Exception:
                        log.info("Documentation indexing completed")
            else:
                log.info("Documentation index already exists, skipping")
                results["documentation"] = {
                    "status": "skipped",
                    "reason": "already_exists",
                }

            # Index codebase if needed
            if needs_code_index:
                log.info("Indexing codebase...")
                code_results = self.code_indexer.index_all_projects()
                results["codebase"] = code_results
                if isinstance(code_results, dict):
                    try:
                        log.info(f"Codebase indexing completed: {sum(code_results.values())} elements")
                    except Exception:
                        log.info("Codebase indexing completed")
            else:
                log.info("Codebase index already exists, skipping")
                results["codebase"] = {"status": "skipped", "reason": "already_exists"}

            # Index consumables if needed (vectorized collection)
            if needs_consumables_index:
                log.info("Indexing consumables...")
                try:
                    cons_results = self.consumables_indexer.index_to_vector(
                        self.vector_store, collection_name="consumables", clear_before=force_rebuild
                    )
                except TypeError:
                    cons_results = self.consumables_indexer.index_to_vector(self.vector_store, collection_name="consumables")
                results["consumables"] = cons_results
                try:
                    log.info(f"Consumables indexing completed: {cons_results.get('upserted', 0)} entries")
                except Exception:
                    log.info("Consumables indexing completed")
            else:
                log.info("Consumables index already exists, skipping")
                results["consumables"] = {"status": "skipped", "reason": "already_exists"}

            # Index actions if needed (vectorized collection)
            if needs_actions_index:
                log.info("Indexing actions...")
                try:
                    actions = self.action_indexer.build_index()
                    documents, metadatas, ids = self._actions_to_vector_documents(actions)
                    if force_rebuild:
                        try:
                            self.vector_store.clear_collection("actions")
                        except Exception:
                            pass
                    ok = self.vector_store.add_documents(
                        collection_name="actions", documents=documents, metadatas=metadatas, ids=ids
                    )
                    results["actions"] = {"status": "success", "upserted": len(documents), "ok": bool(ok)}
                    log.info(f"Actions indexing completed: {len(documents)} entries")
                except Exception as e:
                    log.error(f"Error indexing actions: {e}")
                    results["actions"] = {"status": "error", "message": str(e)}
            else:
                log.info("Actions index already exists, skipping")
                results["actions"] = {"status": "skipped", "reason": "already_exists"}

        except Exception as e:
            error_obj = {"message": f"Error during index initialization: {e}", "code": "init_error", "source": "context_manager"}
            log.error(error_obj["message"])
            results.setdefault("errors", []).append(error_obj)

        return results

    def get_context_for_query(
        self,
        query: str,
        strategy: str = "balanced",
        max_results: int = 10,
        include_metadata: bool = True,
        score_mode: str = "auto",
        fused: bool = True,
        token_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get relevant context for a query using specified retrieval strategy.

        Args:
            query: Search query
            strategy: Retrieval strategy (balanced, documentation_focused, etc.)
            max_results: Maximum total results to return
            include_metadata: Whether to include detailed metadata

        Returns:
            Dictionary with context results from all sources
        """
        if strategy not in self.retrieval_strategies:
            log.warning(f"Unknown strategy '{strategy}', using 'balanced'")
            strategy = "balanced"

        strategy_config = self.retrieval_strategies[strategy]

        # Calculate per-source allocations
        weights = strategy_config.get("weights")
        if weights:
            allocations = self._allocate_counts_multi(max_results, weights)
        else:
            # Backward compatibility with older two-weight strategy
            d, c = self._allocate_counts(
                max_results, strategy_config.get("doc_weight", 0.0), strategy_config.get("code_weight", 0.0)
            )
            allocations = {"documentation": d, "codebase": c, "consumables": 0, "actions": 0}

        context = {
            "query": query,
            "strategy": strategy,
            "documentation": [],
            "codebase": [],
            "consumables": [],
            "actions": [],
            "items": [],
            "summary": {
                "total_results": 0,
                "doc_results": 0,
                "code_results": 0,
                "consumables_results": 0,
                "actions_results": 0,
                "allocation": allocations,
            },
        }

        try:
            # Search documentation if requested
            if allocations.get("documentation", 0) > 0:
                doc_results = self.doc_indexer.search_documentation(query=query, n_results=allocations["documentation"])
                formatted_docs = self._format_results(
                    doc_results, include_metadata, source_override="documentation", score_mode=score_mode
                )
                context["documentation"] = formatted_docs
                context["summary"]["doc_results"] = len(formatted_docs)

            # Search codebase if requested
            if allocations.get("codebase", 0) > 0:
                code_results = self.code_indexer.search_codebase(query=query, n_results=allocations["codebase"])
                formatted_code = self._format_results(
                    code_results, include_metadata, source_override="codebase", score_mode=score_mode
                )
                context["codebase"] = formatted_code
                context["summary"]["code_results"] = len(formatted_code)

            # Search consumables index
            if allocations.get("consumables", 0) > 0:
                cons_results = self.consumables_indexer.search_consumables(
                    self.vector_store, query=query, n_results=allocations["consumables"], collection_name="consumables"
                )
                formatted_cons = self._format_results(
                    cons_results, include_metadata, source_override="consumables", score_mode=score_mode
                )
                context["consumables"] = formatted_cons
                context["summary"]["consumables_results"] = len(formatted_cons)

            # Search actions index
            if allocations.get("actions", 0) > 0:
                act_results = self.vector_store.search_documents(
                    collection_name="actions", query=query, n_results=allocations["actions"], where=None
                )
                formatted_act = self._format_results(
                    act_results, include_metadata, source_override="actions", score_mode=score_mode
                )
                context["actions"] = formatted_act
                context["summary"]["actions_results"] = len(formatted_act)

            # If we underfilled, try to top-up from highest-weight sources
            total_now = (
                context["summary"]["doc_results"]
                + context["summary"]["code_results"]
                + context["summary"].get("consumables_results", 0)
                + context["summary"].get("actions_results", 0)
            )
            shortfall = max(0, max_results - total_now)
            if shortfall > 0:
                weights_for_topup = strategy_config.get("weights") or {
                    "documentation": strategy_config.get("doc_weight", 0),
                    "codebase": strategy_config.get("code_weight", 0),
                }
                ordered = sorted(weights_for_topup.items(), key=lambda kv: kv[1], reverse=True)
                for src, _w in ordered:
                    if shortfall <= 0:
                        break
                    if src == "documentation" and allocations.get("documentation", 0) > 0:
                        more_docs = self.doc_indexer.search_documentation(
                            query=query, n_results=allocations["documentation"] + shortfall
                        )
                        formatted_more = self._format_results(
                            more_docs, include_metadata, source_override="documentation", score_mode=score_mode
                        )
                        new_items = formatted_more[context["summary"]["doc_results"] :]
                        context["documentation"].extend(new_items)
                        added = len(new_items)
                        context["summary"]["doc_results"] += added
                        shortfall -= added
                    elif src == "codebase" and allocations.get("codebase", 0) > 0:
                        more_code = self.code_indexer.search_codebase(query=query, n_results=allocations["codebase"] + shortfall)
                        formatted_more = self._format_results(
                            more_code, include_metadata, source_override="codebase", score_mode=score_mode
                        )
                        new_items = formatted_more[context["summary"]["code_results"] :]
                        context["codebase"].extend(new_items)
                        added = len(new_items)
                        context["summary"]["code_results"] += added
                        shortfall -= added
                    elif src == "consumables" and allocations.get("consumables", 0) > 0:
                        more_cons = self.consumables_indexer.search_consumables(
                            self.vector_store, query=query, n_results=allocations["consumables"] + shortfall
                        )
                        formatted_more = self._format_results(
                            more_cons, include_metadata, source_override="consumables", score_mode=score_mode
                        )
                        new_items = formatted_more[context["summary"].get("consumables_results", 0) :]
                        context["consumables"].extend(new_items)
                        added = len(new_items)
                        context["summary"]["consumables_results"] = context["summary"].get("consumables_results", 0) + added
                        shortfall -= added
                    elif src == "actions" and allocations.get("actions", 0) > 0:
                        more_act = self.vector_store.search_documents(
                            collection_name="actions", query=query, n_results=allocations["actions"] + shortfall, where=None
                        )
                        formatted_more = self._format_results(
                            more_act, include_metadata, source_override="actions", score_mode=score_mode
                        )
                        new_items = formatted_more[context["summary"].get("actions_results", 0) :]
                        context["actions"].extend(new_items)
                        added = len(new_items)
                        context["summary"]["actions_results"] = context["summary"].get("actions_results", 0) + added
                        shortfall -= added

            # Build fused items list
            if fused:
                fused_items = self._fuse_and_rank(
                    context.get("documentation", []),
                    context.get("codebase", []),
                    context.get("consumables", []),
                    context.get("actions", []),
                )
                # Apply token budget if requested
                if token_budget is not None and token_budget > 0:
                    fused_items, tokens_used = self._apply_token_budget(fused_items, token_budget)
                    context["summary"]["tokens_estimate"] = tokens_used
                context["items"] = fused_items[:max_results]

            context["summary"]["total_results"] = (
                context["summary"]["doc_results"]
                + context["summary"]["code_results"]
                + context["summary"].get("consumables_results", 0)
                + context["summary"].get("actions_results", 0)
            )

        except Exception as e:
            log.error(f"Error retrieving context for query '{query}': {e}")
            context["error"] = {"message": str(e), "code": "context_error", "source": "context_manager"}

        return context

    def get_project_context(self, project_name: str, query: Optional[str] = None, max_results: int = 15) -> Dict[str, Any]:
        """
        Get context specific to a project.

        Args:
            project_name: Name of the SCK project (e.g., sck-core-api)
            query: Optional search query within the project
            max_results: Maximum results to return

        Returns:
            Dictionary with project-specific context
        """
        context = {
            "project": project_name,
            "query": query,
            "documentation": [],
            "codebase": [],
        }

        try:
            # Search codebase for project-specific content
            code_results = self.code_indexer.search_codebase(
                query=query or f"project {project_name}",
                project=project_name,
                n_results=max_results,
            )
            context["codebase"] = self._format_results(code_results, include_metadata=True, source_override="codebase")

            # If there's a specific query, also search documentation
            if query:
                doc_results = self.doc_indexer.search_documentation(query=f"{project_name} {query}", n_results=max_results // 2)
                context["documentation"] = self._format_results(doc_results, include_metadata=True, source_override="documentation")

        except Exception as e:
            log.error(f"Error retrieving project context for {project_name}: {e}")
            context["error"] = {"message": str(e), "code": "project_context_error", "source": "context_manager"}

        return context

    def get_architectural_context(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """
        Get architectural and design pattern context.

        Args:
            query: Architecture-related query
            max_results: Maximum results to return

        Returns:
            Dictionary with architectural context
        """
        # Architecture-focused search terms
        arch_terms = [
            "architecture",
            "design",
            "pattern",
            "framework",
            "structure",
            "lambda",
            "s3",
            "api",
            "authentication",
            "oauth",
            "envelope",
        ]

        # Enhance query with architectural terms
        enhanced_query = f"{query} " + " ".join(arch_terms[:3])

        return self.get_context_for_query(
            query=enhanced_query,
            strategy="documentation_focused",  # Architecture docs are primary
            max_results=max_results,
            include_metadata=True,
        )

    def search_by_element_type(
        self,
        element_type: str,
        query: Optional[str] = None,
        project: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for specific code element types.

        Args:
            element_type: Type of element (class, function, method, module)
            query: Optional search query
            project: Optional project filter
            max_results: Maximum results to return

        Returns:
            List of matching code elements
        """
        search_query = query or element_type

        try:
            results = self.code_indexer.search_codebase(
                query=search_query,
                project=project,
                element_type=element_type,
                n_results=max_results,
            )
            return self._format_results(results, include_metadata=True, source_override="codebase")

        except Exception as e:
            log.error(f"Error searching for {element_type} elements: {e}")
            return []

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with system-wide statistics
        """
        try:
            doc_stats = self.doc_indexer.get_documentation_stats()
            code_stats = self.code_indexer.get_codebase_stats()
            vector_stats = self.vector_store.get_collection_stats()
            consumable_stats = self.consumables_indexer.get_stats()
            actions_stats = vector_stats.get("actions", {})

            return {
                "documentation": doc_stats,
                "codebase": code_stats,
                "vector_store": vector_stats,
                "consumables": consumable_stats,
                "actions": {"document_count": actions_stats.get("document_count", 0)},
                "build_directory": str(self.build_directory),
                "workspace_root": str(self.workspace_root),
                "available_strategies": list(self.retrieval_strategies.keys()),
            }

        except Exception as e:
            log.error(f"Error getting system stats: {e}")
            return {"error": {"message": str(e), "code": "stats_error", "source": "context_manager"}}

    def rebuild_indexes(self) -> Dict[str, Any]:
        """
        Force rebuild of all indexes.

        Returns:
            Dictionary with rebuild results
        """
        log.info("Starting full index rebuild...")
        return self.initialize_indexes(force_rebuild=True)

    def _format_results(
        self,
        results: List[Dict[str, Any]],
        include_metadata: bool,
        source_override: Optional[str] = None,
        score_mode: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Format search results for consistent output.

        Args:
            results: Raw search results
            include_metadata: Whether to include full metadata

        Returns:
            Formatted results list
        """
        formatted: List[Dict[str, Any]] = []

        for result in results:
            distance = result.get("distance", 1.0)
            similarity = self._distance_to_similarity(distance, score_mode)
            # Clamp similarity to [0,1]
            try:
                if similarity < 0:
                    similarity = 0.0
                elif similarity > 1:
                    similarity = 1.0
            except Exception:
                similarity = 0.0

            formatted_result = {
                "content": result.get("document", ""),
                "relevance_score": similarity,
                "id": result.get("id", ""),
            }

            if include_metadata:
                metadata = result.get("metadata", {})
                formatted_result.update(
                    {
                        "source": source_override or metadata.get("source", "unknown"),
                        "type": metadata.get("element_type", metadata.get("section", "unknown")),
                        "location": metadata.get("file_path", metadata.get("page_url", "")),
                        "project": metadata.get("project", metadata.get("section", "")),
                        "line_number": metadata.get("line_number"),
                        "metadata": metadata,
                    }
                )

            formatted.append(formatted_result)

        return formatted

    # ---------- Helper methods ----------
    def _allocate_counts(self, max_results: int, doc_weight: float, code_weight: float) -> Tuple[int, int]:
        total_weight = max(doc_weight + code_weight, 0.0)
        if total_weight == 0:
            return 0, 0

        # Base allocations
        doc_exact = max_results * (doc_weight / total_weight)
        code_exact = max_results * (code_weight / total_weight)
        doc_count = int(doc_exact)
        code_count = int(code_exact)

        # Ensure at least one if weight > 0
        if doc_weight > 0 and doc_count == 0:
            doc_count = 1
        if code_weight > 0 and code_count == 0:
            code_count = 1

        # Distribute leftovers by largest remainder
        allocated = doc_count + code_count
        leftovers = max(0, max_results - allocated)
        remainders = [
            (doc_exact - int(doc_exact), "doc"),
            (code_exact - int(code_exact), "code"),
        ]
        remainders.sort(reverse=True)
        for _ in range(leftovers):
            if remainders and remainders[0][0] >= (remainders[1][0] if len(remainders) > 1 else 0):
                doc_count += 1
            else:
                code_count += 1
        return doc_count, code_count

    def _distance_to_similarity(self, distance: float, mode: str = "auto") -> float:
        """Convert distance to a similarity-like score in [0,1].

        For cosine distance, similarity ~= 1 - distance.
        For unknown modes, default to 1 - distance and clamp.
        """
        try:
            sim = 1.0 - float(distance)
        except Exception:
            sim = 0.0
        return sim

    def _fuse_and_rank(self, *groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fuse multiple result lists, normalize per-source scores, dedupe, and sort by score desc."""

        def normalize(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not items:
                return items
            scores = [i.get("relevance_score", 0.0) for i in items]
            min_s, max_s = min(scores), max(scores)
            if max_s > min_s:
                for i in items:
                    rs = i.get("relevance_score", 0.0)
                    i["relevance_score"] = (rs - min_s) / (max_s - min_s)
            # else keep as-is
            return items

        combined: List[Dict[str, Any]] = []
        for g in groups:
            combined.extend(normalize(list(g)))

        # Deduplicate by id preserving first occurrence
        seen: set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for item in combined:
            _id = item.get("id")
            if _id and _id in seen:
                continue
            if _id:
                seen.add(_id)
            deduped.append(item)

        deduped.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        return deduped

    def _apply_token_budget(self, items: List[Dict[str, Any]], token_budget: int) -> Tuple[List[Dict[str, Any]], int]:
        """Trim contents to fit within an approximate token budget (4 chars per token heuristic)."""
        remaining = max(token_budget, 0)
        kept: List[Dict[str, Any]] = []
        used = 0
        for it in items:
            if remaining <= 0:
                break
            content = it.get("content", "")
            est_tokens = max(int(len(content) / 4), 1)
            if est_tokens <= remaining:
                kept.append(it)
                remaining -= est_tokens
                used += est_tokens
            else:
                # Trim content to remaining tokens
                max_chars = remaining * 4
                it_trim = dict(it)
                it_trim["content"] = content[:max_chars]
                kept.append(it_trim)
                used += remaining
                remaining = 0
        return kept, used

    def _allocate_counts_multi(self, max_results: int, weights: Dict[str, float]) -> Dict[str, int]:
        """Allocate result counts across multiple sources using largest remainder method."""
        total = sum(w for w in weights.values() if w > 0)
        if total <= 0 or max_results <= 0:
            return {k: 0 for k in weights.keys()}
        exacts = {k: max_results * (w / total) for k, w in weights.items()}
        counts = {k: int(v) for k, v in exacts.items()}
        # Ensure at least 1 for any positive weight
        for k, w in weights.items():
            if w > 0 and counts.get(k, 0) == 0:
                counts[k] = 1
        allocated = sum(counts.values())
        leftovers = max(0, max_results - allocated)
        rema = sorted(((exacts[k] - int(exacts[k]), k) for k in weights.keys()), reverse=True)
        i = 0
        while leftovers > 0 and rema:
            _, key = rema[i % len(rema)]
            counts[key] = counts.get(key, 0) + 1
            leftovers -= 1
            i += 1
        return counts

    def _actions_to_vector_documents(self, actions: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Transform action index entries into vector-store documents, metadata, and ids."""
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        for a in actions:
            aid = str(a.get("id"))
            module = a.get("module")
            summary = a.get("summary") or ""
            performs = a.get("performs") or ""
            # Build a concise document string
            lines = [f"Action: {aid}"]
            if summary:
                lines.append(f"Summary: {summary}")
            if performs and performs != summary:
                lines.append(f"Performs: {performs}")
            # Resource fields
            rf = a.get("resource_fields", []) or []
            if rf:
                rf_fmt = ", ".join(
                    [f"{f.get('name')}: {f.get('type')} ({'required' if f.get('required') else 'optional'})" for f in rf][:12]
                )
                lines.append(f"Resource fields: {rf_fmt}")
            # Spec params
            sp = a.get("spec_params", []) or []
            if sp:
                sp_fmt = ", ".join(
                    [f"{p.get('name')}: {p.get('type')} ({'required' if p.get('required') else 'optional'})" for p in sp][:12]
                )
                lines.append(f"Spec params: {sp_fmt}")
            doc = "\n".join(lines)
            documents.append(doc)
            metadatas.append(
                {
                    "source": "actions",
                    "element_type": "action",
                    "project": "sck-core-execute",
                    "module": module,
                    "action_id": aid,
                }
            )
            ids.append(f"action:{aid}")
        return documents, metadatas, ids
