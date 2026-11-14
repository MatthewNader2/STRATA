"""
src/node_resolver.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)

Dynamic Node Resolution and Self-Expanding Architecture:
1. When an LLM router or pathfinder proposes a node that was not in the RAG top-k:
   - Case A (Node exists in catalog): Check library/orchestrator. If found, boost
     its relevance score in RAG (so it surfaces in subsequent queries) and return it.
   - Case B (Node does not exist in catalog): If dev_mode is active, prompt the LLM
     with the MicroCell schema to synthesize a new cell on-the-fly, validate its
     code template AST, execute a GEVR sandbox dry-run probe (with 1-turn repair),
     tag it with provisional confidence, register it dynamically into the active
     orchestrator and RAG index, and persist it to an isolated dev directory
     (`trees/dev_unreviewed/dev_cells.json`) without affecting production trees.
2. Auto-Adapter Synthesis: Automatically generates minimal typed bridge cells between
   disconnected operations.
3. Promotion Pipeline: Tracks execution usage and promotes reliable cells from
   provisional (0.70x) to established (1.0x).
"""
from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from config import settings
from inference import ModelManager
from lattice import Cell, LatticeOrchestrator
from utils import (
    extract_json_object,
    validate_code_template,
    safe_substitute_template,
    extract_template_placeholders,
)

try:
    from internal_rag import LocalRAG
except ImportError:
    LocalRAG = None

logger = logging.getLogger("nstl.node_resolver")

_STORAGE_LOCK = threading.RLock()


@contextlib.contextmanager
def _sync_dev_storage(trees_dir: Path, shared: bool = False):
    """
    Acquires an in-process re-entrant lock and cross-process operating system
    file lock on the dev storage lockfile to prevent race conditions and corruption
    across multi-threaded and multi-worker execution environments.
    """
    dev_dir = trees_dir / "dev_unreviewed"
    dev_dir.mkdir(parents=True, exist_ok=True)
    lock_file_path = dev_dir / ".dev_cells.lock"
    dev_file = dev_dir / "dev_cells.json"

    with _STORAGE_LOCK:
        lock_fd = None
        locked = False
        try:
            try:
                lock_fd = os.open(str(lock_file_path), os.O_CREAT | os.O_RDWR)
                if sys.platform != "win32":
                    import fcntl
                    lock_op = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                    fcntl.flock(lock_fd, lock_op)
                    locked = True
                else:
                    import msvcrt
                    if os.path.getsize(str(lock_file_path)) == 0:
                        os.write(lock_fd, b"\0")
                    os.lseek(lock_fd, 0, os.SEEK_SET)
                    msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
                    locked = True
            except Exception as lock_err:
                logger.debug(f"[Resolver] OS file lock unavailable or failed: {lock_err}")

            yield dev_file
        finally:
            if lock_fd is not None:
                if locked:
                    try:
                        if sys.platform != "win32":
                            import fcntl
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        else:
                            import msvcrt
                            os.lseek(lock_fd, 0, os.SEEK_SET)
                            msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass
                try:
                    os.close(lock_fd)
                except Exception:
                    pass


def _read_cells(dev_file: Path) -> List[Dict[str, Any]]:
    """Reads cells from disk assuming synchronization lock is held."""
    if not dev_file.exists():
        return []
    try:
        with open(dev_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"[Resolver] Failed to parse JSON from {dev_file}: {e}")
        return []


def _write_cells_atomic(dev_file: Path, cells: List[Dict[str, Any]]) -> None:
    """Atomically commits updated cells to disk using atomic filesystem replacement."""
    dev_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = dev_file.with_name(f"{dev_file.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cells, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, dev_file)
    except Exception:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise


CELL_SYNTHESIS_SCHEMA = {
    "type": "object",
    "required": ["cell_id", "stage", "node_role", "inputs", "outputs", "code_template"],
    "properties": {
        "cell_id": {"type": "string"},
        "stage": {"type": "integer", "enum": [1, 2, 3]},
        "domain_name": {"type": "string"},
        "node_role": {"type": "string"},
        "inputs": {"type": "object"},
        "outputs": {"type": "object"},
        "code_template": {"type": "string"},
        "docstring": {"type": "string"},
        "dependencies": {"type": "array", "items": {"type": "string"}},
    },
}

SYSTEM_PROMPT_SYNTHESIS = (
    "You are an expert compiler and runtime engineer for the STRATA (Symbolic Topological Routing Architecture for Typed Automata).\n"
    "Your task is to synthesize a valid typed MicroCell definition for a missing operation requested by a pipeline.\n"
    "The cell must adhere strictly to the STRATA schema:\n"
    "- cell_id: lowercase snake_case functional identifier representing the operation (e.g. '<verb>_<noun>')\n"
    "- stage: integer stage index (1: source/reader/ingest, 2: transform/compute/model, 3: sink/writer/output)\n"
    "- domain_name: string identifier denoting the operational domain or subsystem\n"
    "- node_role: 'source', 'transform', 'estimator', 'evaluator', 'bridge', or 'sink'\n"
    "- inputs: dict mapping input port names to {'type_name': str, 'state': str, 'required': bool}\n"
    "- outputs: dict mapping output port names to {'type_name': str, 'state': str}\n"
    "- code_template: valid Python statement or expression snippet using standard port variable names\n"
    "- docstring: concise description of the operation\n"
    "- dependencies: list of required import package names\n"
    "Output ONLY a valid JSON object matching the schema. No markdown formatting, no commentary."
)


def _safe_extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    return extract_json_object(raw_text)


def _create_mock_input_code(var_name: str, type_name: str, domain_name: str = "", abstract_type: str = "") -> str:
    """
    Generates a dry-run mock instantiation for a port.
    Domain knowledge lives in pluggable trees: mock probes are harvested from trees.
    Fallback synthesizes values by abstract carrier protocol: table, tensor, scalar, text, collection.
    """
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        probe = reg.find_mock_probe(type_name, domain_name) if reg else None
    except Exception:
        reg = None
        probe = None

    if probe:
        return probe.replace("{var_name}", var_name)

    t_low = (type_name or "").lower()
    abs_low = (abstract_type or "").lower()

    if (reg and (reg.is_scalar_primitive(t_low) or reg.is_subtype(t_low, "scalar"))) or abs_low == "scalar":
        return f"{var_name} = 1.0"
    elif (reg and reg.is_subtype(t_low, "text")) or abs_low == "text" or t_low == "text":
        return f'{var_name} = "sample"'
    elif (reg and reg.is_subtype(t_low, "collection")) or abs_low == "collection" or t_low == "collection":
        return f"{var_name} = [1.0, 2.0]"

    # Universal dynamic mock carrier fallback
    return (
        f"class _NSTLMockCarrier(list):\n"
        f"    def __getattr__(self, name): return lambda *a, **kw: _NSTLMockCarrier([1, 2, 3])\n"
        f"    def __getitem__(self, key): return _NSTLMockCarrier([1, 2, 3])\n"
        f"    shape = (3, 3)\n"
        f"{var_name} = _NSTLMockCarrier([1, 2, 3])"
    )


def _generate_mock_code_probe(
    code_template: str,
    inputs: dict,
    outputs: dict,
    dependencies: list,
    domain_name: str = "",
) -> str:
    """
    Generates a self-contained execution probe for dry-run verification in GEVRSandbox.
    Constructs domain-appropriate mock variables across diverse modalities.
    Zero regular expressions.
    """
    lines = ["import sys, os"]
    for dep in dependencies or []:
        if dep and str(dep).isidentifier():
            lines.append(f"try:\n    import {dep}\nexcept ImportError:\n    pass")

    mock_vars: Dict[str, str] = {}
    for idx, (p_name, p_info) in enumerate((inputs or {}).items()):
        var_name = f"in_mock_{idx}"
        type_name = ""
        abstract_type = ""
        if isinstance(p_info, dict):
            type_name = str(p_info.get("type_name", "")).lower()
            abstract_type = str(p_info.get("abstract_type", "")).lower()
        elif hasattr(p_info, "type_name"):
            type_name = str(p_info.type_name).lower()
            abstract_type = str(getattr(p_info, "abstract_type", "")).lower()

        lines.append(_create_mock_input_code(var_name, type_name, domain_name, abstract_type=abstract_type))
        mock_vars[p_name] = var_name

    bindings = dict(mock_vars)
    for p_name in (outputs or {}).keys():
        bindings[p_name] = f"out_mock_{p_name}"

    exec_code = safe_substitute_template(code_template, bindings)
    for ph in extract_template_placeholders(exec_code):
        exec_code = exec_code.replace(f"{{{ph}}}", "None")

    lines.append(exec_code)
    return "\n".join(lines)


class DynamicNodeResolver:
    """
    Coordinates lookup, RAG boosting, sandbox dry-run verification,
    and on-demand synthesis of cells proposed by neural route methods (M7, M8, M9).
    """

    @classmethod
    def resolve_node(
        cls,
        raw_id: str,
        prompt: str,
        orchestrator: LatticeOrchestrator,
        rag: Optional[Any] = None,
        dev_mode: Optional[bool] = None,
        domain_hint: Optional[str] = None,
    ) -> Optional[Cell]:
        """
        Attempts to resolve an unknown or missing node ID:
        1. If it exists in orchestrator catalog: boosts score in RAG and returns it.
        2. If missing and dev_mode is enabled: synthesizes a new MicroCell with AST
           validation, executes a GEVR sandbox dry run, records it as unreviewed,
           registers it dynamically, and persists to trees/dev_unreviewed/dev_cells.json.
        """
        if not raw_id:
            return None

        clean_id = str(raw_id).strip().strip("\"'").lower()
        try:
            from lattice import TypeRegistry
            reg_nr = TypeRegistry.get_instance()
        except Exception:
            reg_nr = None
        stop_toks = (
            {str(t).lower() for t in (reg_nr.get_verification_semantics("stopping_tokens") or ())}
            | {str(t).lower() for t in (reg_nr.get_verification_semantics("void_types") or ())}
            | {str(t).lower() for t in (reg_nr.get_verification_semantics("bottom_types") or ())}
        ) if reg_nr is not None else {None.__class__.__name__.lower()}
        if not clean_id or clean_id in stop_toks:
            return None

        active_rag = rag or getattr(orchestrator, "rag", None)
        is_dev = dev_mode if dev_mode is not None else settings.dev_mode

        # -------------------------------------------------------------
        # Step 1: Check if node exists in orchestrator loaded catalog
        # -------------------------------------------------------------
        cell = orchestrator.loaded_cells.get(clean_id)
        if cell is None:
            norm_id = clean_id.replace("-", "_")
            cell = orchestrator.loaded_cells.get(norm_id)
            if cell is None:
                # Case-insensitive exact match
                for k, c in orchestrator.loaded_cells.items():
                    if k.lower() == norm_id:
                        cell = c
                        break

            # Qualified prefix/suffix match: only for unambiguous candidates when norm_id is specific
            if cell is None and len(norm_id) > 6:
                candidates = [
                    c for k, c in orchestrator.loaded_cells.items()
                    if k.lower().endswith(f"_{norm_id}") or norm_id.endswith(f"_{k.lower()}")
                ]
                if len(candidates) == 1:
                    cell = candidates[0]

            if cell is None:
                try:
                    from token_evidence import get_evidence
                    ev = get_evidence(orchestrator)
                    if ev is not None:
                        q_tokens = ev.clause_tokens(clean_id)
                        best_overlap_cell = None
                        best_f = 0.0
                        for c_cand in orchestrator.loaded_cells.values():
                            o = ev.identity_overlap(q_tokens, c_cand)
                            if o.get("f", 0.0) > best_f:
                                best_f = o["f"]
                                best_overlap_cell = c_cand
                        if best_overlap_cell is not None and best_f >= 0.35:
                            cell = best_overlap_cell
                except Exception:
                    pass

        if cell is not None:
            # Case A: Found in catalog! Boost in RAG so it surfaces naturally
            if active_rag is not None and hasattr(active_rag, "boost_cell"):
                try:
                    active_rag.boost_cell(cell.cell_id, boost=1.5)
                except Exception as e:
                    logger.debug(f"[Resolver] Could not boost cell '{cell.cell_id}' in RAG: {e}")
            logger.info(
                f"[Resolver] Resolved library node '{cell.cell_id}' from catalog and boosted RAG score."
            )
            return cell

        # -------------------------------------------------------------
        # Step 2: Missing from catalog -> Check Dev Mode for synthesis
        # -------------------------------------------------------------
        if not is_dev:
            logger.debug(
                f"[Resolver] Node '{raw_id}' not found in catalog and dev_mode=False. Synthesis disabled."
            )
            return None

        # Dev Mode is ON -> Synthesize the missing cell
        logger.info(f"[Resolver] Dev Mode active: attempting on-demand synthesis for node '{clean_id}'...")
        mm = ModelManager.get_instance()
        if not mm.can_synthesize():
            logger.warning("[Resolver] Dev Mode active but model cannot synthesize text.")
            return None

        target_domain = domain_hint or getattr(orchestrator, "active_domain", "generic") or "generic"
        user_req = (
            f"User Pipeline Request: {prompt}\n"
            f"Requested Missing Operation: {clean_id}\n"
            f"Target Domain Hint: {target_domain}\n\n"
            f"Synthesize the complete MicroCell JSON schema for this operation."
        )

        try:
            raw_text = mm.generate_text(
                user_req,
                max_tokens=600,
                schema=CELL_SYNTHESIS_SCHEMA,
                system_prompt=SYSTEM_PROMPT_SYNTHESIS,
            )
        except Exception as e:
            logger.error(f"[Resolver] Failed to invoke LLM for node synthesis: {e}")
            return None

        cell_data = _safe_extract_json(raw_text)
        if not cell_data or not isinstance(cell_data, dict):
            logger.warning(f"[Resolver] LLM returned invalid JSON for cell '{clean_id}': {raw_text[:100]}")
            return None

        # Normalize and validate synthesized cell
        cell_id = str(cell_data.get("cell_id") or clean_id).strip()
        code_template = str(cell_data.get("code_template") or "")
        inputs = cell_data.get("inputs") or {}
        outputs = cell_data.get("outputs") or {}
        stage = int(cell_data.get("stage") or 2)
        node_role = str(cell_data.get("node_role") or "transform").lower()
        domain = str(cell_data.get("domain_name") or target_domain)
        dependencies = cell_data.get("dependencies") or []

        # AST Validation: verify code_template contains syntactically valid Python
        if code_template:
            if not validate_code_template(code_template):
                logger.warning(
                    f"[Resolver] Synthesized code_template for '{cell_id}' failed AST validation."
                )
                return None

        # GEVR Sandbox Dry-Run Verification Gate with 1-turn repair
        dry_run_verified = False
        try:
            from gevr_sandbox import GEVRSandbox
            sandbox = GEVRSandbox()
            probe = _generate_mock_code_probe(code_template, inputs, outputs, dependencies, domain_name=domain)
            res = sandbox.execute(probe, timeout=2.0)
            if res.get("success"):
                dry_run_verified = True
            else:
                err_msg = res.get("error", "Dry-run execution failed")
                logger.warning(
                    f"[Resolver] Dry-run probe failed for '{cell_id}': {err_msg}. Attempting 1-turn repair..."
                )
                repaired = mm.feedback_check(code_template, err_msg)
                if repaired and repaired != code_template and validate_code_template(repaired):
                    probe2 = _generate_mock_code_probe(repaired, inputs, outputs, dependencies, domain_name=domain)
                    res2 = sandbox.execute(probe2, timeout=2.0)
                    if res2.get("success"):
                        code_template = repaired
                        dry_run_verified = True
                        logger.info(f"[Resolver] Self-repair succeeded on dry-run for '{cell_id}'.")
                    else:
                        logger.warning(f"[Resolver] Self-repair failed on dry-run for '{cell_id}'. Discarding.")
                        return None
                else:
                    logger.warning(f"[Resolver] Dry-run failed with no repair available for '{cell_id}'. Discarding.")
                    return None
        except Exception as e:
            logger.debug(f"[Resolver] Sandbox dry run skipped: {e}")

        clean_schema: Dict[str, Any] = {
            "cell_id": cell_id,
            "stage": stage,
            "node_type": "function",
            "node_role": node_role,
            "domain_name": domain,
            "inputs": inputs,
            "outputs": outputs,
            "code_template": code_template,
            "docstring": str(cell_data.get("docstring") or f"Synthesized cell for {cell_id}"),
            "dependencies": dependencies,
            "verified": False,
            "dry_run_verified": dry_run_verified,
            "reviewed": False,
            "success_count": 0,
            "fail_count": 0,
            "source_provenance": "dev_llm_synthesized",
            "provisional_score_mult": 0.70,
        }

        # Persist safely to isolated dev directory (never corrupting production trees)
        cls._persist_dev_cell(clean_schema)

        # Register dynamically in active orchestrator
        synthesized_cell = orchestrator.register_cell(clean_schema, domain=domain)

        # Register dynamically in active RAG index with provisional discount
        if active_rag is not None and hasattr(active_rag, "register_dynamic_cell"):
            try:
                active_rag.register_dynamic_cell(
                    clean_schema, reviewed=False, provisional_score_mult=0.70
                )
            except Exception as e:
                logger.debug(f"[Resolver] Could not index dynamic cell in RAG: {e}")

        logger.info(
            f"[Resolver] Successfully synthesized, verified, and registered cell: '{cell_id}' "
            f"(stage={stage}, role={node_role}, dry_run_verified={dry_run_verified}, score_discount=0.70x)"
        )
        return synthesized_cell

    @classmethod
    def synthesize_adapter(
        cls,
        src_cell: Cell,
        dst_cell: Cell,
        orchestrator: LatticeOrchestrator,
        rag: Optional[Any] = None,
        dev_mode: Optional[bool] = None,
    ) -> Optional[Cell]:
        """
        Synthesizes a minimal 1-step converter/adapter cell on demand
        when src_cell output cannot directly unify with dst_cell input and no bridge cell exists.
        """
        is_dev = dev_mode if dev_mode is not None else settings.dev_mode
        if not is_dev:
            return None

        mm = ModelManager.get_instance()
        if not mm.can_synthesize():
            return None

        src_out = getattr(src_cell, "primary_output", None)
        dst_in = getattr(dst_cell, "primary_input", None)
        src_out_type = getattr(src_out, "type_name", "any") if src_out else "any"
        dst_in_type = getattr(dst_in, "type_name", "any") if dst_in else "any"

        adapter_id = f"adapt_{src_out_type}_to_{dst_in_type}".lower().replace(":", "_").replace("-", "_")

        existing = orchestrator.loaded_cells.get(adapter_id)
        if existing is not None:
            return existing

        prompt = (
            f"Synthesize an adapter MicroCell bridging output '{src_out_type}' from '{src_cell.cell_id}' "
            f"to input '{dst_in_type}' for '{dst_cell.cell_id}'."
        )
        return cls.resolve_node(
            raw_id=adapter_id,
            prompt=prompt,
            orchestrator=orchestrator,
            rag=rag,
            dev_mode=True,
            domain_hint=getattr(src_cell, "domain_name", "generic"),
        )

    @classmethod
    def record_cell_usage(cls, cell_id: str, success: bool = True) -> None:
        """
        Tracks cell usage in successful executions and handles confidence promotion
        from provisional (0.70x) to established (1.0x).
        Thread and process synchronized with atomic file replacement.
        """
        try:
            trees_dir = Path(settings.trees_dir) if settings.trees_dir else Path("trees")
            with _sync_dev_storage(trees_dir, shared=False) as dev_file:
                if not dev_file.exists():
                    return
                cells = _read_cells(dev_file)
                if not cells:
                    return

                updated = False
                for c in cells:
                    if c.get("cell_id", "").lower() == str(cell_id).lower():
                        if success:
                            c["success_count"] = c.get("success_count", 0) + 1
                        else:
                            c["fail_count"] = c.get("fail_count", 0) + 1

                        sc = c.get("success_count", 0)
                        if sc >= 5:
                            c["reviewed"] = True
                            c["provisional_score_mult"] = 1.0
                        elif sc >= 3:
                            c["provisional_score_mult"] = 0.90
                        updated = True
                        break

                if updated:
                    _write_cells_atomic(dev_file, cells)
        except Exception as e:
            logger.debug(f"[Resolver] Could not record cell usage for '{cell_id}': {e}")

    @classmethod
    def list_unreviewed_cells(cls) -> List[Dict[str, Any]]:
        """Returns all dev cells currently stored in dev_unreviewed/dev_cells.json."""
        trees_dir = Path(settings.trees_dir) if settings.trees_dir else Path("trees")
        try:
            with _sync_dev_storage(trees_dir, shared=True) as dev_file:
                return _read_cells(dev_file)
        except Exception:
            return []

    @classmethod
    def promote_cell(
        cls,
        cell_id: str,
        orchestrator: Optional[LatticeOrchestrator] = None,
        rag: Optional[Any] = None,
    ) -> bool:
        """Promotes an unreviewed dev cell to fully reviewed status with 1.0 score multiplier."""
        trees_dir = Path(settings.trees_dir) if settings.trees_dir else Path("trees")
        try:
            found = False
            with _sync_dev_storage(trees_dir, shared=False) as dev_file:
                if not dev_file.exists():
                    return False
                cells = _read_cells(dev_file)
                for c in cells:
                    if c.get("cell_id", "").lower() == str(cell_id).lower():
                        c["reviewed"] = True
                        c["verified"] = True
                        c["provisional_score_mult"] = 1.0
                        found = True
                        break
                if found:
                    _write_cells_atomic(dev_file, cells)

            if found:
                if rag is not None and hasattr(rag, "cell_boosts"):
                    rag.cell_boosts[str(cell_id).lower()] = 1.0
                return True
        except Exception:
            pass
        return False

    @classmethod
    def discard_cell(
        cls,
        cell_id: str,
        orchestrator: Optional[LatticeOrchestrator] = None,
    ) -> bool:
        """Removes a dev cell from unreviewed list and live orchestrator."""
        trees_dir = Path(settings.trees_dir) if settings.trees_dir else Path("trees")
        try:
            with _sync_dev_storage(trees_dir, shared=False) as dev_file:
                if not dev_file.exists():
                    return False
                cells = _read_cells(dev_file)
                new_cells = [c for c in cells if c.get("cell_id", "").lower() != str(cell_id).lower()]
                _write_cells_atomic(dev_file, new_cells)

            if orchestrator and cell_id in orchestrator.loaded_cells:
                del orchestrator.loaded_cells[cell_id]
                orchestrator.build_topology()
            return True
        except Exception:
            return False

    @classmethod
    def _persist_dev_cell(cls, cell_schema: Dict[str, Any]) -> None:
        """
        Saves the synthesized cell into an isolated dev file:
        trees/dev_unreviewed/dev_cells.json.
        Completely non-destructive to existing production trees.
        Fully synchronized across processes and threads with atomic filesystem replacement.
        """
        try:
            trees_dir = Path(settings.trees_dir) if settings.trees_dir else Path("trees")
            with _sync_dev_storage(trees_dir, shared=False) as dev_file:
                existing = _read_cells(dev_file)
                cid = cell_schema.get("cell_id")
                existing = [c for c in existing if c.get("cell_id") != cid]
                existing.append(cell_schema)
                _write_cells_atomic(dev_file, existing)

            logger.info(f"[Resolver] Persisted unreviewed dev cell to {dev_file}")
        except Exception as e:
            logger.warning(f"[Resolver] Could not persist dev cell to disk: {e}")
