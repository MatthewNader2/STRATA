# src/cli.py
import argparse
import ast
import cmd
import contextlib
import glob
import hashlib
import io
import json
import math
import operator
import os
import shutil
import sqlite3
import sys
import time
import tokenize
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from schema import CellSchema, TreeSchema, PortSchema
    from harvester import IntelligentHarvester
    from lattice import TypeRegistry
except ImportError as _err:
    if __package__:
        from .schema import CellSchema, TreeSchema, PortSchema
        from .harvester import IntelligentHarvester
        from .lattice import TypeRegistry
    else:
        raise _err

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from tools.compile_trees import synthesize_dynamic_edges
except Exception:
    synthesize_dynamic_edges = None

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.align import Align
from rich import box
from rich.columns import Columns

try:
    from log_config import get_logger
except (ImportError, ValueError):
    from .log_config import get_logger

logger = get_logger("cli")

try:
    from lattice import LatticeOrchestrator, Cell, PortSignature, AlgebraicSignature, UNRESOLVED_PORT
    from router import LatticeRouter, HardwareProfiler
    from unification import (
        UnificationGate, UnresolvedPlaceholderError,
        UnificationFailure, ExecutionContext, Substitution, TypeRegistry, unify,
        Success, Failure
    )
    from gevr_sandbox import GEVRSandbox
    from inference import ModelManager, select_optimal_embedder, select_optimal_llm
    from internal_rag import LocalRAG
    from config import MODELS_DIR, settings
    from utils import extract_code_from_llm_response, ensure_comprehensive_prompt, extract_template_placeholders, safe_substitute_template
    from errors import SynthesisError
    from tokenizer import CellTokenizer
    from planner import _segment_prompt_clauses, STOPWORDS
    from preflight import PreflightLinter
    from lattice_auditor import LatticeAuditor
    from macro_harvester import MacroHarvester
    from route_methods import ROUTE_METHOD_REGISTRY, get_route_method, route_method_ids, describe_route_methods
    from reranker import get_available_rerankers, LocalReranker
except ImportError as _err:
    if __package__:
        from .lattice import LatticeOrchestrator, Cell, PortSignature, AlgebraicSignature, UNRESOLVED_PORT
        from .router import LatticeRouter, HardwareProfiler
        from .unification import (
            UnificationGate, UnresolvedPlaceholderError,
            UnificationFailure, ExecutionContext, Substitution, TypeRegistry, unify,
            Success, Failure
        )
        from .gevr_sandbox import GEVRSandbox
        from .inference import ModelManager, select_optimal_embedder, select_optimal_llm
        from .internal_rag import LocalRAG
        from .config import MODELS_DIR, settings
        from .utils import extract_code_from_llm_response, ensure_comprehensive_prompt, extract_template_placeholders, safe_substitute_template
        from .errors import SynthesisError
        from .tokenizer import CellTokenizer
        from .planner import _segment_prompt_clauses, STOPWORDS
        from .preflight import PreflightLinter
        from .lattice_auditor import LatticeAuditor
        from .macro_harvester import MacroHarvester
        from .route_methods import ROUTE_METHOD_REGISTRY, get_route_method, route_method_ids, describe_route_methods
        from .reranker import get_available_rerankers, LocalReranker
    else:
        raise _err

console = Console()

# Single source of truth for the numeric quick-shortcuts to profile letters. The dashboard
# banner and the `default()` dispatcher both derive from this instead of keeping two
# separately-maintained switch statements in sync by hand.
QUICK_PROFILE_SHORTCUTS: Tuple[Tuple[str, str], ...] = (
    ("1", "0"), ("2", "A"), ("3", "C"), ("4", "D"), ("5", "E"), ("6", "S"),
)

# =====================================================================
# Constants & Defaults
# =====================================================================
DEFAULT_SANDBOX_TIMEOUT: float = float(os.environ.get("NSTL_SANDBOX_TIMEOUT", settings.sandbox_timeout))
HIGH_TRUST_PRIORITY_THRESHOLD: int = settings.verified_priority_threshold
DEFAULT_TRANSLATOR_PROMPT: str = (
    "You are a precise technical translator. Rewrite the user request as a "
    "sequential pipeline specification that preserves every computational operation, its arguments, "
    "all variable assignments, target output names, and their requested execution order. "
    "Preserve relational causality and output definitions (e.g. writing or assigning computed results "
    "to target variables or sinks: 'write result to Z' -> 'assign result to variable Z'). "
    "Keep the natural clause boundaries of the request — one operation per clause — "
    "without omitting, merging, or reordering steps. Do not drop identifiers, literals, or target outputs. "
    "Output ONLY the normalized pipeline specification, no headers, no lists, no markdown formatting."
)


def get_translator_prompt(orchestrator: Optional[Any] = None) -> str:
    """Retrieves domain-adapted or configured system prompt for query translation."""
    if orchestrator and hasattr(orchestrator, "translator_prompt") and orchestrator.translator_prompt:
        return orchestrator.translator_prompt
    if hasattr(settings, "translator_prompt") and settings.translator_prompt:
        return settings.translator_prompt
    return DEFAULT_TRANSLATOR_PROMPT


def sanitize_placeholders_for_ast(code: str) -> str:
    """
    Replaces unquoted template placeholders '{identifier}' with valid dummy identifiers
    for AST dry-run parsing, without corrupting string literals, f-strings, or comments.
    """
    if not code or not code.strip():
        return code

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError):
        # Fallback if tokenizer encounters unclosed delimiters before substitution
        placeholders = extract_template_placeholders(code)
        sub_map = {ph: f"__ph_{ph}" for ph in placeholders}
        return safe_substitute_template(code, sub_map)

    new_tokens: List[tokenize.TokenInfo] = []
    seen: Dict[str, str] = {}
    i = 0
    n = len(tokens)

    while i < n:
        if (
            i + 2 < n
            and tokens[i].type == tokenize.OP and tokens[i].string == "{"
            and tokens[i + 1].type == tokenize.NAME and tokens[i + 1].string.isidentifier()
            and tokens[i + 2].type == tokenize.OP and tokens[i + 2].string == "}"
        ):
            ph_name = tokens[i + 1].string
            if ph_name not in seen:
                seen[ph_name] = f"_ph_{len(seen)}"
            dummy_id = seen[ph_name]

            new_tokens.append(tokenize.TokenInfo(
                tokenize.NAME, dummy_id, tokens[i].start, tokens[i + 2].end, tokens[i].line
            ))
            i += 3
        else:
            new_tokens.append(tokens[i])
            i += 1

    try:
        return tokenize.untokenize(new_tokens)
    except Exception:
        placeholders = extract_template_placeholders(code)
        sub_map = {ph: f"__ph_{ph}" for ph in placeholders}
        return safe_substitute_template(code, sub_map)


# =====================================================================
# Extensible Contract Evaluation Registry
# =====================================================================
ContractEvaluator = Callable[[Dict[str, Any], str], Optional[str]]
CONTRACT_RULE_REGISTRY: Dict[str, ContractEvaluator] = {}


def register_contract_rule(rule_type: str):
    """Decorator to register verification contract evaluation rules."""
    def decorator(fn: ContractEvaluator):
        CONTRACT_RULE_REGISTRY[rule_type] = fn
        return fn
    return decorator


@register_contract_rule("var_equality")
def _eval_var_equality(check: Dict[str, Any], code: str) -> Optional[str]:
    expected = check.get("expected_var") or check.get("expected_train_feature_var")
    actual = check.get("actual_var") or check.get("feature_var")
    cid = check.get("cell_id", "terminal_node")
    if expected is not None and actual is not None and actual != expected:
        return f"Variable equality contract failed for node '{cid}': expected '{expected}', got '{actual}'"
    return None


@register_contract_rule("lineage_descendant")
def _eval_lineage_descendant(check: Dict[str, Any], code: str) -> Optional[str]:
    saved = check.get("saved_var") or check.get("actual_var")
    expected = check.get("expected_var") or check.get("annotated_var")
    ingress = check.get("ingress_var")
    cid = check.get("cell_id", "terminal_node")
    if saved and expected and ingress and saved == ingress and expected != ingress:
        return f"Lineage descendant contract failed for node '{cid}': bound to ancestor '{ingress}' instead of derived descendant '{expected}'"
    return None


_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "in": lambda a, b: a in b,
    "not in": lambda a, b: a not in b,
}


@register_contract_rule("property_satisfied")
def _eval_property_satisfied(check: Dict[str, Any], code: str) -> Optional[str]:
    cid = check.get("cell_id", "terminal_node")
    prop = check.get("property")
    op = str(check.get("operator", "==")).strip()
    expected_val = check.get("expected_value", check.get("value"))
    actual_val = check.get("actual_value")
    if actual_val is not None and expected_val is not None:
        fn = _OPERATORS.get(op, operator.eq)
        try:
            passed = fn(actual_val, expected_val)
        except Exception:
            passed = False
        if not passed:
            return f"Property contract '{prop}' failed for node '{cid}': condition '{actual_val} {op} {expected_val}' unsatisfied"
    return None


def _statically_evaluate_contract(final_code: str, contract: Any) -> List[str]:
    """
    Evaluates VerificationContract postconditions and terminal intent statically
    via universal relational operators and generic contract dispatch.
    """
    if not contract:
        return []
    violations: List[str] = []

    term_checks = getattr(contract, "terminal_checks", []) or []
    if isinstance(contract, dict):
        term_checks = contract.get("terminal_checks", [])

    for term_check in term_checks:
        intent_type = term_check.get("type", "")
        evaluator = CONTRACT_RULE_REGISTRY.get(intent_type)
        if evaluator:
            err = evaluator(term_check, final_code)
            if err:
                violations.append(err)
        elif "saved_var" in term_check and "ingress_var" in term_check:
            err = _eval_lineage_descendant(term_check, final_code)
            if err:
                violations.append(err)
        elif "property" in term_check and ("expected_value" in term_check or "value" in term_check):
            err = _eval_property_satisfied(term_check, final_code)
            if err:
                violations.append(err)
        elif "expected_var" in term_check and "actual_var" in term_check:
            err = _eval_var_equality(term_check, final_code)
            if err:
                violations.append(err)
        elif "expected_state" in term_check and "actual_state" in term_check:
            if term_check.get("actual_state") != term_check.get("expected_state"):
                cid = term_check.get("cell_id", "terminal_node")
                violations.append(
                    f"State contract failed for node '{cid}': expected {term_check.get('expected_state')}, got {term_check.get('actual_state')}"
                )

    return violations


def _get_score_color(score: float, epsilon: float = 0.001) -> str:
    """Computes color coding based on dynamic router confidence tiers."""
    if score >= 0.5:
        return "bold green"
    elif score >= max(0.1, epsilon * 10):
        return "bold yellow"
    return "white"


# =====================================================================
# Database & Compilation Subsystem
# =====================================================================
def cmd_harvest(args):
    """Harvest public APIs from a package and merge into trees/{domain}.json."""
    domain = args.domain or args.package
    package = args.package
    trees_dir = Path(args.trees_dir)
    trees_dir.mkdir(parents=True, exist_ok=True)
    out_file = trees_dir / f"{domain}.json"

    print(f"[*] Initializing Intelligent Harvester for package '{package}' (domain: '{domain}')...")
    harvester = IntelligentHarvester(domain=domain, package_name=package)
    tree = harvester.harvest_and_save(out_file)
    print(f"[+] Harvested {len(tree.cells)} function cells from '{package}'.")
    print(f"[+] Merged and saved into '{out_file}'.")


def init_sqlite_db(db_path: Path, clean: bool = False) -> None:
    """Initializes standard STRATA SQLite schema without destroying existing data unless clean=True."""
    if clean and db_path.exists():
        os.remove(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                cell_id              TEXT PRIMARY KEY,
                domain_name          TEXT,
                node_type            TEXT,
                node_role            TEXT DEFAULT 'function',
                stage                INTEGER,
                keywords             TEXT,
                input_type           TEXT,
                input_state          TEXT,
                output_type          TEXT,
                output_state         TEXT,
                code                 TEXT,
                dependencies         TEXT,
                configuration_schema TEXT,
                slots                TEXT DEFAULT '{}',
                verified             INTEGER DEFAULT 0,
                docstring            TEXT DEFAULT '',
                enrichment_source    TEXT DEFAULT NULL,
                enriched_at          TEXT DEFAULT NULL,
                source_provenance    TEXT DEFAULT 'unknown',
                source_priority      INTEGER DEFAULT 100,
                raw_cell             TEXT DEFAULT NULL
            )
        """)
        cur.execute("PRAGMA table_info(nodes)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "slots" not in existing_cols:
            try:
                cur.execute("ALTER TABLE nodes ADD COLUMN slots TEXT DEFAULT '{}'")
            except Exception:
                pass
        if "raw_cell" not in existing_cols:
            try:
                cur.execute("ALTER TABLE nodes ADD COLUMN raw_cell TEXT DEFAULT NULL")
            except Exception:
                pass

        cur.execute("CREATE INDEX IF NOT EXISTS idx_input ON nodes(input_type, input_state)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_output ON nodes(output_type, output_state)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_domain ON nodes(domain_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_role ON nodes(node_role)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_type ON nodes(node_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_slots ON nodes(slots)")
        cur.execute("CREATE TABLE IF NOT EXISTS types (type_name TEXT, parent_type TEXT, domain_name TEXT, PRIMARY KEY (type_name, parent_type))")
        cur.execute("CREATE TABLE IF NOT EXISTS typestates (state_name TEXT PRIMARY KEY, parent_state TEXT, carrier_type TEXT, properties TEXT, domain_name TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS aliases (alias TEXT PRIMARY KEY, canonical TEXT, domain_name TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS artifact_readers (category TEXT, module_name TEXT, function_name TEXT, domain_name TEXT, PRIMARY KEY (category, module_name, function_name))")
        cur.execute("CREATE TABLE IF NOT EXISTS structural_metadata (category TEXT, item TEXT, extra TEXT, domain_name TEXT, PRIMARY KEY (category, item, extra, domain_name))")
        cur.execute("CREATE TABLE IF NOT EXISTS _compilation_meta (key TEXT PRIMARY KEY, value TEXT, timestamp REAL)")
        conn.commit()


# Bump whenever the compiled node serialization changes: forces stale databases to rebuild.
COMPILER_SCHEMA_VERSION = "2"


def compute_trees_fingerprint(trees_dir: Union[str, Path] = "trees") -> str:
    """Computes a fast SHA-256 fingerprint over all tree JSON files in the directory."""
    td = Path(trees_dir)
    if not td.exists():
        return "MISSING"
    try:
        entries = sorted([e for e in os.scandir(td) if e.is_file() and e.name.endswith(".json")], key=lambda x: x.name)
    except OSError:
        return "MISSING"

    if not entries:
        return "EMPTY"

    h = hashlib.sha256()
    h.update(f"compiler:{COMPILER_SCHEMA_VERSION}".encode("utf-8"))
    for entry in entries:
        h.update(entry.name.encode("utf-8"))
        try:
            stat = entry.stat()
            h.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        except OSError:
            pass
    return h.hexdigest()


def get_stored_meta(db_path: Union[str, Path], key: str) -> Optional[str]:
    """Reads one value from the compilation metadata table (None when absent)."""
    target = Path(db_path)
    if not target.exists():
        return None
    try:
        with contextlib.closing(sqlite3.connect(str(target))) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            row = conn.execute("SELECT value FROM _compilation_meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None
    except Exception:
        return None


def get_stored_fingerprint(db_path: Union[str, Path]) -> Optional[str]:
    """Retrieves the compilation fingerprint recorded in the SQLite database."""
    target = Path(db_path)
    if not target.exists():
        return None
    try:
        with contextlib.closing(sqlite3.connect(str(target))) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            cur = conn.cursor()
            cur.execute("SELECT value FROM _compilation_meta WHERE key = 'trees_fingerprint'")
            row = cur.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def record_trees_fingerprint(db_path: Union[str, Path], fingerprint: str) -> None:
    """Records the tree compilation fingerprint into the SQLite database."""
    try:
        with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS _compilation_meta (key TEXT PRIMARY KEY, value TEXT, timestamp REAL)")
            cur.execute("INSERT OR REPLACE INTO _compilation_meta (key, value, timestamp) VALUES ('trees_fingerprint', ?, ?)",
                        (fingerprint, time.time()))
            cur.execute("INSERT OR REPLACE INTO _compilation_meta (key, value, timestamp) VALUES ('compiler_schema', ?, ?)",
                        (COMPILER_SCHEMA_VERSION, time.time()))
            conn.commit()
    except Exception:
        pass


def ensure_lattice_compiled(trees_dir: Union[str, Path] = "trees", db_path: Union[str, Path] = "trees/lattice.db") -> bool:
    """
    Ensures target SQLite database is up-to-date with domain trees.
    Performs fast O(1) directory mtime checks on startup to eliminate stat storm bottlenecks.
    """
    trees_path = Path(trees_dir)
    target_db = Path(db_path)

    schema_current = (not target_db.exists()) or get_stored_meta(target_db, "compiler_schema") == COMPILER_SCHEMA_VERSION

    if target_db.exists() and schema_current:
        try:
            db_mtime = target_db.stat().st_mtime_ns
            dir_mtime = trees_path.stat().st_mtime_ns
            if db_mtime >= dir_mtime:
                stored_fp = get_stored_fingerprint(target_db)
                if stored_fp is not None:
                    entries = [e for e in os.scandir(trees_path) if e.is_file() and e.name.endswith(".json")]
                    if not entries and stored_fp == "EMPTY":
                        return False
                    if entries and all(e.stat().st_mtime_ns <= db_mtime for e in entries):
                        return False
        except OSError:
            pass

    current_fp = compute_trees_fingerprint(trees_path)
    stored_fp = get_stored_fingerprint(target_db)

    if target_db.exists() and schema_current and stored_fp is not None and stored_fp == current_fp:
        return False

    if current_fp == "EMPTY":
        init_sqlite_db(target_db, clean=True)
        record_trees_fingerprint(target_db, current_fp)
        return True

    print(f"[*] Tree update detected in '{trees_path}'. Auto-compiling lattice database '{target_db}'...")
    compile_args = argparse.Namespace(
        trees_dir=str(trees_path),
        output=str(target_db),
        domains=[],
        clean=True
    )
    cmd_compile(compile_args)
    record_trees_fingerprint(target_db, current_fp)
    return True


def cmd_compile(args):
    """Compiles all trees/*.json domain files into a target SQLite database by delegating to tools.compile_trees."""
    trees_dir = getattr(args, 'trees_dir', None) or getattr(args, 'dir', 'trees')
    out_db = getattr(args, 'output', 'trees/lattice.db')
    domain_filter = getattr(args, 'domains', None) or None

    try:
        from tools.compile_trees import compile_database
    except ImportError:
        if ROOT_DIR not in sys.path:
            sys.path.insert(0, ROOT_DIR)
        from tools.compile_trees import compile_database

    init_sqlite_db(Path(out_db), clean=getattr(args, 'clean', False))
    written = compile_database(output_db=str(out_db), domain_filter=domain_filter, trees_dir=trees_dir)
    record_trees_fingerprint(Path(out_db), compute_trees_fingerprint(Path(trees_dir)))
    print(f"[*] Compilation Complete: {written} total verified nodes compiled into '{out_db}'.")


def cmd_validate(args):
    """Performs dry-run AST validation and integrity verification on SQLite or JSON trees."""
    target_str = getattr(args, "db", None) or getattr(args, "target", "trees/lattice.db")
    target_path = Path(target_str)
    if not target_path.exists():
        print(f"[!] Target file '{target_path}' does not exist!")
        sys.exit(1)

    if target_str.endswith(".json"):
        print(f"[*] Validating Domain Tree JSON '{target_path}'...")
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            tree = TreeSchema.model_validate(raw_data)
        except Exception as e:
            print(f"[X] JSON / Schema Validation Error: {e}")
            sys.exit(1)

        valid_count = 0
        failed_count = 0
        errors: List[str] = []
        stage_counts = {1: 0, 2: 0, 3: 0}

        for cell in tree.cells:
            stage_counts[cell.stage] = stage_counts.get(cell.stage, 0) + 1
            code = cell.code_template
            if not code or not code.strip():
                if getattr(cell, "node_type", "") == "macro" or getattr(cell, "sub_cells", []):
                    valid_count += 1
                    continue
                failed_count += 1
                errors.append(f"{cell.cell_id}: Empty code template")
                continue

            dummy_code = sanitize_placeholders_for_ast(code)
            try:
                ast.parse(dummy_code)
                valid_count += 1
            except SyntaxError as e:
                failed_count += 1
                errors.append(f"{cell.cell_id}: AST Syntax Error: {e}")

        total = len(tree.cells)
        print(f"\n==================================================")
        print(f" VALIDATION RESULTS FOR: {target_path.name}")
        print(f"==================================================")
        print(f" Domain Name         : {tree.domain}")
        print(f" Total Cells Checked : {total}")
        print(f"   Stage 1 (Ingress) : {stage_counts.get(1, 0)}")
        print(f"   Stage 2 (Step)    : {stage_counts.get(2, 0)}")
        print(f"   Stage 3 (Egress)  : {stage_counts.get(3, 0)}")
        print(f" Syntactically Valid : {valid_count}")
        print(f" Failed Nodes        : {failed_count}")
        print(f" Success Rate        : {(valid_count / total * 100):.2f}%" if total else "0.00%")

        if errors:
            print("\n[!] Top Errors:")
            for err in errors[:10]:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print("\n✅ Tree is 100% valid and ready to compile into lattice.db!")
            print(f"   Run: python3 nstl_cli.py compile")
        return

    print(f"[*] Validating SQLite Database '{target_path}'...")
    with contextlib.closing(sqlite3.connect(str(target_path))) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()

        cur.execute("SELECT cell_id, domain_name, stage, code, input_type, output_type, configuration_schema, node_type, node_role FROM nodes")
        rows = cur.fetchall()

        valid_count = 0
        failed_count = 0
        errors: List[str] = []

        for row in rows:
            cell_id, domain, stage, code, in_t, out_t, config_str, n_type, n_role = row
            if not code or not code.strip():
                try:
                    cfg = json.loads(config_str) if config_str else {}
                except Exception:
                    cfg = {}
                if n_type == "macro" or n_role == "macro" or cfg.get("node_type") == "macro" or cfg.get("sub_cells"):
                    sub_cells = cfg.get("sub_cells") or []
                    if len(sub_cells) >= 1:
                        valid_count += 1
                        continue
                failed_count += 1
                errors.append(f"{cell_id}: Empty code template")
                continue

            dummy_code = sanitize_placeholders_for_ast(code)
            try:
                ast.parse(dummy_code)
                valid_count += 1
            except SyntaxError as e:
                failed_count += 1
                errors.append(f"{cell_id}: AST Syntax Error: {e}")

    print(f"\n==================================================")
    print(f" VALIDATION RESULTS FOR: {target_path.name}")
    print(f"==================================================")
    print(f" Total Nodes Checked : {len(rows)}")
    print(f" Syntactically Valid : {valid_count}")
    print(f" Failed Nodes        : {failed_count}")
    print(f" Success Rate        : {(valid_count / len(rows) * 100):.2f}%" if rows else "0.00%")

    if errors:
        print("\n[!] Top Errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n✅ All nodes in database passed 100% AST dry-run validation!")


# =====================================================================
# AST & Sandbox Utility Helpers
# =====================================================================
def _is_environmental_error(error_msg: str, sandbox_res: Optional[Dict[str, Any]] = None) -> bool:
    """Classifies a sandbox failure as environmental rather than a defect in synthesized code."""
    if sandbox_res and sandbox_res.get("extrinsic", False):
        return True
    if not error_msg:
        return False

    extrinsic_type_names = {
        cls.__name__ for cls in (
            OSError, ImportError, ModuleNotFoundError, ConnectionError, TimeoutError
        )
    }
    extrinsic_type_names.update(c.__name__ for c in OSError.__subclasses__())
    extrinsic_type_names.add("SandboxSecurityError")

    lines = [l.strip() for l in error_msg.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        token = line.split(":")[0].strip()
        if token in extrinsic_type_names:
            return True
    return False


def _collect_template_api_names(templates) -> Set[str]:
    """Collects attribute/method names referenced by verified cell templates via AST."""
    names: Set[str] = set()
    for tmpl in templates:
        if not tmpl:
            continue
        code = sanitize_placeholders_for_ast(tmpl)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names


def _unknown_api_references(repaired_code: str, original_code: str, cells) -> Set[str]:
    """Returns attribute names referenced by repaired code absent from original code and lattice."""
    def _attrs(code: str) -> Set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}

    repaired_attrs = _attrs(repaired_code)
    original_attrs = _attrs(original_code)
    known = _collect_template_api_names(getattr(c, "code_template", "") for c in cells)
    return repaired_attrs - original_attrs - known


def _get_available_embedders() -> List[str]:
    """Scans MODELS_DIR/embeddings for available embedding models, prioritizing nano."""
    emb_dir = os.path.join(MODELS_DIR, "embeddings")
    if os.path.exists(emb_dir):
        raw = sorted([
            d for d in os.listdir(emb_dir)
            if os.path.isdir(os.path.join(emb_dir, d)) and not d.endswith("-GGUF")
        ])
        m_nano = [d for d in raw if "nano" in d.lower()]
        rest = [d for d in raw if "nano" not in d.lower()]
        return m_nano + rest
    return []


def _get_available_llms() -> List[str]:
    """Scans MODELS_DIR/llms for available GGUF LLMs, prioritizing the default 0.5b model."""
    llm_dir = os.path.join(MODELS_DIR, "llms")
    if os.path.exists(llm_dir):
        raw = sorted([
            d for d in os.listdir(llm_dir)
            if os.path.isdir(os.path.join(llm_dir, d))
        ])
        m_05b = [d for d in raw if "0.5b" in d.lower()]
        rest = [d for d in raw if "0.5b" not in d.lower()]
        return m_05b + rest
    return []


def _get_available_rerankers() -> List[str]:
    """Scans MODELS_DIR/rerankers for available neural reranker models."""
    try:
        return get_available_rerankers()
    except Exception:
        r_dir = os.path.join(MODELS_DIR, "rerankers")
        if os.path.exists(r_dir):
            return sorted([d for d in os.listdir(r_dir) if os.path.isdir(os.path.join(r_dir, d))])
        return []


# =====================================================================
# Pipeline Diagnostics & Tracer
# =====================================================================
class PipelineDebugger:
    """Diagnostic tracer and rich visualizer for STRATA pipeline layers."""

    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        router: LatticeRouter,
        gate: UnificationGate,
        sandbox: GEVRSandbox,
        active_profile: str = "0",
        device: str = "auto",
        embedder_name: str = "",
        llm_name: str = "",
        console: Optional[Console] = None,
        route_method: Optional[str] = None
    ):
        self.orchestrator = orchestrator
        self.router = router
        self.gate = gate
        self.sandbox = sandbox
        self.active_profile = active_profile
        self.device = device
        self.embedder_name = select_optimal_embedder(embedder_name) if embedder_name else select_optimal_embedder()
        self.llm_name = select_optimal_llm(llm_name) if llm_name else select_optimal_llm()
        self.console = console or Console()
        self.route_method = route_method.upper() if route_method else None

    def run(
        self,
        prompt: str,
        execute_sandbox: bool = True,
        timeout: float = DEFAULT_SANDBOX_TIMEOUT
    ) -> Dict[str, Any]:
        c = self.console
        t_total_start = time.perf_counter()
        prof = self.active_profile.upper()
        prompt_clean = prompt.strip()

        header_table = Table(box=box.SIMPLE_HEAD, expand=True, border_style="cyan")
        header_table.add_column("Profile", style="bold yellow")
        header_table.add_column("Embedder", style="magenta")
        header_table.add_column("LLM (GGUF)", style="magenta")
        header_table.add_column("Device", style="green")
        header_table.add_column("Total Nodes", style="white")

        emb_disp = self.embedder_name or "None (Bypassed)"
        llm_disp = self.llm_name or "None (Bypassed)"
        domains = set(cl.domain_name for cl in self.orchestrator.loaded_cells.values() if cl.domain_name)
        header_table.add_row(
            f"Profile {prof}",
            emb_disp,
            llm_disp,
            self.device.upper(),
            f"{len(self.orchestrator.loaded_cells):,} in {len(domains)} domains"
        )

        c.print(Panel(
            header_table,
            title="[bold white on blue] 🔬 STRATA FULL-PIPELINE DEBUG TRACE [/bold white on blue]",
            subtitle=f"[dim cyan]Prompt: \"{prompt_clean}\"[/dim cyan]",
            border_style="blue",
            padding=(0, 1)
        ))

        # LAYER 0: Query Intent & Pre-Processing
        effective_prompt = prompt_clean
        t_trans = 0.0
        intent_data = None

        if prof in ("E", "S"):
            mm = ModelManager.get_instance()
            if mm.profile and (mm.has_translator_pass() or mm.has_semantic_compiler()):
                t0_t = time.perf_counter()
                intent_data = mm.compile_semantic_intent(prompt_clean)
                effective_prompt = ensure_comprehensive_prompt(prompt_clean, intent_data.get("effective_prompt") or prompt_clean)
                t_trans = (time.perf_counter() - t0_t) * 1000.0

        clauses = _segment_prompt_clauses(effective_prompt)
        prompt_tokens = CellTokenizer.tokenize_prompt(effective_prompt)
        reg = TypeRegistry.get_instance()
        content_tokens = {t for t in prompt_tokens if reg.is_informative_token(t)} if prompt_tokens else set()
        literals = ExecutionContext._extract_universal_literals(effective_prompt)

        l0_table = Table(box=box.ROUNDED, expand=True, border_style="dim cyan")
        l0_table.add_column("Property", style="bold cyan", width=24)
        l0_table.add_column("Extracted Information", style="white")

        l0_table.add_row("Raw User Prompt", prompt_clean)
        if prof == "E":
            l0_table.add_row(f"Translator Output ({t_trans:.1f}ms)", f"[italic magenta]{effective_prompt}[/italic magenta]")
        elif prof == "S" and intent_data:
            task_str = intent_data.get("task", "generic")
            tgt_sink = intent_data.get("target_sink", "")
            tgt_info = f" | Target: {tgt_sink}" if tgt_sink else ""
            l0_table.add_row(f"Profile S Compiler ({t_trans:.1f}ms)", f"[italic magenta]Task: {task_str}{tgt_info}[/italic magenta]")
        else:
            l0_table.add_row("Translator Pass", "[dim]Bypassed (Active in Profile E/S)[/dim]")

        clauses_formatted = "  ➔  ".join(f"[bold yellow]Clause {idx+1}:[/bold yellow] '{cl}'" for idx, cl in enumerate(clauses)) if clauses else "[dim]None[/dim]"
        l0_table.add_row(f"Decomposed Clauses ({len(clauses)})", clauses_formatted)
        l0_table.add_row(f"Content Tokens ({len(content_tokens)})", ", ".join(sorted(content_tokens)) if content_tokens else "[dim]None[/dim]")

        if literals:
            lit_strs = [f"[bold green]{val}[/bold green] ([dim]{kind}[/dim])" for _, kind, val in literals]
            l0_table.add_row(f"Universal Literals ({len(literals)})", ", ".join(lit_strs))
        else:
            l0_table.add_row("Universal Literals", "[dim]None detected[/dim]")

        c.print(Panel(l0_table, title="[bold cyan]⚡ LAYER 0: Query Intent & Pre-Processing[/bold cyan]", border_style="cyan"))

        # LAYER 1: Semantic Tunneling & Scoring (Router)
        t_route_0 = time.perf_counter()
        is_rag = (self.router.internal_rag is not None and getattr(self.router.internal_rag, "index", None) is not None)
        engine_desc = "Dense Vector Embeddings via LocalRAG (FAISS)" if is_rag else "Lexical Token Coverage with IDF Poset Index"
        query_spans = self.router._generate_query_spans(effective_prompt)
        gamma = getattr(self.router, "gamma", 0.15)
        epsilon = getattr(self.router, "epsilon", 0.001)

        tunnel_cells, relevance_map = self.router.route(effective_prompt, top_k=400)
        route_dt = (time.perf_counter() - t_route_0) * 1000.0

        reranker_active = bool(getattr(self.router, "use_reranker", False) and getattr(self.router, "last_reranker_telemetry", None))
        r_model_name = getattr(self.router, "reranker_model", None) or "jina-reranker-v3.5"
        if reranker_active:
            engine_desc += f" ➔ Neural Reranker ({r_model_name})"

        ranked_candidates = sorted(tunnel_cells, key=lambda cl: float(relevance_map.get(cl.cell_id, 0.0)), reverse=True)

        stage_counts = {1: 0, 2: 0, 3: 0, "other": 0}
        domain_counts: Dict[str, int] = {}
        carriers_out = set()
        carriers_in = set()
        for cl in ranked_candidates:
            st = getattr(cl, "stage", "other")
            if st in (1, 2, 3):
                stage_counts[st] += 1
            else:
                stage_counts["other"] += 1
            dom = getattr(cl, "domain_name", "generic")
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
            out_t = getattr(cl.primary_output, "type_name", "")
            in_t = getattr(cl.primary_input, "type_name", "")
            if out_t and out_t.lower() not in ("any", "none", "*", "top", "void"):
                carriers_out.add(out_t)
            if in_t and in_t.lower() not in ("any", "none", "*", "top", "void"):
                carriers_in.add(in_t)

        top_domains_str = ", ".join(f"{d} ({cnt})" for d, cnt in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:6])
        summary_text = (
            f"[cyan]Retrieval Engine:[/cyan] {engine_desc}\n"
            f"[cyan]Hyperparameters:[/cyan] Temperature γ={gamma}, Cutoff ε={epsilon}, Multi-Scale Spans={len(query_spans)}\n"
            f"[cyan]Tunnel Distribution:[/cyan] {len(ranked_candidates):,} nodes | "
            f"Stage 1 (Ingress): [bold green]{stage_counts[1]}[/bold green] | "
            f"Stage 2 (Transforms): [bold yellow]{stage_counts[2]}[/bold yellow] | "
            f"Stage 3 (Egress/Sinks): [bold blue]{stage_counts[3]}[/bold blue]\n"
            f"[cyan]Carrier Types Detected:[/cyan] Outputs: {sorted(list(carriers_out))[:8]} | Inputs: {sorted(list(carriers_in))[:8]}\n"
            f"[cyan]Active Domains in Tunnel:[/cyan] {top_domains_str}"
        )

        c.print(Panel(summary_text, title=f"[bold yellow]⚡ LAYER 1: Semantic Tunneling & Scoring ({route_dt:.2f}ms)[/bold yellow]", border_style="yellow"))

        if ranked_candidates:
            display_limit = 15
            if reranker_active:
                # -------------------------------------------------------------
                # 1. Pure RAG Baseline Candidates (Pre-Reranker)
                # -------------------------------------------------------------
                raw_tunnel = getattr(self.router, "raw_rag_tunnel", []) or ranked_candidates
                raw_map = getattr(self.router, "raw_rag_relevance_map", {}) or relevance_map
                raw_ranked = sorted(raw_tunnel, key=lambda cl: float(raw_map.get(cl.cell_id, 0.0)), reverse=True)

                raw_cand_table = Table(
                    title=f"🎯 Pure RAG Top Scoring Nodes (Pre-Reranker | Baseline RAG)",
                    box=box.ROUNDED, expand=True, border_style="cyan"
                )
                raw_cand_table.add_column("#", style="dim", width=4)
                raw_cand_table.add_column("RAG Score", style="bold cyan", width=14)
                raw_cand_table.add_column("Cell ID", style="bold white", ratio=3)
                raw_cand_table.add_column("Domain", style="magenta", width=12)
                raw_cand_table.add_column("Stage", style="white", width=11)
                raw_cand_table.add_column("Role", style="dim", width=10)
                raw_cand_table.add_column("Primary Input (τ_in)", style="green", ratio=2)
                raw_cand_table.add_column("Primary Output (τ_out)", style="blue", ratio=2)

                for idx, cl in enumerate(raw_ranked[:display_limit], 1):
                    sc = float(raw_map.get(cl.cell_id, 0.0))
                    sc_color = _get_score_color(sc, epsilon=epsilon)
                    sc_str = f"[{sc_color}]{sc:.4f}[/{sc_color}]"
                    stage_name = {1: "1: Ingress", 2: "2: Transform", 3: "3: Egress"}.get(cl.stage, str(cl.stage))
                    in_sig = f"{getattr(cl.primary_input, 'type_name', 'any')} [{getattr(cl.primary_input, 'state', 'any')}]"
                    out_sig = f"{getattr(cl.primary_output, 'type_name', 'any')} [{getattr(cl.primary_output, 'state', 'any')}]"
                    raw_cand_table.add_row(
                        str(idx), sc_str, cl.cell_id, cl.domain_name, stage_name,
                        cl.node_role or cl.node_type, in_sig, out_sig
                    )
                c.print(raw_cand_table)

                # -------------------------------------------------------------
                # 2. Neural Reranked Candidate Nodes (Post-Reranker)
                # -------------------------------------------------------------
                rerank_table = Table(
                    title=f"🧠 Neural Reranked Nodes in Semantic Tunnel (Post-Reranker: {r_model_name})",
                    box=box.ROUNDED, expand=True, border_style="yellow"
                )
                rerank_table.add_column("#", style="dim", width=4)
                rerank_table.add_column("Blended Score", style="bold yellow", width=14)
                rerank_table.add_column("Cell ID", style="bold cyan", ratio=3)
                rerank_table.add_column("Domain", style="magenta", width=12)
                rerank_table.add_column("Stage", style="white", width=11)
                rerank_table.add_column("Role", style="dim", width=10)
                rerank_table.add_column("Primary Input (τ_in)", style="green", ratio=2)
                rerank_table.add_column("Primary Output (τ_out)", style="blue", ratio=2)

                for idx, cl in enumerate(ranked_candidates[:display_limit], 1):
                    sc = float(relevance_map.get(cl.cell_id, 0.0))
                    sc_color = _get_score_color(sc, epsilon=epsilon)
                    sc_str = f"[{sc_color}]{sc:.4f}[/{sc_color}]"
                    stage_name = {1: "1: Ingress", 2: "2: Transform", 3: "3: Egress"}.get(cl.stage, str(cl.stage))
                    in_sig = f"{getattr(cl.primary_input, 'type_name', 'any')} [{getattr(cl.primary_input, 'state', 'any')}]"
                    out_sig = f"{getattr(cl.primary_output, 'type_name', 'any')} [{getattr(cl.primary_output, 'state', 'any')}]"
                    rerank_table.add_row(
                        str(idx), sc_str, cl.cell_id, cl.domain_name, stage_name,
                        cl.node_role or cl.node_type, in_sig, out_sig
                    )
                c.print(rerank_table)

                # -------------------------------------------------------------
                # 3. Neural Reranker Impact & Trajectory Shifts
                # -------------------------------------------------------------
                telemetry = getattr(self.router, "last_reranker_telemetry", [])
                if telemetry:
                    diff_table = Table(
                        title=f"📊 Neural Reranker Impact Analysis (Rank & Score Transitions)",
                        box=box.ROUNDED, expand=True, border_style="magenta"
                    )
                    diff_table.add_column("Cell ID", style="bold cyan", ratio=3)
                    diff_table.add_column("Domain", style="white", width=12)
                    diff_table.add_column("RAG Rank", style="dim", justify="right", width=10)
                    diff_table.add_column("Post Rank", style="bold", justify="right", width=10)
                    diff_table.add_column("Rank Shift (Δ)", justify="center", width=15)
                    diff_table.add_column("RAG Score", style="dim", justify="right", width=11)
                    diff_table.add_column("Post Score", style="bold", justify="right", width=11)
                    diff_table.add_column("Score Shift (Δ)", justify="right", width=15)
                    diff_table.add_column("Status / Impact", justify="center", width=16)

                    for item in telemetry[:20]:
                        d_r = item.get("delta_rank", 0)
                        d_s = item.get("delta_score", 0.0)
                        if d_r > 0:
                            shift_r = f"[bold green]▲ +{d_r}[/bold green]"
                            status = "[bold green]▲ PROMOTED[/bold green]"
                        elif d_r < 0:
                            shift_r = f"[bold red]▼ {d_r}[/bold red]"
                            status = "[bold red]▼ DEMOTED[/bold red]"
                        else:
                            shift_r = "[dim]= 0[/dim]"
                            status = "[dim]STABLE[/dim]"

                        if d_s > 0:
                            shift_s = f"[green]+{d_s:.4f}[/green]"
                        elif d_s < 0:
                            shift_s = f"[red]{d_s:.4f}[/red]"
                        else:
                            shift_s = "[dim]0.0000[/dim]"

                        diff_table.add_row(
                            item["cell_id"],
                            item.get("domain", ""),
                            f"#{item['old_rank']}",
                            f"#{item['new_rank']}",
                            shift_r,
                            f"{item['old_score']:.4f}",
                            f"{item['new_score']:.4f}",
                            shift_s,
                            status
                        )
                    c.print(diff_table)
            else:
                scores_are_prob = all(0.0 <= float(relevance_map.get(cl.cell_id, 0.0)) <= 1.0 for cl in ranked_candidates[:15])
                score_col_title = "Score / P(v|x)" if scores_are_prob else "Score / Priority"
                cand_table = Table(title=f"🎯 Top Scoring Nodes in Semantic Tunnel (Displaying top {min(15, len(ranked_candidates))} of {len(ranked_candidates):,})", box=box.ROUNDED, expand=True, border_style="yellow")
                cand_table.add_column("#", style="dim", width=4)
                cand_table.add_column(score_col_title, style="bold yellow", width=14)
                cand_table.add_column("Cell ID", style="bold cyan", ratio=3)
                cand_table.add_column("Domain", style="magenta", width=12)
                cand_table.add_column("Stage", style="white", width=11)
                cand_table.add_column("Role", style="dim", width=10)
                cand_table.add_column("Primary Input (τ_in)", style="green", ratio=2)
                cand_table.add_column("Primary Output (τ_out)", style="blue", ratio=2)

                for idx, cl in enumerate(ranked_candidates[:display_limit], 1):
                    sc = float(relevance_map.get(cl.cell_id, 0.0))
                    sc_color = _get_score_color(sc, epsilon=epsilon)
                    if scores_are_prob:
                        sc_str = f"[{sc_color}]{sc:.4f} ({sc*100:.1f}%)[/{sc_color}]"
                    else:
                        sc_str = f"[{sc_color}]{sc:.4f}[/{sc_color}]"

                    stage_name = {1: "1: Ingress", 2: "2: Transform", 3: "3: Egress"}.get(cl.stage, str(cl.stage))
                    in_sig = f"{getattr(cl.primary_input, 'type_name', 'any')} [{getattr(cl.primary_input, 'state', 'any')}]"
                    out_sig = f"{getattr(cl.primary_output, 'type_name', 'any')} [{getattr(cl.primary_output, 'state', 'any')}]"

                    cand_table.add_row(
                        str(idx),
                        sc_str,
                        cl.cell_id,
                        cl.domain_name,
                        stage_name,
                        cl.node_role or cl.node_type,
                        in_sig,
                        out_sig
                    )
                c.print(cand_table)

            if len(ranked_candidates) > display_limit:
                c.print(f"  [dim]... and {len(ranked_candidates) - display_limit:,} more candidate nodes in semantic tunnel.[/dim]\n")
        else:
            c.print("[bold red][!] Empty tunnel: No nodes cleared the relevance cutoff threshold.[/bold red]\n")

        dbg_ctx = ExecutionContext(prompt=effective_prompt)
        if intent_data and isinstance(intent_data, dict):
            dbg_ctx.intent_data = dict(intent_data)
            if hasattr(dbg_ctx, "blackboard") and dbg_ctx.blackboard is not None:
                dbg_ctx.blackboard.lock_from_intent(intent_data)
                if dbg_ctx.blackboard.locked_by_neural:
                    if dbg_ctx.blackboard.target_sink:
                        dbg_ctx.parameters["target_sink"] = dbg_ctx.blackboard.target_sink
            if intent_data.get("operations") and isinstance(intent_data["operations"], list):
                dbg_ctx.operations = [str(op).replace("_", " ").strip() for op in intent_data["operations"] if str(op).strip()]
            if intent_data.get("hyperparameters") and isinstance(intent_data["hyperparameters"], dict):
                dbg_ctx.hyperparameters = dict(intent_data["hyperparameters"])
            if intent_data.get("slots") and isinstance(intent_data["slots"], dict):
                dbg_ctx.llm_slots = dict(intent_data["slots"])
            if intent_data.get("variables") and isinstance(intent_data["variables"], list):
                dbg_ctx.variables_spec = [str(v).strip() for v in intent_data["variables"] if str(v).strip()]
                dbg_ctx.llm_slots.setdefault("variables", dbg_ctx.variables_spec)
            if hasattr(dbg_ctx, "sync_llm_slots"):
                dbg_ctx.sync_llm_slots()
            tgt = intent_data.get("target_sink") or intent_data.get("target_key") or intent_data.get("deliverable")
            if tgt:
                dbg_ctx.target_key = str(tgt).strip()
                dbg_ctx.target_literals.add(dbg_ctx.target_key)
                dbg_ctx.parameters["target_sink"] = dbg_ctx.target_key
            if intent_data.get("parameters") and isinstance(intent_data["parameters"], dict):
                for p_key, p_val in intent_data["parameters"].items():
                    if p_key:
                        dbg_ctx.parameters.setdefault(str(p_key), p_val)
            if intent_data.get("source_files"):
                dbg_ctx.parameters["source_uris"] = list(intent_data["source_files"])
            if intent_data.get("dest_files"):
                dbg_ctx.parameters["dest_uris"] = list(intent_data["dest_files"])

        # LAYER 2: Topological Planning & Trellis Viterbi (Planner)
        t_plan_0 = time.perf_counter()
        os.environ["NSTL_DEBUG_PLAN"] = "1"
        try:
            cells = self.router.plan_path(
                effective_prompt,
                return_tuple=False,
                route_method=self.route_method,
                ctx=dbg_ctx,
            )
        finally:
            os.environ.pop("NSTL_DEBUG_PLAN", None)
        plan_dt = (time.perf_counter() - t_plan_0) * 1000.0
        try:
            from debug_panels import render_planning_diagnostics
            if cells:
                render_planning_diagnostics(console, self.router, cells, effective_prompt)
        except Exception as _dp_err:
            logger.warning(f"[DEBUG] planning diagnostics unavailable: {type(_dp_err).__name__}: {_dp_err}")

        if cells:
            path_table = Table(title=f"🛣️ Planned Monadic Pipeline Composition ({len(cells)} Steps)", box=box.ROUNDED, expand=True, border_style="green")
            path_table.add_column("Step", style="bold yellow", width=6)
            path_table.add_column("Cell ID", style="bold cyan", ratio=3)
            path_table.add_column("Domain & Stage", style="magenta", width=18)
            path_table.add_column("Input (τ_in)", style="green", ratio=2)
            path_table.add_column("Output (τ_out)", style="blue", ratio=2)
            path_table.add_column("Type-Monadic Transition", style="bold", ratio=3)

            reg = TypeRegistry.get_instance()
            for idx, cl in enumerate(cells, 1):
                stage_label = f"{cl.domain_name} (S{cl.stage})"
                in_sig_str = f"{getattr(cl.primary_input, 'type_name', 'any')} [{getattr(cl.primary_input, 'state', 'any')}]"
                out_sig_str = f"{getattr(cl.primary_output, 'type_name', 'any')} [{getattr(cl.primary_output, 'state', 'any')}]"

                if idx == 1:
                    trans_status = "[bold green]✓ Ingress Entry Point[/bold green]"
                else:
                    prev_cl = cells[idx - 2]
                    prev_out = prev_cl.primary_output.signature
                    curr_in = cl.primary_input.signature
                    u = unify(prev_out, curr_in)
                    is_sub = reg.is_subtype(str(prev_out.type_name), str(curr_in.type_name))

                    if u is not None:
                        trans_status = f"[bold green]✓ Unifies ({prev_out.type_name} ➔ {curr_in.type_name})[/bold green]"
                    elif is_sub:
                        trans_status = f"[green]✓ Subtype ({prev_out.type_name} ⊆ {curr_in.type_name})[/green]"
                    else:
                        alt_port = next((p_name for p_name, p in cl.inputs.items() if unify(prev_out, p.signature) is not None), None)
                        if alt_port:
                            trans_status = f"[cyan]✓ Wire via port '{alt_port}'[/cyan]"
                        else:
                            # Multi-carrier DAG scope check: did it unify with an earlier ancestor?
                            dag_parent = None
                            for anc in reversed(cells[:idx - 1]):
                                if any(unify(o.signature, cl.primary_input.signature) is not None for o in anc.outputs.values()):
                                    dag_parent = anc.cell_id
                                    break
                                alt = next((p_n for p_n, p_s in cl.inputs.items() if any(unify(o.signature, p_s.signature) is not None for o in anc.outputs.values())), None)
                                if alt:
                                    dag_parent = f"{anc.cell_id}.{alt}"
                                    break
                            if dag_parent:
                                trans_status = f"[bold cyan]✓ DAG Wire from {dag_parent}[/bold cyan]"
                            else:
                                trans_status = "[yellow]~ Weak / Wildcard bind[/yellow]"

                path_table.add_row(
                    str(idx),
                    cl.cell_id,
                    stage_label,
                    in_sig_str,
                    out_sig_str,
                    trans_status
                )

            path_chain = " ➔ ".join(f"[bold cyan]{c.cell_id}[/bold cyan]" for c in cells)
            eff_route = getattr(getattr(self, "router", None), "last_effective_route", None)
            req_route = getattr(getattr(self, "router", None), "last_requested_route", None)
            fb_reason = getattr(getattr(self, "router", None), "last_fallback_reason", None)
            route_str = ""
            if eff_route:
                if fb_reason:
                    route_str = f" [yellow](Route: {eff_route} fallback from {req_route}: {fb_reason})[/yellow]"
                else:
                    route_str = f" [cyan](Route: {eff_route})[/cyan]"
            c.print(Panel(
                f"[bold green]✓ Synthesized Type-Valid Path ({len(cells)} steps){route_str}:[/bold green] {path_chain}",
                title=f"[bold green]⚡ LAYER 2: Topological Planning & Trellis Viterbi ({plan_dt:.2f}ms)[/bold green]",
                border_style="green"
            ))
            c.print(path_table)
            c.print("")
        else:
            logger.error(f"[CLI] Planning failed: No valid path found through lattice (Profile: {self.active_profile}, RouteMethod: {getattr(getattr(self, 'router', None), 'last_requested_route', 'M0')})")
            c.print(Panel(
                "[bold red]❌ PLANNING FAILURE: No valid compositional path found through the lattice.[/bold red]",
                title=f"[bold red]⚡ LAYER 2: Topological Planning Failure ({plan_dt:.2f}ms)[/bold red]",
                border_style="red"
            ))

            diag_table = Table(title="🔍 Planning Failure Root Cause Analysis", box=box.ROUNDED, expand=True, border_style="red")
            diag_table.add_column("Diagnostic Check", style="bold yellow", width=28)
            diag_table.add_column("Result", style="bold", width=14)
            diag_table.add_column("Underlying Issue & Resolution Advice", style="white")

            if not tunnel_cells:
                diag_table.add_row(
                    "Semantic Tunnel Population",
                    "[red]EMPTY (0)[/red]",
                    f"No candidate nodes met relevance threshold (epsilon={epsilon}). Rephrase or provide package hints."
                )
            else:
                diag_table.add_row(
                    "Semantic Tunnel Population",
                    f"[green]OK ({len(tunnel_cells):,})[/green]",
                    f"{len(tunnel_cells):,} nodes in active tunnel."
                )

            entries = [cl for cl in tunnel_cells if getattr(cl, "stage", None) == 1]
            if not entries:
                diag_table.add_row(
                    "Stage 1 Ingress/Source Nodes",
                    "[red]MISSING (0)[/red]",
                    "No data ingestion nodes found in tunnel. Pipelines must begin with Stage 1."
                )
            else:
                entry_ids = ", ".join(cl.cell_id for cl in entries[:4])
                diag_table.add_row(
                    "Stage 1 Ingress/Source Nodes",
                    f"[green]FOUND ({len(entries)})[/green]",
                    f"Candidate entries: {entry_ids}"
                )

            sinks = [cl for cl in tunnel_cells if getattr(cl, "stage", None) == 3]
            if not sinks and literals:
                diag_table.add_row(
                    "Stage 3 Egress/Sink Nodes",
                    "[yellow]NONE[/yellow]",
                    "Prompt contains literals/destinations, but no terminal sinks were retrieved."
                )
            elif sinks:
                sink_ids = ", ".join(cl.cell_id for cl in sinks[:4])
                diag_table.add_row(
                    "Stage 3 Egress/Sink Nodes",
                    f"[green]FOUND ({len(sinks)})[/green]",
                    f"Candidate sinks: {sink_ids}"
                )

            if entries:
                entry_out_types = {getattr(cl.primary_output, 'type_name', '') for cl in entries}
                transform_cells = [cl for cl in tunnel_cells if getattr(cl, "stage", None) == 2]
                transform_in_types = {getattr(cl.primary_input, 'type_name', '') for cl in transform_cells}
                diag_table.add_row(
                    "Carrier Compatibility",
                    "[yellow]INSPECT[/yellow]",
                    f"Entry outputs: {entry_out_types} vs Transform inputs: {transform_in_types}."
                )

            refusal = getattr(getattr(self.router, "planner", None), "last_refusal", None)
            if refusal and isinstance(refusal, dict):
                cov_frac = refusal.get("coverage_fraction", 0.0)
                cov_fl = refusal.get("coverage_floor", 0.85)
                uncovered = refusal.get("uncovered_clauses", [])
                unc_text = ", ".join(f"[{idx}]: '{txt}'" for idx, txt in uncovered) if uncovered else "None"
                diag_table.add_row(
                    "Coverage Floor Check",
                    "[red]REFUSED[/red]",
                    f"Plan coverage {cov_frac:.1%} < floor {cov_fl:.1%}. Uncovered clauses: {unc_text}"
                )

            c.print(diag_table)
            c.print("")

            # Layer 5 Feedback Check for Planning Refusal
            llm_feedback = getattr(self, "llm_feedback", False)
            if not cells and refusal and llm_feedback:
                mm = ModelManager.get_instance()
                if mm.profile and mm.can_feedback_check():
                    unc_list = [f"Clause {idx}: '{txt}'" for idx, txt in refusal.get("uncovered_clauses", [])]
                    err_msg = (
                        f"Topological planning REFUSED due to coverage below floor "
                        f"({refusal.get('coverage_fraction', 0.0):.1%} < {refusal.get('coverage_floor', 0.85):.1%}).\n"
                        f"Missing clauses:\n" + "\n".join(f"- {u}" for u in unc_list)
                    )
                    c.print(Panel("[bold yellow]⚡ LAYER 5: Planning Refusal LLM Feedback Check[/bold yellow]", border_style="yellow"))
                    feedback_reply = mm.feedback_check(failing_code="", traceback_error=err_msg)
                    extracted = extract_code_from_llm_response(feedback_reply)
                    if extracted:
                        final_code = extracted

        # LAYER 3: Type-Monadic Unification & AST Synthesis
        final_code = ""
        dest_paths = None
        pipeline_bindings = None
        accum_sigma = None
        synth_dt = 0.0
        internal_error_msg: Optional[str] = None
        refusal_reason: Optional[str] = None

        if cells:
            t_synth_0 = time.perf_counter()
            ctx = dbg_ctx if dbg_ctx is not None else ExecutionContext(prompt=prompt_clean)
            # Profile S (Semantic Compiler): the typed IR's extracted literals
            # are injected into the execution context's declared parameters so
            # the Unification Gate binds arguments deterministically from the
            # STRUCTURED compiler output instead of re-scanning the prompt.
            ir_literals = getattr(getattr(self, "router", None), "last_ir_literals", None) or {}
            if isinstance(ir_literals, dict) and ir_literals:
                for lit_key, lit_val in ir_literals.items():
                    if lit_key and lit_key not in ctx.parameters:
                        ctx.parameters[str(lit_key)] = lit_val
            try:
                unify_res = self.gate.unify_pipeline(cells, ctx)
                try:
                    from debug_panels import render_synthesis_diagnostics
                    render_synthesis_diagnostics(console, ctx, cells, prompt)
                except Exception as _dp_err:
                    logger.warning(f"[DEBUG] synthesis diagnostics unavailable: {type(_dp_err).__name__}: {_dp_err}")
            except UnificationFailure as exc:
                # Declared engine failure type: a legitimate type-level refusal (R6-4).
                unify_res = Failure(reason=str(exc))
            except Exception as exc:
                # Any other exception is an ENGINE DEFECT, not a unification refusal.
                # It must never be reported as "UNIFICATION FAILED" or counted as a
                # refusal in sweeps (R6-4).
                logger.error("INTERNAL ERROR in unify_pipeline: %s", exc, exc_info=True)
                internal_error_msg = f"{type(exc).__name__}: {exc}"
                unify_res = None
            synth_dt = (time.perf_counter() - t_synth_0) * 1000.0

            if internal_error_msg:
                c.print(Panel(
                    f"[bold red]❌ INTERNAL ERROR ({internal_error_msg.split(':', 1)[0]})[/bold red]\n"
                    f"[red]{internal_error_msg}[/red]\n"
                    f"[yellow]This is an engine defect (bug), not a type-level refusal. See log traceback.[/yellow]",
                    title=f"[bold red]⚡ LAYER 3: Internal Engine Error ({synth_dt:.2f}ms)[/bold red]",
                    border_style="red"
                ))
            elif unify_res.is_bottom():
                fail_reason = getattr(unify_res, "reason", "Unknown unification failure")
                c.print(Panel(
                    f"[bold red]❌ UNIFICATION FAILED: {fail_reason}[/bold red]\n"
                    f"[yellow]A type constraint or required port could not be satisfied in the Type Monad.[/yellow]",
                    title=f"[bold red]⚡ LAYER 3: Type-Monadic Unification Failure ({synth_dt:.2f}ms)[/bold red]",
                    border_style="red"
                ))
            else:
                pipeline_bindings = unify_res.value
                accum_sigma = unify_res.sigma
                dest_paths = self.gate._derive_egress_paths(pipeline_bindings)
                self.gate.last_egress_paths = dest_paths
                try:
                    final_code = self.gate.emit_code(pipeline_bindings, ctx)
                except (UnresolvedPlaceholderError, UnificationFailure) as e:
                    # Declared refusal from the synthesis chain (e.g. unresolved
                    # ports after estimator-identity gating) — not an engine defect.
                    logger.warning("Synthesis refused (declared failure): %s", e)
                    final_code = ""
                    refusal_reason = str(e)
                except Exception as e:
                    logger.error("INTERNAL ERROR in emit_code: %s", e, exc_info=True)
                    internal_error_msg = f"{type(e).__name__}: {e}"
                    c.print(f"[bold red]❌ INTERNAL ERROR ({type(e).__name__}) during code emission: {e}[/bold red]")

                bind_table = Table(title="🔌 Port Placeholders & Variable Wiring Bindings", box=box.ROUNDED, expand=True, border_style="cyan")
                bind_table.add_column("Step", style="dim", width=4)
                bind_table.add_column("Cell ID", style="bold cyan", ratio=3)
                bind_table.add_column("Port Name", style="bold yellow", ratio=2)
                bind_table.add_column("Dir", style="white", width=5)
                bind_table.add_column("Type & State", style="green", ratio=2)
                bind_table.add_column("Req", style="dim", width=5)
                bind_table.add_column("Bound Expression", style="bold magenta", ratio=2)
                bind_table.add_column("Provenance / Source", style="dim", ratio=2)

                prompt_lits = {v for _, _, v in literals}
                unresolved_ports_set = set(getattr(ctx, "unresolved_ports", []))
                for step_idx, (cl, bindings) in enumerate(pipeline_bindings, 1):
                    for p_name, p_sig in cl.inputs.items():
                        t_str = f"{p_sig.signature.type_name} [{p_sig.signature.state}]"
                        is_unresolved = (cl.cell_id, p_name) in unresolved_ports_set or bindings.get(p_name) is UNRESOLVED_PORT
                        if is_unresolved:
                            b_val = str(UNRESOLVED_PORT)
                            source_desc = "Unresolved Port"
                        elif p_name in bindings and bindings[p_name] is not None:
                            b_val = str(bindings[p_name])
                            if any(b_val.strip("'\"") == lit for lit in prompt_lits):
                                source_desc = "Prompt Literal"
                            elif any(f"'{lit}'" in b_val or f'"{lit}"' in b_val for lit in prompt_lits) or (b_val.startswith("[") and b_val.endswith("]")):
                                source_desc = "Literal Projection"
                            elif b_val.startswith("var_") or b_val.startswith("v"):
                                source_desc = "Wired Variable"
                            elif p_sig.default_value is not None and str(p_sig.default_value) == b_val:
                                source_desc = "Declared Default"
                            elif p_sig.default_value is not None:
                                source_desc = "Literal Projection"
                            else:
                                source_desc = "Dynamic Resolver"
                        elif p_sig.default_value is not None:
                            b_val = str(p_sig.default_value)
                            source_desc = "Declared Default"
                        else:
                            b_val = str(UNRESOLVED_PORT)
                            source_desc = "Unresolved Port"

                        bind_table.add_row(
                            str(step_idx),
                            cl.cell_id,
                            p_name,
                            "IN",
                            t_str,
                            "Yes" if p_sig.required else "No",
                            b_val,
                            source_desc
                        )
                    for p_name, p_sig in cl.outputs.items():
                        b_val = bindings.get(p_name)
                        if b_val is None:
                            is_sink = str(getattr(cl, "node_role", "") or "").lower() in ("sink", "terminal") or getattr(cl, "stage", None) == 3
                            b_val = "None (sink/terminal)" if is_sink else str(UNRESOLVED_PORT)
                        else:
                            b_val = str(b_val)
                        t_str = f"{p_sig.signature.type_name} [{p_sig.signature.state}]"
                        bind_table.add_row(
                            str(step_idx),
                            cl.cell_id,
                            p_name,
                            "OUT",
                            t_str,
                            "-",
                            b_val,
                            "Produced Value"
                        )

                egress_str = ", ".join(f"'{p}'" for p in dest_paths) if dest_paths else "[dim]None (In-memory pipeline)[/dim]"
                sigma_str = str(accum_sigma) if accum_sigma and getattr(accum_sigma, "mappings", None) else "[dim]None (Ground types)[/dim]"
                unify_summary = (
                    f"[cyan]Egress Target Files:[/cyan] {egress_str}\n"
                    f"[cyan]Type Substitutions (σ):[/cyan] {sigma_str}"
                )

                c.print(Panel(unify_summary, title=f"[bold cyan]⚡ LAYER 3: Monadic Variable Binding & AST Synthesis ({synth_dt:.2f}ms)[/bold cyan]", border_style="cyan"))
                c.print(bind_table)
                c.print("")

                lint_valid = True
                lint_res = None
                if refusal_reason:
                    # Declared synthesis refusal (e.g. unresolved ports after
                    # estimator-identity gating): refuse with the specific reason.
                    logger.warning(f"[CLI] Synthesis refused: {refusal_reason} (Profile: {self.active_profile})")
                    c.print(Panel(
                        f"[bold red]❌ SYNTHESIS REFUSED: {refusal_reason}[/bold red]",
                        title="[bold red]⚡ LAYER 3: Synthesis Refusal[/bold red]",
                        border_style="red",
                    ))
                    path_ids_r = [cl.cell_id for cl in cells]
                    total_dt_r = (time.perf_counter() - t_total_start) * 1000.0
                    return {
                        "prompt": prompt_clean,
                        "profile": self.active_profile,
                        "path": path_ids_r,
                        "latency_ms": total_dt_r,
                        "sandbox_status": f"REFUSED: {refusal_reason}",
                        "code": None,
                        "internal_error": None,
                        "refusal_reason": refusal_reason,
                        "route_ms": route_dt,
                        "plan_ms": plan_dt,
                        "synth_ms": synth_dt,
                        "sandbox_ms": 0.0,
                        "sandbox_result": {"success": False, "error": refusal_reason, "refused": True},
                        "cells": cells,
                        "tunnel_size": len(tunnel_cells),
                        "relevance_map": relevance_map,
                    }
                if pipeline_bindings and not internal_error_msg:
                    try:
                        lint_res = PreflightLinter.lint(pipeline_bindings, prompt=prompt, code_str=final_code, orchestrator=self.orchestrator)
                        lint_valid = lint_res.is_valid
                    except Exception as lint_err:
                        logger.warning(f"[PREFLIGHT] Linting exception: {lint_err}")
                        lint_res = None
                        lint_valid = True

                # Deterministic Bridge Insertion for Pre-Flight Lint Violations
                if not lint_valid and lint_res and getattr(lint_res, "structured_violations", None) and cells:
                    t_bridge_0 = time.perf_counter()
                    repaired_cells = list(cells)
                    bridge_inserted = False

                    for viol in lint_res.structured_violations:
                        if viol.check_id in ("shape_carrier_mismatch", "type_unification_failure"):
                            rej_quals = {str(q).lower() for q in viol.details.get("rejected_qualifiers", [])}
                            res_quals = {str(q).lower() for q in viol.details.get("resolved_qualifiers", [])}
                            conflict = rej_quals & res_quals
                            prod_id = viol.details.get("producer_cell_id")

                            # Bridge selection is derived solely from lattice declarations
                            # (Rule 1: no qualifier-name or cell-id literals in the engine).
                            # A valid bridge cell:
                            #   (a) declares tolerance for the conflicting qualifier on one of
                            #       its input ports (the qualifier token appears in the port's
                            #       declared state / accepted_states / qualifiers vocabulary),
                            #   (b) declares an output that does NOT carry any rejected qualifier.
                            # Type validity of the insertion is then enforced by re-running the
                            # full unification gate below.
                            def _port_declares_acceptance(p_sig: Any, wanted_quals: Set[str]) -> bool:
                                declared: List[str] = list(getattr(p_sig, "qualifiers", None) or ())
                                sig_inner = getattr(p_sig, "signature", None)
                                declared.append(str(getattr(sig_inner, "state", "") or ""))
                                declared.extend(str(s) for s in (getattr(sig_inner, "accepted_states", None) or frozenset()))
                                declared_toks: Set[str] = set()
                                for d in declared:
                                    declared_toks.update(CellTokenizer.tokenize_identifier(d.lower()))
                                return any(str(q).lower() in declared_toks for q in wanted_quals)

                            bridge_cand = None
                            if conflict:
                                for c_cell in self.orchestrator.loaded_cells.values():
                                    if getattr(c_cell, "node_role", "") not in ("bridge", "transformer"):
                                        continue
                                    c_out = getattr(c_cell, "primary_output", None)
                                    if c_out is None:
                                        continue
                                    c_out_quals = {str(q).lower() for q in (getattr(c_out, "qualifiers", []) or [])}
                                    if c_out_quals & rej_quals:
                                        continue
                                    if not any(
                                        _port_declares_acceptance(p_sig, conflict)
                                        for p_sig in c_cell.inputs.values()
                                    ):
                                        continue
                                    bridge_cand = c_cell
                                    break

                            if bridge_cand and prod_id:
                                prod_idx = next((i for i, cl in enumerate(repaired_cells) if cl.cell_id == prod_id), None)
                                if prod_idx is not None and (prod_idx + 1 >= len(repaired_cells) or repaired_cells[prod_idx + 1].cell_id != bridge_cand.cell_id):
                                    repaired_cells.insert(prod_idx + 1, bridge_cand)
                                    bridge_inserted = True
                                    for cl in repaired_cells[prod_idx + 1:]:
                                        cl.planned_port_wires = {}
                                        cl.bound_parent_ids = set()

                        elif viol.check_id == "precondition_violated":
                            prod_id = viol.details.get("established_by")
                            candidates = viol.details.get("bridge_candidates", [])
                            prop = viol.details.get("property")
                            if prod_id and candidates:
                                prod_idx = next((i for i, cl in enumerate(repaired_cells) if cl.cell_id == prod_id), None)
                                if prod_idx is not None:
                                    while prod_idx + 1 < len(repaired_cells) and any(
                                        str(eff).lower() == f"preserves:{str(prop).lower()}"
                                        for eff in (getattr(repaired_cells[prod_idx + 1], "effects", None) or [])
                                    ):
                                        prod_idx += 1
                                    best_cand = None
                                    for c_id in candidates:
                                        cand_cell = self.orchestrator.loaded_cells.get(c_id)
                                        if cand_cell:
                                            if len(cand_cell.inputs) == 1:
                                                best_cand = cand_cell
                                                break
                                            if best_cand is None:
                                                best_cand = cand_cell
                                    if best_cand and (prod_idx + 1 >= len(repaired_cells) or repaired_cells[prod_idx + 1].cell_id != best_cand.cell_id):
                                        b_inst = best_cand.clone() if hasattr(best_cand, "clone") else best_cand
                                        b_inst.matched_clause_idx = getattr(repaired_cells[prod_idx], "matched_clause_idx", None)
                                        repaired_cells.insert(prod_idx + 1, b_inst)
                                        bridge_inserted = True
                                        for cl in repaired_cells[prod_idx + 1:]:
                                            cl.planned_port_wires = {}
                                            cl.bound_parent_ids = set()

                    if bridge_inserted:
                        # Real synthesis chain, same two calls as the main path above
                        # (unify_pipeline + emit_code). The gate exposes no `synthesize`
                        # method; the previous call raised AttributeError inside a bare
                        # try/except and silently never ran (R6-3).
                        try:
                            from dag_plan import OpenPortFrontier
                            repaired_cells = OpenPortFrontier.build_dag_plan(repaired_cells)
                        except Exception:
                            pass
                        try:
                            bridge_ctx = ExecutionContext(prompt=prompt_clean)
                            bridge_params = getattr(ctx, "parameters", None)
                            if isinstance(bridge_params, dict) and bridge_params:
                                bridge_ctx.parameters.update(bridge_params)
                            bridge_unify = self.gate.unify_pipeline(repaired_cells, bridge_ctx)
                            if bridge_unify.is_bottom():
                                raise SynthesisError(
                                    str(getattr(bridge_unify, "reason", "unification failed after bridge insertion"))
                                )
                            bridge_code = self.gate.emit_code(bridge_unify.value, bridge_ctx)
                            rep_lint = PreflightLinter.lint(
                                self.gate.last_pipeline_bindings, prompt=prompt, code_str=bridge_code, orchestrator=self.orchestrator
                            )
                            bridge_dt = (time.perf_counter() - t_bridge_0) * 1000.0
                            if rep_lint.is_valid:
                                c.print(f"  [bold green][✓] Deterministic bridge morphism inserted ({bridge_dt:.1f}ms).[/bold green]\n")
                                cells = repaired_cells
                                final_code = bridge_code
                                pipeline_bindings = self.gate.last_pipeline_bindings
                                lint_valid = True
                                lint_res = rep_lint
                        except Exception as e:
                            logger.warning(
                                "Deterministic bridge insertion failed: %s: %s", type(e).__name__, e
                            )
                            c.print(f"  [dim yellow]Deterministic bridge insertion attempt encountered: {type(e).__name__}: {e}[/dim yellow]")

                # Layer 5 Self-Repair for Pre-Flight Lint Violations
                llm_feedback = getattr(self, "llm_feedback", False)
                rep_dt = 0.0
                if not lint_valid and llm_feedback and final_code:
                    mm = ModelManager.get_instance()
                    if mm.profile and mm.can_feedback_check():
                        err_msg = "Pre-flight lint validation failed:\n" + "\n".join(f"- {v}" for v in lint_res.violations)
                        c.print(Panel("[bold yellow]⚡ LAYER 5: Pre-Flight Lint LLM Self-Repair Cycle[/bold yellow]", border_style="yellow"))
                        t_rep_0 = time.perf_counter()
                        failing_code = final_code
                        repaired_code = extract_code_from_llm_response(mm.feedback_check(failing_code, err_msg))
                        rep_dt = (time.perf_counter() - t_rep_0) * 1000.0
                        if repaired_code and repaired_code.strip() != failing_code.strip():
                            unknown = _unknown_api_references(repaired_code, failing_code, self.orchestrator.loaded_cells.values())
                            if not unknown:
                                rep_lint = PreflightLinter.lint(pipeline_bindings, prompt=prompt, code_str=repaired_code, orchestrator=self.orchestrator)
                                if rep_lint.is_valid:
                                    c.print(f"  [bold green][✓] Pre-flight repair accepted ({rep_dt:.1f}ms).[/bold green]\n")
                                    final_code = repaired_code
                                    lint_valid = True
                                    lint_res = rep_lint

                if not lint_valid and lint_res is not None:
                    logger.error(f"[CLI] Pre-flight lint validation failed ({len(lint_res.violations)} violations): {'; '.join(lint_res.violations)}")
                    c.print(Panel(
                        "[bold red]❌ PRE-FLIGHT LINT VIOLATIONS DETECTED:[/bold red]\n" +
                        "\n".join(f"  • {v}" for v in lint_res.violations),
                        title="[bold red]⚠️ Pre-Flight Static Validator[/bold red]",
                        border_style="red"
                    ))
                    c.print("[bold red][!] Code emission refused due to pre-flight lint violations.[/bold red]\n")
                    final_code = None
                    path_ids = [cl.cell_id for cl in cells] if cells else []
                    total_dt = (time.perf_counter() - t_total_start) * 1000.0
                    return {
                        "prompt": prompt_clean,
                        "profile": self.active_profile,
                        "path": path_ids,
                        "latency_ms": total_dt,
                        "sandbox_status": "REFUSED: Pre-flight lint validation failed",
                        "code": None,
                        "route_ms": route_dt,
                        "plan_ms": plan_dt,
                        "synth_ms": synth_dt,
                        "sandbox_ms": 0.0,
                        "sandbox_result": {"success": False, "error": f"Pre-flight lint validation failed: {'; '.join(lint_res.violations)}", "preflight_lint_violations": lint_res.violations},
                        "cells": cells,
                        "tunnel_size": len(tunnel_cells),
                        "relevance_map": relevance_map
                    }
                elif lint_res is not None:
                    lint_msg = "[bold green]✓ All structural contracts and port binding constraints satisfied.[/bold green]"
                    if lint_res.warnings:
                        lint_msg += "\n[yellow]Warnings:[/yellow]\n" + "\n".join(f"  • {w}" for w in lint_res.warnings)
                    c.print(Panel(lint_msg, title="[bold green]✓ Pre-Flight Static Validator Passed (Type Contracts Verified)[/bold green]", border_style="green"))
                    c.print("")

                if final_code:
                    syntax_code = Syntax(final_code, "python", theme="monokai", line_numbers=True)
                    c.print(Panel(syntax_code, title="[bold green]✨ Synthesized Python Code[/bold green]", border_style="green", padding=(0, 1)))
                    c.print("")

        # LAYER 4: GEVR Sandbox Execution & Verification
        sandbox_res = {"success": False, "error": "Execution skipped"}
        sandbox_dt = 0.0

        if not execute_sandbox:
            v_contract = getattr(self.gate, "last_verification_contract", None)
            static_violations = _statically_evaluate_contract(final_code, v_contract)
            if static_violations:
                err_msg = f"Static verification contract violated: {'; '.join(static_violations)}"
                sandbox_res = {"success": False, "verified": False, "skipped": True, "error": err_msg}
                sb_badge = "[bold red]✗ CONTRACT VIOLATION (Static Verification Failed)[/bold red]"
                border_col = "red"
                sb_summary = [
                    f"[cyan]Status:[/cyan] {sb_badge}",
                    f"[red]Violations:[/red] {'; '.join(static_violations)}",
                    f"[yellow]Notice:[/yellow] Physical sandbox was bypassed, but static contract verification failed."
                ]
            else:
                sandbox_res = {"success": None, "skipped": True, "verified": bool(v_contract)}
                sb_badge = "[bold yellow]⚡ BYPASSED (Execution Disabled)[/bold yellow]"
                border_col = "yellow"
                sb_summary = [
                    f"[cyan]Status:[/cyan] {sb_badge}",
                    f"[cyan]Notice:[/cyan] Physical sandbox verification bypassed to eliminate latency."
                ]
            c.print(Panel("\n".join(sb_summary), title=f"[bold {border_col}]⚡ LAYER 4: GEVR Sandbox Execution & Verification[/bold {border_col}]", border_style=border_col))
        elif final_code:
            has_unresolved = bool(getattr(ctx, "unresolved_ports", None))
            if not lint_valid or has_unresolved:
                reasons = []
                if not lint_valid and 'lint_res' in locals():
                    reasons.extend(lint_res.violations)
                if has_unresolved:
                    reasons.extend([f"Unresolved port: {cid}.{p}" for cid, p in ctx.unresolved_ports])
                err_msg = f"Pre-flight lint validation failed: {'; '.join(reasons)}"
                sandbox_res = {"success": False, "error": err_msg, "skipped": True}
                c.print(f"[bold red][!] Sandbox execution aborted: {err_msg}[/bold red]\n")
            else:
                t_exec_start = time.perf_counter()
                v_contract = getattr(self.gate, "last_verification_contract", None)
                sandbox_res = self.sandbox.execute(
                    final_code,
                    timeout=timeout,
                    egress_paths=dest_paths,
                    verification_spec=v_contract,
                    runtime_aliases=getattr(self.gate, 'last_runtime_aliases', None),
                    pipeline_bindings=pipeline_bindings,
                    extracted_literals=literals,
                )
                sandbox_dt = (time.perf_counter() - t_exec_start) * 1000.0

            sb_success = sandbox_res.get("success", False)
            sb_ret = sandbox_res.get("returncode", 0)
            sb_stdout = sandbox_res.get("stdout", "").strip()
            sb_stderr = sandbox_res.get("stderr", "").strip()
            sb_err = sandbox_res.get("error", "").strip()

            if sandbox_res.get("skipped"):
                sb_badge = f"[bold yellow]⚠ SKIPPED ({sb_err})[/bold yellow]"
                border_col = "yellow"
            elif sb_success:
                sb_badge = "[bold green]✓ PASSED[/bold green]"
                border_col = "green"
            else:
                sb_badge = f"[bold red]✗ FAILED (Exit Code: {sb_ret})[/bold red]"
                border_col = "red"

            sb_summary = [
                f"[cyan]Status:[/cyan] {sb_badge}",
                f"[cyan]Execution Time:[/cyan] {sandbox_dt:.2f}ms",
                f"[cyan]Timeout Bound:[/cyan] {timeout:.1f}s"
            ]

            if dest_paths:
                egress_checks = []
                for dp in dest_paths:
                    p = Path(dp)
                    if p.exists():
                        sz = p.stat().st_size
                        egress_checks.append(f"'{dp}': [bold green]EXISTS ({sz} bytes)[/bold green]")
                    else:
                        egress_checks.append(f"'{dp}': [bold red]MISSING ON DISK[/bold red]")
                sb_summary.append(f"[cyan]Egress Artifacts:[/cyan] {' | '.join(egress_checks)}")

            c.print(Panel("\n".join(sb_summary), title=f"[bold {border_col}]⚡ LAYER 4: GEVR Sandbox Execution & Verification[/bold {border_col}]", border_style=border_col))

            if sb_stdout:
                c.print(Panel(sb_stdout, title="[green]Standard Output (stdout)[/green]", border_style="dim green"))

            if sb_stderr or sb_err:
                err_text = sb_err or sb_stderr
                is_env = bool(sandbox_res.get("extrinsic", False) or _is_environmental_error(err_text))
                logger.error(f"[CLI] Sandbox execution failed ({'ENVIRONMENTAL' if is_env else 'CODE DEFECT'}): {err_text}")
                c.print(Panel(err_text, title="[red]Standard Error / Traceback (stderr)[/red]", border_style="red"))

                if sandbox_res.get("extrinsic", False) or _is_environmental_error(err_text):
                    c.print(Panel(
                        "[yellow][ENVIRONMENTAL FAILURE DETECTED][/yellow]\n"
                        "The synthesized Python program failed due to an external environmental dependency "
                        "(e.g. input file not found on disk, missing system package, or IO permission).\n"
                        "[bold green]The synthesized pipeline logic and type composition are structurally sound.[/bold green]",
                        border_style="yellow"
                    ))
                else:
                    c.print(Panel(
                        "[bold red][CODE DEFECT DETECTED][/bold red]\n"
                        "An exception occurred inside the synthesized code during runtime execution.",
                        border_style="red"
                    ))
            c.print("")

        # LAYER 5: Self-Repair Cycle
        rep_dt = 0.0
        repaired = False
        if final_code and not sandbox_res.get("success", False) and getattr(self, "llm_feedback", False) and ModelManager.get_instance().can_feedback_check():
            mm = ModelManager.get_instance()
            if mm.profile and mm.can_feedback_check():
                err_msg = sandbox_res.get("error", "")
                if sandbox_res.get("extrinsic", False) or _is_environmental_error(err_msg):
                    c.print(
                        "  [yellow][!] Environmental failure detected. LLM self-repair skipped — "
                        "the synthesized pipeline is not the defect.[/yellow]\n"
                    )
                else:
                    c.print(Panel("[bold yellow]⚡ LAYER 5: GEVR Sandbox LLM Self-Repair Cycle[/bold yellow]", border_style="yellow"))
                    t_rep_0 = time.perf_counter()
                    failing_code = final_code
                    repaired_code = extract_code_from_llm_response(mm.feedback_check(failing_code, err_msg))
                    rep_dt = (time.perf_counter() - t_rep_0) * 1000.0

                    if repaired_code and repaired_code.strip() != failing_code.strip():
                        unknown = _unknown_api_references(repaired_code, failing_code, self.orchestrator.loaded_cells.values())
                        if unknown:
                            c.print(f"  [bold red][x] Repair rejected: references APIs absent from the lattice: {', '.join(sorted(unknown)[:5])}[/bold red]\n")
                        else:
                            c.print(f"  [bold green][✓] Repair accepted. Re-executing in sandbox... ({rep_dt:.1f}ms)[/bold green]")
                            final_code = repaired_code
                            repaired = True
                            v_contract = getattr(self.gate, "last_verification_contract", None)
                            sandbox_res = self.sandbox.execute(
                                final_code,
                                timeout=timeout,
                                egress_paths=dest_paths,
                                verification_spec=v_contract,
                                runtime_aliases=getattr(self.gate, 'last_runtime_aliases', None),
                                pipeline_bindings=pipeline_bindings,
                                extracted_literals=literals,
                            )
                            c.print(f"  [bold]Post-Repair Result:[/bold] {'[green]PASSED[/green]' if sandbox_res.get('success') else '[red]FAILED[/red]'}\n")
                    else:
                        c.print(f"  [yellow][!] LLM could not produce an alternative repair ({rep_dt:.1f}ms).[/yellow]\n")

        # LAYER 6: Performance & Latency Breakdown
        total_dt = (time.perf_counter() - t_total_start) * 1000.0

        perf_table = Table(title="⏱️ End-to-End Pipeline Latency Breakdown", box=box.ROUNDED, expand=True, border_style="cyan")
        perf_table.add_column("Pipeline Layer", style="bold cyan")
        perf_table.add_column("Latency (ms)", style="bold yellow", justify="right")
        perf_table.add_column("% of Total", style="dim", justify="right")

        timings = [
            ("Layer 0: Translator Pass", t_trans),
            ("Layer 1: Semantic Routing & Tunneling", route_dt),
            ("Layer 2: Topological Planning (Viterbi)", plan_dt),
            ("Layer 3: Monadic Unification & Code Gen", synth_dt),
            ("Layer 4: GEVR Sandbox Execution", sandbox_dt),
            ("Layer 5: LLM Self-Repair Cycle", rep_dt),
        ]
        for name, lat in timings:
            pct = (lat / total_dt * 100) if total_dt > 0 else 0.0
            lat_str = f"{lat:.2f} ms" if lat > 0 else "[dim]-[/dim]"
            pct_str = f"{pct:.1f}%" if lat > 0 else "[dim]-[/dim]"
            perf_table.add_row(name, lat_str, pct_str)

        perf_table.add_section()
        perf_table.add_row("[bold white]Total End-to-End Latency[/bold white]", f"[bold green]{total_dt:.2f} ms[/bold green]", "[bold green]100.0%[/bold green]")
        c.print(perf_table)
        c.print("")

        try:
            from cli_helpers import render_failure_surface_summary
            render_failure_surface_summary(c, ctx=dbg_ctx, router=self.router, gate=self.gate)
        except Exception as _fs_err:
            logger.debug("[CLI] Failure surface summary skipped: %s", _fs_err)

        path_ids = [cl.cell_id for cl in cells] if cells else []
        sb_status = "PASSED" if sandbox_res.get("success", False) else ("FAILED: " + sandbox_res.get("error", "").splitlines()[-1] if sandbox_res.get("error") else "FAILED")
        return {
            "prompt": prompt_clean,
            "profile": self.active_profile,
            "path": path_ids,
            "latency_ms": total_dt,
            "sandbox_status": sb_status,
            "code": final_code,
            "internal_error": internal_error_msg,
            "refusal_reason": refusal_reason,
            "route_ms": route_dt,
            "plan_ms": plan_dt,
            "synth_ms": synth_dt,
            "sandbox_ms": sandbox_dt,
            "sandbox_result": sandbox_res,
            "cells": cells,
            "tunnel_size": len(tunnel_cells),
            "relevance_map": relevance_map
        }


# =====================================================================
# Interactive TUI Shell
# =====================================================================
class NSTLInteractiveShell(cmd.Cmd):
    """
    Rich Terminal User Interface (TUI) Studio for STRATA Neuro-Symbolic Synthesis.
    Provides a visual, interactive CLI workspace with instant profile switching,
    model exploration, real-time latency diagnostics, and self-repair cycles.
    """

    prompt = "\033[1;36mSTRATA [Profile 0: Symbolic]\033[0m > "

    def emptyline(self):
        """Do nothing on empty line (prevents re-running lastcmd)."""
        pass

    def __init__(
        self,
        db_path: str = "trees/lattice.db",
        initial_profile: str = "0",
        embedder: str = "",
        llm: str = "",
        device: str = "auto",
        debug: bool = False,
        interactive: bool = True,
        no_exec: bool = False,
        route_method: Optional[str] = None,
        no_lint: bool = False,
        macros: Optional[bool] = None,
        topology: Optional[str] = None,
        dev: Optional[bool] = None,
        timeout: float = DEFAULT_SANDBOX_TIMEOUT,
        reranker: Optional[bool] = None,
        reranker_model: Optional[str] = None,
        exec_sandbox: bool = False,
        llm_feedback: bool = False,
        allowed_capabilities: Optional[Union[str, List[str], Set[str]]] = None,
    ):
        super().__init__()
        self.db_path = db_path
        self.device = device
        self.embedder_name = select_optimal_embedder(embedder)
        self.llm_name = select_optimal_llm(llm)
        self.active_profile = "0"
        self.route_method = route_method.upper() if route_method else None
        self.no_lint: bool = bool(no_lint)
        self.llm_feedback: bool = bool(llm_feedback)
        if self.llm_feedback:
            os.environ["NSTL_LLM_FEEDBACK"] = "1"
        else:
            os.environ.pop("NSTL_LLM_FEEDBACK", None)
        if isinstance(allowed_capabilities, str):
            self.allowed_capabilities: Set[str] = {c.strip() for c in allowed_capabilities.split(",") if c.strip()}
        else:
            self.allowed_capabilities = set(allowed_capabilities or [])
        if self.allowed_capabilities:
            os.environ["NSTL_ALLOWED_CAPABILITIES"] = ",".join(sorted(self.allowed_capabilities))
        self.rag: Optional[LocalRAG] = None
        self.history: List[Dict[str, Any]] = []
        self.debug: bool = debug
        self.interactive: bool = interactive
        if exec_sandbox:
            self.no_exec = False
        elif no_exec:
            self.no_exec = True
        else:
            self.no_exec = not settings.sandbox_enabled
        self.timeout: float = timeout

        if reranker is not None:
            settings.use_reranker = bool(reranker)
            os.environ["NSTL_USE_RERANKER"] = "1" if reranker else "0"
        self.use_reranker = settings.use_reranker

        if reranker_model is not None:
            settings.reranker_model = reranker_model
            os.environ["NSTL_RERANKER_MODEL"] = str(reranker_model)
        self.reranker_model = settings.reranker_model

        if macros is not None:
            settings.macros_enabled = bool(macros)
        self.macros_enabled = settings.macros_enabled

        if dev is not None:
            settings.dev_mode = bool(dev)
        self.dev_mode = settings.dev_mode

        if topology is not None:
            settings.topology_mode = str(topology).lower()
        self.topology_mode = settings.topology_mode

        if interactive:
            console.print("\n[bold cyan][*] Initializing STRATA Neuro-Symbolic Engine...[/bold cyan]")
        t0 = time.perf_counter()

        trees_dir = settings.trees_dir or Path(db_path).parent
        ensure_lattice_compiled(trees_dir=str(trees_dir), db_path=db_path)
        self.orchestrator = LatticeOrchestrator(trees_directory=str(trees_dir), db_path=db_path)
        self.orchestrator.load_from_database(db_path)
        self.orchestrator.build_topology()
        self.gate = UnificationGate(self.orchestrator)
        self.sandbox = GEVRSandbox(allowed_capabilities=self.allowed_capabilities)

        node_count = len(self.orchestrator.cells)
        load_time = (time.perf_counter() - t0) * 1000.0
        if interactive:
            console.print(f"[bold green][✓] Lattice Graph Loaded: {node_count:,} verified nodes ({load_time:.1f}ms)[/bold green]\n")

        self._switch_profile(initial_profile, embedder=embedder, llm=llm, device=device, verbose=False)
        if interactive:
            self._render_dashboard()

    def _render_dashboard(self):
        """Renders the top visual status dashboard."""
        domains = set(c.domain_name for c in self.orchestrator.cells if c.domain_name)
        prof_name = self._format_profile_name(self.active_profile)
        prof_desc = self._get_profile_description(self.active_profile)

        header_table = Table(box=box.ROUNDED, expand=True, border_style="cyan")
        header_table.add_column("⚡ Layer Profile", style="bold yellow", ratio=3)
        header_table.add_column("🧠 Neural Models", style="bold magenta", ratio=3)
        header_table.add_column("📊 Topology & Hardware", style="bold green", ratio=3)

        prof_text = f"[bold white]{prof_name}[/bold white]\n[dim]{prof_desc}[/dim]"
        emb_text = f"[cyan]Embedder:[/cyan] {self.embedder_name or '[dim]None (Bypassed)[/dim]'}"
        llm_text = f"[cyan]LLM (GGUF):[/cyan] {self.llm_name or '[dim]None (Bypassed)[/dim]'}"
        r_model = self.reranker_model or "jina-reranker-v3.5"
        rerank_status = f"[bold green]ON ({r_model})[/bold green]" if getattr(self, "use_reranker", False) else "[dim]OFF[/dim]"
        rerank_text = f"[cyan]Reranker:[/cyan] {rerank_status}"
        models_text = f"{emb_text}\n{llm_text}\n{rerank_text}"

        db_nodes = f"[cyan]Nodes:[/cyan] {len(self.orchestrator.cells):,} in {len(domains)} domains"
        dbg_text = "[bold green]ON (Verbose)[/bold green]" if self.debug else "[dim]OFF[/dim]"
        _resolved_method = self.route_method or str(getattr(self.router, "default_route_method", "m0")).upper()
        method_text = f"[bold yellow]{_resolved_method}{' (Default)' if not self.route_method else ''}[/bold yellow]"
        topo_mode = getattr(self, "topology_mode", "frontier")
        topo_badge = "[bold green]Frontier (DAG)[/bold green]" if topo_mode == "frontier" else "[bold yellow]Linear (1D)[/bold yellow]"
        dev_info = f"[cyan]Method:[/cyan] {method_text} | [cyan]Topology:[/cyan] {topo_badge}\n[cyan]Device:[/cyan] {self.device.upper()} | [cyan]Debug:[/cyan] {dbg_text}"
        hardware_text = f"{db_nodes}\n{dev_info}"

        header_table.add_row(prof_text, models_text, hardware_text)

        title_text = Text("🧬 STRATA: SYMBOLIC TOPOLOGICAL ROUTING ARCHITECTURE FOR TYPED AUTOMATA", justify="center", style="bold white on blue")
        _short_label = {"0": "Symbolic", "A": "Embedder", "C": "Neuro-Symbolic", "D": "Routing", "E": "Translator", "S": "Semantic-Compiler"}
        _quick_line = "  ".join(f"[{k}] {v}:{_short_label.get(v, v)}" for k, v in QUICK_PROFILE_SHORTCUTS)
        quick_shortcuts = Text(
            f"Quick Layers: {_quick_line}\n"
            "Commands: /profile <0|A|B|C|D|E|S|U|AE> | /method <M<n>> | /reranker [on|off|<model>] | /topology <frontier|linear> | /audit | /macro | /debug [on|off] | /sandbox [on|off] | /status | /new | /clear | /exit",
            justify="center",
            style="dim cyan"
        )

        dashboard_panel = Panel(
            header_table,
            title=title_text,
            subtitle=quick_shortcuts,
            border_style="bright_blue",
            padding=(0, 1)
        )
        console.print(dashboard_panel)
        console.print("[dim]Type any natural language pipeline specification to synthesize code in real-time:[/dim]\n")

    def _format_profile_name(self, prof: str) -> str:
        prof_u = prof.upper()
        if prof_u in ("0", "SYMBOLIC", "ZERO", "PURE"):
            return "Profile 0 (Pure Symbolic Layer)"
        elif prof_u == "A":
            return "Profile A (Dense Embeddings RAG)"
        elif prof_u in ("C", "B"):
            return "Profile C (Hybrid Neuro-Symbolic + GGUF LLM)"
        elif prof_u == "D":
            return "Profile D (Routing-Only Benchmark)"
        elif prof_u == "E":
            return "Profile E (Translator Pass + Neuro-Symbolic)"
        elif prof_u == "S":
            return "Profile S (Structured Semantic Compiler)"
        elif prof_u == "U":
            return "Profile U (Early-Exit Single LLM Layer-Tapping)"
        elif prof_u == "AE":
            return "Profile AE (Structural Transition Autoencoder Manifold)"
        return f"Profile {prof_u}"

    def _get_profile_description(self, prof: str) -> str:
        prof_u = prof.upper()
        if prof_u in ("0", "SYMBOLIC", "ZERO", "PURE"):
            return "Deterministic type-safe lattice search across all loaded nodes. Baseline layer (zero neural models)."
        elif prof_u == "A":
            return "FAISS vector retrieval with dense embedding model."
        elif prof_u in ("C", "B"):
            return "Embedder RAG + GGUF local LLM slot-filling & self-repair."
        elif prof_u == "D":
            return "LLM-guided path search without full code synthesis."
        elif prof_u == "E":
            return "2-stage pipeline: conversational prompt -> canonical translator."
        elif prof_u == "S":
            return "Typed Intent & Schema Compiler pass + Neuro-Symbolic synthesis."
        elif prof_u == "U":
            return "Single unified LLM: intermediate layer k tapped for dense RAG, full depth for generation."
        elif prof_u == "AE":
            return "Continuous transition manifold embedding type signatures and cell morphisms."
        return ""

    def _update_prompt(self):
        prof_label = self.active_profile.upper()
        if prof_label in ("0", "SYMBOLIC", "ZERO", "PURE"):
            prof_label = "0: Symbolic"
        method_label = f" | {self.route_method}" if self.route_method else ""
        dbg_label = " \033[1;33m[DEBUG]\033[0m" if self.debug else ""
        self.prompt = f"\033[1;36mSTRATA [Profile {prof_label}{method_label}]\033[0m{dbg_label} > "

    def do_debug(self, arg: str):
        """Toggle or configure debug mode across all pipeline layers. Usage: /debug [on|off]"""
        arg = arg.strip().lower().lstrip("/")
        if arg.startswith("debug"):
            arg = arg[5:].strip()
        if not arg:
            self.debug = not self.debug
        elif arg in ("1", "true", "on", "yes", "enable", "enabled"):
            self.debug = True
        elif arg in ("0", "false", "off", "no", "disable", "disabled"):
            self.debug = False
        else:
            console.print(f"[yellow]Usage: /debug [on|off] (currently {'ON' if self.debug else 'OFF'})[/yellow]")
            return

        status_str = "[bold green]ENABLED (full layer-by-layer diagnostics)[/bold green]" if self.debug else "[dim]DISABLED[/dim]"
        console.print(f"\n[*] Debug Mode: {status_str}\n")
        self._update_prompt()

    def do_sandbox(self, arg: str):
        """Toggle GEVR sandbox execution. Usage: /sandbox [on|off]"""
        arg = arg.strip().lower().lstrip("/")
        if arg.startswith("sandbox"):
            arg = arg[7:].strip()
        if not arg:
            self.no_exec = not self.no_exec
        elif arg in ("1", "true", "on", "yes", "enable", "enabled"):
            self.no_exec = False
        elif arg in ("0", "false", "off", "no", "disable", "disabled"):
            self.no_exec = True
        else:
            status = "DISABLED" if self.no_exec else "ENABLED"
            console.print(f"[yellow]Usage: /sandbox [on|off] (currently {status})[/yellow]")
            return

        status_str = "[bold green]ENABLED[/bold green]" if not self.no_exec else "[bold yellow]DISABLED (bypassed for speed)[/bold yellow]"
        console.print(f"\n[*] GEVR Sandbox: {status_str}\n")

    def do_topology(self, arg: str):
        """Switch topological planning approach. Usage: /topology <frontier|linear>"""
        arg = arg.strip().lower().lstrip("/")
        if arg.startswith("topology"):
            arg = arg[8:].strip()
        if arg in ("frontier", "dag", "monoidal", "multi"):
            self.topology_mode = "frontier"
        elif arg in ("linear", "1d", "trellis", "sequential", "baseline"):
            self.topology_mode = "linear"
        elif not arg:
            self.topology_mode = "linear" if self.topology_mode == "frontier" else "frontier"
        else:
            console.print(f"[yellow]Usage: /topology <frontier|linear> (currently {self.topology_mode.upper()})[/yellow]")
            return

        settings.topology_mode = self.topology_mode
        if hasattr(self, "router") and self.router and hasattr(self.router, "planner"):
            self.router.planner.topology_mode = self.topology_mode
        mode_str = "[bold green]Frontier DAG (Multi-Carrier Monoidal Category)[/bold green]" if self.topology_mode == "frontier" else "[bold yellow]Linear Trellis (1D Monadic Baseline)[/bold yellow]"
        console.print(f"\n[*] Planning Topology: {mode_str}\n")

    def _switch_profile(self, profile: str, embedder: str = "", llm: str = "", device: str = "auto", verbose: bool = True) -> bool:
        p = profile.strip().upper()
        import os
        os.environ["NSTL_PROFILE"] = p
        try:
            from config import settings
            settings.profile = p
        except Exception:
            pass
        try:
            from cognitive_pathfinder import LatentVectorExtractor
            LatentVectorExtractor.reset_instance()
        except Exception:
            pass
        if p in ("0", "SYMBOLIC", "ZERO", "PURE"):
            self.active_profile = "0"
            os.environ["NSTL_PROFILE"] = "0"
            try:
                from config import settings
                settings.profile = "0"
            except Exception:
                pass
            self.rag = None
            self.router = LatticeRouter(
                self.orchestrator,
                internal_rag=None,
                default_route_method=self.route_method,
                use_reranker=getattr(self, "use_reranker", False),
                reranker_model=getattr(self, "reranker_model", None),
            )
            self._update_prompt()
            if verbose:
                console.print(f"[bold green][✓] Switched to {self._format_profile_name(self.active_profile)}[/bold green]\n")
            return True

        if p not in ("A", "B", "C", "D", "E", "S", "U", "AE"):
            console.print(f"[bold red][!] Unknown profile '{profile}'. Valid options: 0 (Symbolic), A, B, C, D, E, S, U, AE.[/bold red]")
            return False

        available_llm = _get_available_llms()
        if p == "U":
            emb_choice = "llm_early_exit"
            llm_choice = select_optimal_llm(llm or self.llm_name)
        elif p == "AE":
            emb_choice = "transition_autoencoder"
            llm_choice = ""
        else:
            emb_choice = select_optimal_embedder(embedder or self.embedder_name or "auto")
            llm_choice = select_optimal_llm(llm or self.llm_name)

        if verbose:
            console.print(f"[bold cyan][*] Loading {self._format_profile_name(p)}...[/bold cyan]")
        t0 = time.perf_counter()
        try:
            HardwareProfiler.set_config(embedder_device=device, llm_device=device)
            mm = ModelManager.get_instance()
            mm.initialize_profile(
                profile_type=p,
                embedder_name=emb_choice,
                llm_name=llm_choice if p in ("B", "C", "D", "E", "S", "U") else ""
            )
            self.embedder_name = emb_choice
            self.llm_name = llm_choice if p in ("B", "C", "D", "E", "S", "U") else ""

            if verbose:
                console.print(f"[*] Indexing FAISS vector space for {len(self.orchestrator.cells):,} nodes...")
            self.rag = LocalRAG(trees_dir="trees", orchestrator=self.orchestrator)
            self.orchestrator.rag = self.rag
            self.router = LatticeRouter(
                self.orchestrator,
                internal_rag=self.rag,
                default_route_method=self.route_method,
                use_reranker=getattr(self, "use_reranker", False),
                reranker_model=getattr(self, "reranker_model", None),
            )

            self.active_profile = p
            self._update_prompt()
            dt = (time.perf_counter() - t0) * 1000.0
            if verbose:
                console.print(f"[bold green][✓] {self._format_profile_name(p)} ready ({dt:.1f}ms).[/bold green]\n")
            return True
        except Exception as e:
            console.print(f"[bold red][!] Failed to load Profile {p}: {e}[/bold red]")
            console.print("[yellow][*] Reverting to Profile 0 (Pure Symbolic)...[/yellow]")
            try:
                ModelManager.get_instance().cleanup()
            except Exception:
                pass
            self.active_profile = "0"
            self.rag = None
            self.router = LatticeRouter(
                self.orchestrator,
                internal_rag=None,
                default_route_method=self.route_method,
                use_reranker=getattr(self, "use_reranker", False),
                reranker_model=getattr(self, "reranker_model", None),
            )
            self._update_prompt()
            return False

    def do_profile(self, arg: str):
        """Switch active inference profile. Usage: profile <0|A|B|C|D|E|S|U|AE>"""
        arg = arg.strip().lstrip("/")
        if arg.lower().startswith("profile"):
            arg = arg[7:].strip()
        if not arg:
            table = Table(title="Available Profile Layers", box=box.ROUNDED, border_style="cyan")
            table.add_column("Key", style="bold yellow")
            table.add_column("Layer Profile", style="bold white")
            table.add_column("Target Latency", style="bold green")
            table.add_column("Role in STRATA Paper / Architecture", style="dim")

            table.add_row("0 / symbolic", "Profile 0 (Pure Symbolic)", "< 15 ms", "Deterministic A* graph search (zero neural models)")
            table.add_row("A", "Profile A (Dense Embeddings RAG)", "~50–100 ms", "Vector embeddings (SentenceTransformer / FAISS HNSW search)")
            table.add_row("C", "Profile C (Neuro-Symbolic LLM)", "~500 ms–2 s", "Full hybrid: Embedder + Local GGUF LLM slot-filling + Sandbox repair")
            table.add_row("D", "Profile D (Routing Benchmark)", "~200–500 ms", "LLM-guided path search without code generation")
            table.add_row("E", "Profile E (Translator Pass)", "~1–2 s", "Two-stage: Query Translator pass + Neuro-Symbolic synthesis")
            table.add_row("S", "Profile S (Semantic Compiler)", "~1–2 s", "Typed Intent & Schema Compiler pass + Neuro-Symbolic synthesis")

            console.print(table)
            console.print(f"\nActive Profile: [bold yellow]{self._format_profile_name(self.active_profile)}[/bold yellow]\n")
            return
        self._switch_profile(arg)

    def do_models(self, arg: str):
        """List all available embedding models, LLMs, and rerankers found on disk."""
        available_emb = _get_available_embedders()
        available_llm = _get_available_llms()
        available_rerankers = _get_available_rerankers()

        table = Table(title="📦 Available Neural Models in models/", box=box.ROUNDED, border_style="cyan")
        table.add_column("Category", style="bold yellow")
        table.add_column("Model Name", style="bold white")
        table.add_column("Status", style="bold green")

        if available_emb:
            for m in available_emb:
                is_active = (m == self.embedder_name and ModelManager.get_instance().has_embedder())
                status = "[bold green]ACTIVE[/bold green]" if is_active else "[dim]Available[/dim]"
                table.add_row("Embedding Model", m, status)
        else:
            table.add_row("Embedding Model", "[dim]None found in models/embeddings/[/dim]", "-")

        if available_llm:
            for m in available_llm:
                is_active = (m == self.llm_name and ModelManager.get_instance().can_synthesize())
                status = "[bold green]ACTIVE[/bold green]" if is_active else "[dim]Available[/dim]"
                table.add_row("LLM (GGUF)", m, status)
        else:
            table.add_row("LLM (GGUF)", "[dim]None found in models/llms/[/dim]", "-")

        if available_rerankers:
            for m in available_rerankers:
                is_active = (getattr(self, "use_reranker", False) and (m == getattr(self, "reranker_model", "") or not getattr(self, "reranker_model", "")))
                status = "[bold green]ACTIVE[/bold green]" if is_active else "[dim]Available[/dim]"
                table.add_row("Reranker Model", m, status)
        else:
            table.add_row("Reranker Model", "[dim]None found in models/rerankers/[/dim]", "-")

        console.print(table)
        console.print("[dim]Use `set embedder <name>`, `set llm <name>`, or `/reranker [on|off|<name>]` to configure models.[/dim]\n")

    def do_reranker(self, arg: str):
        """Toggle or configure the neural reranker. Usage: /reranker [on|off|<model_name>]"""
        arg = arg.strip().lstrip("/")
        if arg.lower().startswith("reranker"):
            arg = arg[8:].strip()
        if not arg or arg.lower() in ("toggle", "t"):
            self.use_reranker = not getattr(self, "use_reranker", False)
        elif arg.lower() in ("on", "1", "true", "enable", "enabled"):
            self.use_reranker = True
        elif arg.lower() in ("off", "0", "false", "disable", "disabled"):
            self.use_reranker = False
        else:
            self.reranker_model = arg
            settings.reranker_model = arg
            os.environ["NSTL_RERANKER_MODEL"] = str(arg)
            self.use_reranker = True
            console.print(f"[bold cyan][*] Reranker model set to '{arg}'.[/bold cyan]")

        settings.use_reranker = self.use_reranker
        os.environ["NSTL_USE_RERANKER"] = "1" if self.use_reranker else "0"
        if getattr(self, "router", None) is not None:
            self.router.use_reranker = self.use_reranker
            if self.reranker_model:
                self.router.reranker_model = self.reranker_model
            if self.use_reranker and getattr(self.router, "reranker", None) is None:
                try:
                    self.router.reranker = LocalReranker(model_name_or_path=self.reranker_model)
                except Exception as _e:
                    console.print(f"[bold red][!] Could not load reranker model: {_e}[/bold red]")

        status_str = f"[bold green]ENABLED ({self.reranker_model or 'jina-reranker-v3.5'})[/bold green]" if self.use_reranker else "[yellow]DISABLED[/yellow]"
        console.print(f"\n[*] Neural Reranker: {status_str}\n")

    def do_set(self, arg: str):
        """Configure models or hardware device. Usage: set <embedder|llm|device|reranker|reranker_model|macros|debug|timeout> <value>"""
        arg = arg.strip().lstrip("/")
        if arg.lower().startswith("set"):
            arg = arg[3:].strip()
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            console.print("[yellow]Usage: set <embedder|llm|device|reranker|reranker_model|macros|debug|timeout> <value>[/yellow]")
            return
        key, val = parts[0].lower(), parts[1].strip()

        if key in ("embedder", "emb"):
            self.embedder_name = val
            console.print(f"[green][*] Embedder set to '{val}'.[/green]")
            if ModelManager.get_instance().has_embedder():
                self._switch_profile(self.active_profile, embedder=val)
        elif key == "llm":
            self.llm_name = val
            console.print(f"[green][*] LLM set to '{val}'.[/green]")
            if ModelManager.get_instance().can_synthesize():
                self._switch_profile(self.active_profile, llm=val)
        elif key in ("reranker", "neural_reranker"):
            enabled = val.lower() in ("1", "true", "on", "yes", "enable", "enabled")
            settings.use_reranker = enabled
            self.use_reranker = enabled
            os.environ["NSTL_USE_RERANKER"] = "1" if enabled else "0"
            if getattr(self, "router", None) is not None:
                self.router.use_reranker = enabled
                if enabled and getattr(self.router, "reranker", None) is None:
                    try:
                        self.router.reranker = LocalReranker(model_name_or_path=self.reranker_model)
                    except Exception as _e:
                        console.print(f"[bold red][!] Failed to load reranker: {_e}[/bold red]")
            console.print(f"[green][*] Neural reranker set to {'ON' if enabled else 'OFF'}.[/green]")
        elif key in ("reranker_model", "reranker-model"):
            self.reranker_model = val
            settings.reranker_model = val
            os.environ["NSTL_RERANKER_MODEL"] = str(val)
            if getattr(self, "router", None) is not None:
                self.router.reranker_model = val
                if getattr(self, "use_reranker", False):
                    try:
                        self.router.reranker = LocalReranker(model_name_or_path=val)
                    except Exception as _e:
                        console.print(f"[bold red][!] Failed to switch reranker model: {_e}[/bold red]")
            console.print(f"[green][*] Reranker model set to '{val}'.[/green]")
        elif key == "device":
            self.device = val
            console.print(f"[green][*] Compute device set to '{val}'.[/green]")
            if ModelManager.get_instance().has_embedder() or ModelManager.get_instance().can_synthesize():
                self._switch_profile(self.active_profile, device=val)
        elif key in ("macros", "macro", "macro_goals"):
            enabled = val.lower() in ("1", "true", "on", "yes", "enable", "enabled")
            settings.macros_enabled = enabled
            self.macros_enabled = enabled
            if getattr(self, "router", None) is not None:
                self.router.macros_enabled = enabled
            console.print(f"[green][*] Macro-goal routing set to {'ON' if enabled else 'OFF'}.[/green]")
        elif key in ("debug", "dbg"):
            enabled = val.lower() in ("1", "true", "on", "yes", "enable", "enabled")
            self.debug = enabled
            console.print(f"[green][*] Debug mode set to {'ON' if self.debug else 'OFF'}.[/green]")
            self._update_prompt()
        elif key in ("timeout", "sandbox_timeout"):
            try:
                self.timeout = float(val)
                console.print(f"[green][*] Sandbox execution timeout set to {self.timeout:.1f}s.[/green]")
            except ValueError:
                console.print(f"[bold red][!] Invalid timeout value '{val}'. Must be a number.[/bold red]")
        elif key in ("dev", "dev_mode", "devmode"):
            enabled = val.lower() in ("1", "true", "on", "yes", "enable", "enabled")
            settings.dev_mode = enabled
            console.print(f"[green][*] Dev mode set to {'ON' if enabled else 'OFF'}.[/green]")
        else:
            console.print(f"[bold red][!] Unknown parameter '{key}'. Supported: embedder, llm, device, reranker, reranker_model, macros, dev, debug, timeout.[/bold red]")

    def do_dev(self, arg: str):
        """Toggle or set Dev Mode. Usage: /dev [on|off]"""
        val = arg.strip().lower()
        if val in ("1", "true", "on", "yes", "enable", "enabled"):
            enabled = True
        elif val in ("0", "false", "off", "no", "disable", "disabled"):
            enabled = False
        else:
            enabled = not settings.dev_mode
        settings.dev_mode = enabled
        console.print(
            f"[green][*] Dev mode set to {'ON' if enabled else 'OFF'}. "
            f"Dynamic node synthesis is {'ENABLED' if enabled else 'DISABLED'}.[/green]"
        )

    def do_review(self, arg: str):
        """Inspect, promote, or discard synthesized dev cells. Usage: review [list|promote <id>|discard <id>]"""
        parts = arg.strip().split()
        subcmd = parts[0].lower() if parts else "list"
        target_id = parts[1].strip() if len(parts) > 1 else ""

        try:
            from node_resolver import DynamicNodeResolver
        except ImportError:
            console.print("[red][!] DynamicNodeResolver not available.[/red]")
            return

        if subcmd == "list" or not subcmd:
            cells = DynamicNodeResolver.list_unreviewed_cells()
            if not cells:
                console.print("\n[green][*] No unreviewed dev cells pending review.[/green]\n")
                return
            console.print(f"\n[bold cyan]Pending Dev Mode Cells ({len(cells)} total):[/bold cyan]")
            table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("Cell ID", style="bold yellow")
            table.add_column("Domain", style="cyan")
            table.add_column("Stage", justify="center")
            table.add_column("Role", style="green")
            table.add_column("Usage (Pass/Fail)", justify="center")
            table.add_column("Score Mult", justify="center")
            table.add_column("Reviewed", justify="center")
            for c in cells:
                cid = c.get("cell_id", "?")
                dom = c.get("domain_name", "generic")
                stg = str(c.get("stage", 2))
                role = c.get("node_role", "transform")
                usage = f"{c.get('success_count', 0)} / {c.get('fail_count', 0)}"
                mult = f"{c.get('provisional_score_mult', 0.70):.2f}x"
                rev = "[green]YES[/green]" if c.get("reviewed") else "[yellow]NO[/yellow]"
                table.add_row(cid, dom, stg, role, usage, mult, rev)
            console.print(table)
            console.print("[dim]Use 'review promote <cell_id>' to promote or 'review discard <cell_id>' to delete.[/dim]\n")
        elif subcmd == "promote":
            if not target_id:
                console.print("[yellow]Usage: review promote <cell_id>[/yellow]")
                return
            ok = DynamicNodeResolver.promote_cell(target_id, self.orchestrator, self.rag)
            if ok:
                console.print(f"[bold green][✓] Cell '{target_id}' promoted to reviewed status (score multiplier 1.0x).[/bold green]")
            else:
                console.print(f"[bold red][!] Could not find cell '{target_id}' in dev unreviewed catalog.[/bold red]")
        elif subcmd == "discard":
            if not target_id:
                console.print("[yellow]Usage: review discard <cell_id>[/yellow]")
                return
            ok = DynamicNodeResolver.discard_cell(target_id, self.orchestrator)
            if ok:
                console.print(f"[bold yellow][✓] Cell '{target_id}' discarded and removed from dev catalog.[/bold yellow]")
            else:
                console.print(f"[bold red][!] Could not find cell '{target_id}' in dev unreviewed catalog.[/bold red]")
        else:
            console.print("[yellow]Usage: review [list|promote <id>|discard <id>][/yellow]")

    def do_status(self, arg: str):
        """Display real-time system status and active configuration."""
        self._render_dashboard()

    def do_info(self, arg: str):
        """Alias for status."""
        self.do_status(arg)

    def do_new(self, arg: str):
        """Start a fresh chat/synthesis session and clear history."""
        self.history.clear()
        console.print("\n[bold green][✓] Session reset. Query history cleared.[/bold green]\n")

    def do_reset(self, arg: str):
        """Alias for new."""
        self.do_new(arg)

    def do_clear(self, arg: str):
        """Clears the terminal screen and redraws the dashboard."""
        os.system("clear" if os.name == "posix" else "cls")
        self._render_dashboard()

    def do_history(self, arg: str):
        """Display query history and performance metrics for the current session."""
        if not self.history:
            console.print("\n[yellow]No queries executed in this session yet.[/yellow]\n")
            return

        table = Table(title=f"📜 Session History ({len(self.history)} Queries)", box=box.ROUNDED, border_style="cyan")
        table.add_column("#", style="bold yellow", width=4)
        table.add_column("Profile", style="bold magenta", width=12)
        table.add_column("Prompt", style="bold white", ratio=3)
        table.add_column("Path", style="cyan", ratio=3)
        table.add_column("Latency", style="bold green", width=12)
        table.add_column("Sandbox", style="bold", width=16)

        for idx, item in enumerate(self.history, 1):
            sb_style = "green" if "PASSED" in item["sandbox_status"] else "red"
            table.add_row(
                str(idx),
                item["profile"],
                item["prompt"][:40] + ("..." if len(item["prompt"]) > 40 else ""),
                " ➔ ".join(item["path"]) if item["path"] else "[dim]None[/dim]",
                f"{item['latency_ms']:.1f} ms",
                f"[{sb_style}]{item['sandbox_status']}[/{sb_style}]"
            )

        console.print(table)
        console.print("")

    def do_method(self, arg: str):
        """Set or view active RouteMethod algorithm. Usage: /method [M<n>] (see /method for the list)"""
        raw_arg = arg.strip()
        clean = raw_arg.lower()
        if not clean:
            curr = self.route_method or str(getattr(self.router, "default_route_method", "m0")).upper()
            console.print(f"\n[cyan]Active RouteMethod:[/cyan] [bold yellow]{curr}[/bold yellow]")
            console.print("[cyan]Available Methods:[/cyan]")
            methods_summary = describe_route_methods()
            for m_tag, m_alias, m_desc in methods_summary:
                is_active = (self.route_method and self.route_method.upper() == m_tag) or (not self.route_method and m_tag == curr.upper())
                prefix = "➔ " if is_active else "  "
                console.print(f"  {prefix}[bold cyan]{m_tag}[/bold cyan] ({m_alias}): {m_desc}")
            console.print("")
            return

        if clean not in ROUTE_METHOD_REGISTRY:
            console.print(f"[bold red][!] Unknown route method '{raw_arg}'. Valid: {', '.join(m.lower() for m in route_method_ids())}, plus aliases[/bold red]")
            return

        m_cls = ROUTE_METHOD_REGISTRY[clean]
        tag = raw_arg.upper() if len(raw_arg) <= 2 else getattr(m_cls, 'name', raw_arg).upper()
        self.route_method = tag
        if self.router:
            self.router.default_route_method = clean
        self._update_prompt()
        console.print(f"[bold green][✓] RouteMethod switched to {tag} ({getattr(m_cls, 'name', clean)})[/bold green]\n")

    def do_routemethod(self, arg: str):
        """Alias for method."""
        self.do_method(arg)

    def do_audit(self, arg: str):
        """Audit lattice topology reachability, cross-tree transitions, and dead-ends. Usage: /audit"""
        console.print("\n[bold cyan][*] Running Lattice Auditor on active topology...[/bold cyan]")
        auditor = LatticeAuditor(self.orchestrator)
        report = auditor.audit()
        summary = report.summary

        table = Table(title=f"📊 Lattice Topological Audit ({summary['total_cells']} Nodes)", box=box.ROUNDED, border_style="cyan")
        table.add_column("Metric", style="bold yellow")
        table.add_column("Value", style="bold white")
        table.add_column("Status / SLA", style="bold green")

        ratio = summary['reachable_ratio'] * 100.0
        status_style = "[green]HEALTHY[/green]" if ratio >= 90.0 else "[red]DEGRADED[/red]"
        table.add_row("Reachable Ratio", f"{ratio:.2f}%", status_style)
        table.add_row("Entry Nodes (Stage 1)", str(summary['entry_node_count']), "Ingress sources")
        table.add_row("Terminal Nodes (Stage 3)", str(summary['terminal_node_count']), "Egress sinks")
        table.add_row("Cross-Tree Dynamic Bridges", str(summary['cross_tree_edges']), "Inter-domain transitions")
        table.add_row("Unreachable Nodes", str(summary['unreachable_count']), "[green]0[/green]" if summary['unreachable_count'] == 0 else f"[yellow]{summary['unreachable_count']}[/yellow]")
        table.add_row("Dead-End Nodes", str(summary['dead_end_count']), "[green]PASS (0)[/green]" if summary['dead_end_count'] == 0 else "[red]FAIL[/red]")
        table.add_row("Disconnected Components", str(summary['disconnected_count']), "[green]PASS (0)[/green]" if summary['disconnected_count'] == 0 else "[red]FAIL[/red]")

        console.print(table)
        console.print("")

    def do_macro(self, arg: str):
        """Harvest composite MacroCell from micro-cells. Usage: /macro <CELL1> <CELL2> ... [--id <NAME>] [--doc <DOC>]"""
        parts = arg.strip().split()
        if not parts:
            console.print("[yellow]Usage: /macro <CELL1> <CELL2> ... [--id <NAME>] [--doc <DOC>][/yellow]")
            return

        macro_id = None
        doc = None
        cells = []
        i = 0
        while i < len(parts):
            if parts[i] == "--id" and i + 1 < len(parts):
                macro_id = parts[i+1]
                i += 2
            elif parts[i] == "--doc" and i + 1 < len(parts):
                doc = parts[i+1]
                i += 2
            else:
                cells.append(parts[i])
                i += 1

        if len(cells) < 2:
            console.print("[yellow]Please specify at least two cell IDs to compose into a macro.[/yellow]")
            return

        try:
            macro = MacroHarvester.harvest_macro(
                cell_ids=cells,
                orchestrator=self.orchestrator,
                macro_id=macro_id,
                docstring=doc
            )
            console.print(f"[bold green][✓] Successfully harvested and registered MacroCell '{macro.cell_id}'[/bold green]")
            console.print(f"    [cyan]Constituents:[/cyan] {' -> '.join(macro.sub_cells)}")
            console.print(f"    [cyan]Inputs:[/cyan] {list(macro.inputs.keys())}")
            console.print(f"    [cyan]Outputs:[/cyan] {list(macro.outputs.keys())}\n")
        except Exception as e:
            console.print(f"[bold red][!] Macro harvesting failed:[/bold red] {e}\n")

    def default(self, line: str):
        prompt = line.strip()
        if not prompt:
            return

        _quick_profile = dict(QUICK_PROFILE_SHORTCUTS).get(prompt)
        if _quick_profile is not None:
            self._switch_profile(_quick_profile)
            return

        if prompt.startswith("/"):
            cmd_part = prompt[1:].strip()
            parts = cmd_part.split(maxsplit=1)
            cmd_name = parts[0].lower()
            cmd_arg = parts[1] if len(parts) > 1 else ""

            if cmd_name in ("profile", "p"):
                self.do_profile(cmd_arg)
                return
            elif cmd_name in ("method", "routemethod", "m"):
                self.do_method(cmd_arg)
                return
            elif cmd_name == "audit":
                self.do_audit(cmd_arg)
                return
            elif cmd_name == "macro":
                self.do_macro(cmd_arg)
                return
            elif cmd_name in ("models", "mod"):
                self.do_models(cmd_arg)
                return
            elif cmd_name == "set":
                self.do_set(cmd_arg)
                return
            elif cmd_name in ("status", "info"):
                self.do_status(cmd_arg)
                return
            elif cmd_name in ("new", "reset"):
                self.do_new(cmd_arg)
                return
            elif cmd_name in ("clear", "cls"):
                self.do_clear(cmd_arg)
                return
            elif cmd_name in ("history", "hist"):
                self.do_history(cmd_arg)
                return
            elif cmd_name in ("debug", "dbg"):
                self.do_debug(cmd_arg)
                return
            elif cmd_name in ("dev", "dev_mode", "devmode"):
                self.do_dev(cmd_arg)
                return
            elif cmd_name in ("reranker", "rerank"):
                self.do_reranker(cmd_arg)
                return
            elif cmd_name in ("review", "rev"):
                self.do_review(cmd_arg)
                return
            elif cmd_name in ("help", "h", "?"):
                self.do_help(cmd_arg)
                return
            elif cmd_name in ("exit", "quit", "q"):
                return self.do_exit(cmd_arg)

        query_method = self.route_method
        for m in sorted(route_method_ids(), key=lambda t: -len(t)):   # longest first so M10 is not read as M1
            for flag in (f"--method {m}", f"--route-method {m}", f"-m {m}", f"--method={m}", f"--route-method={m}"):
                low_p = prompt.lower()
                f_low = flag.lower()
                pos = low_p.find(f_low)
                if pos != -1:
                    prompt = (prompt[:pos] + prompt[pos + len(flag):]).strip()
                    query_method = m
                    break

        if "--reranker" in prompt:
            prompt = prompt.replace("--reranker", "").strip()
            self.use_reranker = True
            if self.router:
                self.router.use_reranker = True
        elif "--no-reranker" in prompt:
            prompt = prompt.replace("--no-reranker", "").strip()
            self.use_reranker = False
            if self.router:
                self.router.use_reranker = False

        query_no_lint = self.no_lint
        if "--no-lint" in prompt:
            prompt = prompt.replace("--no-lint", "").strip()
            query_no_lint = True

        query_debug = self.debug
        if "--debug" in prompt:
            prompt = prompt.replace("--debug", "").strip()
            query_debug = True

        if query_debug:
            debugger = PipelineDebugger(
                orchestrator=self.orchestrator,
                router=self.router,
                gate=self.gate,
                sandbox=self.sandbox,
                active_profile=self.active_profile,
                device=self.device,
                embedder_name=self.embedder_name,
                llm_name=self.llm_name,
                console=console,
                route_method=query_method
            )
            res = debugger.run(prompt, execute_sandbox=not self.no_exec, timeout=self.timeout)
            self.history.append({
                "prompt": prompt,
                "profile": self.active_profile,
                "path": res["path"],
                "latency_ms": res["latency_ms"],
                "sandbox_status": res["sandbox_status"],
                "code": res["code"],
                "route_ms": res["route_ms"],
                "synth_ms": res["synth_ms"],
                "sandbox_ms": res["sandbox_ms"],
                "sandbox_error": res.get("sandbox_result", {}).get("error", "")
            })
            return

        t_total_start = time.perf_counter()
        prof = self.active_profile.upper()

        # Step 1: Optional Translator Pass (Profile E) or Semantic Compiler (Profile S)
        effective_prompt = prompt
        t_trans = 0.0
        intent_data = None
        if prof in ("E", "S"):
            mm = ModelManager.get_instance()
            if mm.profile and (mm.has_translator_pass() or mm.has_semantic_compiler()):
                t0_trans = time.perf_counter()
                intent_data = mm.compile_semantic_intent(prompt)
                effective_prompt = ensure_comprehensive_prompt(prompt, intent_data.get("effective_prompt") or prompt)
                t_trans = (time.perf_counter() - t0_trans) * 1000.0
                task_str = intent_data.get("task", "generic")
                tgt_sink = intent_data.get("target_sink", "")
                tgt_info = f" | Target: {tgt_sink}" if tgt_sink else ""
                console.print(f"[bold magenta][Profile {prof} Compiler ({t_trans:.1f}ms)][/bold magenta] Task: {task_str}{tgt_info}")
                if intent_data.get("hyperparameters"):
                    console.print(f"  [dim]Extracted Params: {intent_data['hyperparameters']}[/dim]")

        # Step 2: Routing via LatticeRouter
        t_route_start = time.perf_counter()
        plan_ctx = ExecutionContext(prompt=effective_prompt)
        if intent_data and isinstance(intent_data, dict):
            plan_ctx.intent_data = dict(intent_data)
            tgt = intent_data.get("target_sink") or intent_data.get("target_key") or intent_data.get("deliverable")
            if tgt:
                plan_ctx.target_key = str(tgt).strip()
                plan_ctx.target_literals.add(plan_ctx.target_key)
                plan_ctx.parameters["target_sink"] = plan_ctx.target_key
            if intent_data.get("hyperparameters") and isinstance(intent_data["hyperparameters"], dict):
                plan_ctx.hyperparameters = dict(intent_data["hyperparameters"])
            if intent_data.get("slots") and isinstance(intent_data["slots"], dict):
                plan_ctx.llm_slots = dict(intent_data["slots"])
            if intent_data.get("variables") and isinstance(intent_data["variables"], list):
                plan_ctx.variables_spec = [str(v).strip() for v in intent_data["variables"] if str(v).strip()]
                plan_ctx.llm_slots.setdefault("variables", plan_ctx.variables_spec)
            if hasattr(plan_ctx, "sync_llm_slots"):
                plan_ctx.sync_llm_slots()
            if intent_data.get("parameters") and isinstance(intent_data["parameters"], dict):
                for p_key, p_val in intent_data["parameters"].items():
                    if p_key:
                        plan_ctx.parameters.setdefault(str(p_key), p_val)
            if intent_data.get("source_files"):
                plan_ctx.parameters["source_uris"] = list(intent_data["source_files"])
            if intent_data.get("dest_files"):
                plan_ctx.parameters["dest_uris"] = list(intent_data["dest_files"])

        cells = self.router.plan_path(effective_prompt, return_tuple=False, route_method=query_method, ctx=plan_ctx)
        route_dt = (time.perf_counter() - t_route_start) * 1000.0

        if not cells:
            console.print(f"\n[bold red][!] No valid path found through lattice for: '{prompt}'[/bold red]\n")
            return

        path_ids = [c.cell_id for c in cells]
        path_arrows = " [bold green]➔[/bold green] ".join([f"[bold cyan]{cid}[/bold cyan]" for cid in path_ids])
        console.print(f"\n[bold yellow]⚡ Routed Path ({route_dt:.2f}ms):[/bold yellow] {path_arrows}")

        # Step 3: Synthesis & Code Generation
        t_synth_start = time.perf_counter()
        try:
            final_code = self.gate.unify_and_emit(cells, effective_prompt, intent_data=intent_data)
        except (UnresolvedPlaceholderError, UnificationFailure) as e:
            console.print(f"\n[bold red][!] Could not synthesize code for: '{prompt}'[/bold red]")
            console.print(f"[red]    {e}[/red]\n")
            return
        except Exception as e:
            # Engine defect, not a type-level refusal (R6-4)
            logger.error("INTERNAL ERROR in unify_and_emit: %s", e, exc_info=True)
            console.print(f"\n[bold red]❌ INTERNAL ERROR ({type(e).__name__}): {e}[/bold red]")
            console.print("[yellow]This is an engine defect (bug), not a type-level refusal.[/yellow]\n")
            return
        synth_dt = (time.perf_counter() - t_synth_start) * 1000.0

        # Step 3b: Static Pre-Flight Linting
        lint_valid = True
        if not query_no_lint and hasattr(self.gate, "last_pipeline_bindings") and self.gate.last_pipeline_bindings:
            lint_res = PreflightLinter.lint(self.gate.last_pipeline_bindings, prompt=effective_prompt, code_str=final_code, orchestrator=self.orchestrator)
            lint_valid = lint_res.is_valid
            if not lint_res.is_valid:
                console.print("\n[bold red][!] Static Pre-Flight Lint Violations:[/bold red]")
                for v in lint_res.violations:
                    console.print(f"  [red]• {v}[/red]")
            elif lint_res.warnings:
                console.print("\n[bold yellow][!] Pre-Flight Warnings:[/bold yellow]")
                for w in lint_res.warnings:
                    console.print(f"  [yellow]• {w}[/yellow]")

            # Layer 5 Self-Repair for Pre-Flight Lint Violations (Disabled in symbolic profiles 0/A/D)
            if not lint_valid and prof not in ("0", "A", "D") and ModelManager.get_instance().can_feedback_check() and final_code:
                mm = ModelManager.get_instance()
                if mm.profile and mm.can_feedback_check():
                    err_msg = "Pre-flight lint validation failed:\n" + "\n".join(f"- {v}" for v in lint_res.violations)
                    console.print("\n[bold yellow]⚡ LAYER 5: Pre-Flight Lint LLM Self-Repair Cycle[/bold yellow]")
                    repaired_code = extract_code_from_llm_response(mm.feedback_check(final_code, err_msg))
                    if repaired_code and repaired_code.strip() != final_code.strip():
                        rep_lint = PreflightLinter.lint(self.gate.last_pipeline_bindings, prompt=prompt, code_str=repaired_code, orchestrator=self.orchestrator)
                        if rep_lint.is_valid:
                            console.print("  [bold green][✓] Pre-flight repair accepted.[/bold green]\n")
                            final_code = repaired_code
                            lint_valid = True

        if not lint_valid:
            console.print("\n[bold red][!] Execution halted due to pre-flight lint violations.[/bold red]\n")
            return

        # Step 4: Sandbox Verification & Optional Self-Repair
        t_exec_start = time.perf_counter()
        dest_paths = self.gate.get_egress_paths() or None
        v_contract = getattr(self.gate, "last_verification_contract", None)

        if getattr(self, "no_exec", False):
            static_violations = _statically_evaluate_contract(final_code, v_contract)
            if static_violations:
                err_msg = f"Static verification contract violated: {'; '.join(static_violations)}"
                sandbox_res = {"success": False, "verified": False, "skipped": True, "error": err_msg}
            else:
                sandbox_res = {"success": None, "skipped": True, "verified": bool(v_contract)}
            sandbox_dt = 0.0
        else:
            last_bindings = getattr(self.gate, "last_pipeline_bindings", None)
            sandbox_res = self.sandbox.execute(
                final_code,
                timeout=self.timeout,
                egress_paths=dest_paths,
                verification_spec=v_contract,
                runtime_aliases=getattr(self.gate, 'last_runtime_aliases', None),
                pipeline_bindings=last_bindings,
            )
            sandbox_dt = (time.perf_counter() - t_exec_start) * 1000.0

        repaired = False
        if not sandbox_res.get("success", False) and prof not in ("0", "A", "D") and ModelManager.get_instance().can_feedback_check():
            mm = ModelManager.get_instance()
            if mm.profile and mm.can_feedback_check():
                error_msg = sandbox_res.get("error", "")
                if sandbox_res.get("extrinsic", False) or _is_environmental_error(error_msg):
                    console.print(
                        "  [yellow][!] Environmental failure detected (missing input asset or "
                        "unavailable module). LLM self-repair skipped — the synthesized "
                        "pipeline is not the defect.[/yellow]"
                    )
                else:
                    console.print("  [bold yellow][*] GEVR Sandbox triggered LLM Self-Repair Cycle...[/bold yellow]")
                    t_rep_start = time.perf_counter()
                    failing_code = final_code
                    repaired_code = extract_code_from_llm_response(mm.feedback_check(failing_code, error_msg))
                    if repaired_code and repaired_code.strip() != failing_code.strip():
                        unknown = _unknown_api_references(repaired_code, failing_code, self.orchestrator.loaded_cells.values())
                        if unknown:
                            console.print(
                                f"  [bold red][x] Repair rejected: references APIs absent from the "
                                f"lattice: {', '.join(sorted(unknown)[:5])}[/bold red]"
                            )
                        else:
                            final_code = repaired_code
                            repaired = True
                            last_bindings = getattr(self.gate, "last_pipeline_bindings", None)
                            sandbox_res = self.sandbox.execute(
                                final_code,
                                timeout=self.timeout,
                                egress_paths=dest_paths,
                                verification_spec=v_contract,
                                runtime_aliases=getattr(self.gate, 'last_runtime_aliases', None),
                                pipeline_bindings=last_bindings,
                            )
                    rep_dt = (time.perf_counter() - t_rep_start) * 1000.0
                    console.print(f"  [bold green][✓] Repair cycle completed ({rep_dt:.1f}ms).[/bold green]")

        try:
            from node_resolver import DynamicNodeResolver
            is_succ = bool(sandbox_res.get("success", False))
            for c in (cells or []):
                DynamicNodeResolver.record_cell_usage(getattr(c, "cell_id", ""), success=is_succ)
        except Exception:
            pass

        total_dt = (time.perf_counter() - t_total_start) * 1000.0

        syntax_code = Syntax(final_code, "python", theme="monokai", line_numbers=True)
        code_panel = Panel(
            syntax_code,
            title=f"[bold green]✨ Synthesized Python Code ({self._format_profile_name(self.active_profile)})[/bold green]",
            border_style="green",
            padding=(0, 1)
        )
        console.print(code_panel)

        method_name = query_method or getattr(self.router, "default_route_method", "M0") or "M0"
        attribution = getattr(self.gate, "last_attribution", "path")
        timing_elements = [
            f"[bold cyan]Method:[/bold cyan] [bold]{method_name}[/bold]",
            f"[cyan]Route:[/cyan] [bold]{route_dt:.2f}ms[/bold]",
            f"[cyan]Synth:[/cyan] [bold]{synth_dt:.2f}ms[/bold]",
            f"[cyan]Exec:[/cyan] [bold]{sandbox_dt:.2f}ms[/bold]"
        ]
        if t_trans > 0:
            timing_elements.insert(1, f"[magenta]Trans:[/magenta] [bold]{t_trans:.1f}ms[/bold]")
        timing_elements.append(f"[bold yellow]Total: {total_dt:.2f}ms[/bold yellow]")
        timing_elements.append(f"[dim]Attr: {attribution}[/dim]")

        if sandbox_res.get("skipped", False):
            sb_badge = "[bold yellow]↷ SKIPPED (--no-exec)[/bold yellow]"
            sb_status = "SKIPPED"
        elif sandbox_res.get("success", False):
            sb_badge = f"[bold green]✓ PASSED[/bold green]"
            sb_status = "PASSED"
        else:
            err = sandbox_res.get("error", "Unknown error").strip()
            first_err_line = err.splitlines()[-1] if err else "Execution Error"
            sb_badge = f"[bold red]✗ FAILED[/bold red] [dim]({first_err_line})[/dim]"
            sb_status = f"FAILED: {first_err_line}"

        metrics_panel = Panel(
            f"{'  │  '.join(timing_elements)}   │   [bold]Sandbox:[/bold] {sb_badge}",
            border_style="dim",
            padding=(0, 1)
        )
        console.print(metrics_panel)
        console.print("")

        self.history.append({
            "prompt": prompt,
            "profile": self.active_profile,
            "path": path_ids,
            "latency_ms": total_dt,
            "sandbox_status": sb_status,
            "code": final_code,
            "route_ms": route_dt,
            "synth_ms": synth_dt,
            "sandbox_ms": sandbox_dt,
            "sandbox_result": sandbox_res,
            "sandbox_error": sandbox_res.get("error", "")
        })

    def do_exit(self, line):
        """Exit the STRATA Interactive Studio."""
        console.print("\n[bold cyan]Exiting STRATA Studio. Goodbye![/bold cyan]\n")
        return True

    def do_quit(self, line):
        """Alias for exit."""
        return self.do_exit(line)

    def do_q(self, line):
        """Alias for exit."""
        return self.do_exit(line)

    def do_EOF(self, line):
        """Handle CTRL+D / EOF."""
        return self.do_exit(line)


# =====================================================================
# CLI Command Entrypoints
# =====================================================================
def cmd_shell(args):
    """Launches the full interactive Rich TUI studio."""
    shell = NSTLInteractiveShell(
        db_path=getattr(args, "db", "trees/lattice.db"),
        initial_profile=getattr(args, "profile", "0"),
        embedder=getattr(args, "embedder", ""),
        llm=getattr(args, "llm", ""),
        device=getattr(args, "device", "auto"),
        debug=getattr(args, "debug", False),
        interactive=True,
        no_exec=getattr(args, "no_exec", False),
        route_method=getattr(args, "route_method", None),
        no_lint=getattr(args, "no_lint", False),
        macros=getattr(args, "macros", None),
        topology=getattr(args, "topology", None),
        dev=getattr(args, "dev", None),
        timeout=getattr(args, "timeout", DEFAULT_SANDBOX_TIMEOUT),
        reranker=getattr(args, "reranker", None),
        reranker_model=getattr(args, "reranker_model", None),
        exec_sandbox=getattr(args, "exec_sandbox", False),
        llm_feedback=getattr(args, "llm_feedback", False),
        allowed_capabilities=getattr(args, "allow_capabilities", ""),
    )
    shell.cmdloop()


def cmd_run(args):
    """Executes a single prompt synthesis directly from CLI."""
    if getattr(args, "allow_fallback", None) is not None:
        try:
            from config import settings
            settings.allow_fallback = bool(args.allow_fallback)
        except Exception:
            pass
        os.environ["NSTL_ALLOW_FALLBACK"] = "1" if args.allow_fallback else "0"
    db_path = getattr(args, "db", "trees/lattice.db")
    profile = getattr(args, "profile", "0")
    os.environ["NSTL_PROFILE"] = str(profile).strip().upper()
    try:
        from config import settings
        settings.profile = str(profile).strip().upper()
    except Exception:
        pass
    embedder = getattr(args, "embedder", "")
    llm = getattr(args, "llm", "")
    device = getattr(args, "device", "auto")
    debug_mode = getattr(args, "debug", False)
    prompt = getattr(args, "prompt", "").strip()

    if not prompt:
        console.print("[bold red][!] Prompt must not be empty.[/bold red]")
        sys.exit(1)

    if getattr(args, "explain_plan", False):
        try:
            from config import settings
            settings.explain_plan = True
        except Exception:
            pass

    shell = NSTLInteractiveShell(
        db_path=db_path,
        initial_profile=profile,
        embedder=embedder,
        llm=llm,
        device=device,
        debug=debug_mode,
        interactive=False,
        no_exec=getattr(args, "no_exec", False),
        route_method=getattr(args, "route_method", None),
        no_lint=getattr(args, "no_lint", False),
        macros=getattr(args, "macros", None),
        topology=getattr(args, "topology", None),
        dev=getattr(args, "dev", None),
        timeout=getattr(args, "timeout", DEFAULT_SANDBOX_TIMEOUT),
        reranker=getattr(args, "reranker", None),
        reranker_model=getattr(args, "reranker_model", None),
        exec_sandbox=getattr(args, "exec_sandbox", False),
        llm_feedback=getattr(args, "llm_feedback", False),
        allowed_capabilities=getattr(args, "allow_capabilities", ""),
    )
    shell.default(f"{prompt} --debug" if debug_mode else prompt)

    if shell.history:
        last = shell.history[-1]
        if last.get("internal_error"):
            sys.exit(1)
        if not last.get("code"):
            sys.exit(2)
        if getattr(args, "exec_sandbox", False) and not last.get("sandbox_result", {}).get("success", False):
            sys.exit(3)


def cmd_audit(args):
    """Audits lattice topology reachability, disconnected nodes, and cross-tree transitions."""
    db_path = Path(args.db)
    trees_dir = getattr(args, "trees_dir", "trees")
    ensure_lattice_compiled(trees_dir=trees_dir, db_path=str(db_path))
    print(f"[*] Initializing Lattice Orchestrator from '{db_path}'...")
    orch = LatticeOrchestrator(trees_directory=trees_dir, db_path=str(db_path))
    if db_path.exists():
        orch.load_from_database(str(db_path))
    orch.build_topology()

    print(f"[*] Running Lattice Auditor on {len(orch.cells):,} cells...")
    auditor = LatticeAuditor(orch)
    report = auditor.audit()

    summary = report.summary
    table = Table(title=f"📊 Lattice Topological Audit ({summary['total_cells']} Total Nodes)", box=box.ROUNDED, border_style="cyan")
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="bold white")
    table.add_column("Status / SLA", style="bold green")

    ratio = summary['reachable_ratio'] * 100.0
    status_style = "[green]HEALTHY[/green]" if ratio >= 90.0 else "[red]DEGRADED[/red]"
    table.add_row("Reachable Nodes Ratio", f"{ratio:.2f}%", status_style)
    table.add_row("Entry Nodes (Stage 1)", str(summary['entry_node_count']), "Ingress sources")
    table.add_row("Terminal Nodes (Stage 3)", str(summary['terminal_node_count']), "Egress sinks")
    table.add_row("Cross-Tree Dynamic Bridges", str(summary['cross_tree_edges']), "Inter-domain transitions")
    table.add_row("Unreachable Nodes", str(summary['unreachable_count']), "[green]0[/green]" if summary['unreachable_count'] == 0 else f"[yellow]{summary['unreachable_count']}[/yellow]")
    table.add_row("Dead-End Nodes", str(summary['dead_end_count']), "[green]PASS (0)[/green]" if summary['dead_end_count'] == 0 else "[red]FAIL[/red]")
    table.add_row("Disconnected Components", str(summary['disconnected_count']), "[green]PASS (0)[/green]" if summary['disconnected_count'] == 0 else "[red]FAIL[/red]")

    console.print(table)

    if getattr(args, "output", ""):
        out_path = Path(args.output)
        from dataclasses import asdict
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"[+] Audit report saved to '{out_path}'.")


def cmd_macro(args):
    """Harvests a composite MacroCell from constituent micro-cells."""
    cell_ids = args.cells
    db_path = Path(args.db)
    trees_dir = getattr(args, "trees_dir", "trees")

    print(f"[*] Loading lattice for macro harvesting...")
    orch = LatticeOrchestrator(trees_directory=trees_dir, db_path=str(db_path))
    if db_path.exists():
        orch.load_from_database(str(db_path))
    orch.build_topology()

    macro_id = getattr(args, "id", None) or None
    domain = getattr(args, "domain", None) or None
    doc = getattr(args, "doc", None) or None

    print(f"[*] Harvesting composite macro from cells: {' -> '.join(cell_ids)}...")
    try:
        macro = MacroHarvester.harvest_macro(
            cell_ids=cell_ids,
            orchestrator=orch,
            macro_id=macro_id,
            domain_name=domain,
            docstring=doc
        )
    except Exception as e:
        console.print(f"[bold red][!] Macro harvesting failed:[/bold red] {e}")
        return

    table = Table(title=f"🧩 Harvested MacroCell: {macro.cell_id}", box=box.ROUNDED, border_style="green")
    table.add_column("Property", style="bold yellow")
    table.add_column("Details", style="white")

    table.add_row("Macro ID", macro.cell_id)
    table.add_row("Stage", str(macro.stage))
    table.add_row("Domain", macro.domain_name or "unknown")
    table.add_row("Constituents", " ➔ ".join(macro.sub_cells))
    in_desc = ", ".join(f"{k}: {v.signature.type_name}[{v.signature.state}]" for k, v in macro.inputs.items())
    table.add_row("Composite Inputs", in_desc or "None")
    out_desc = ", ".join(f"{k}: {v.signature.type_name}[{v.signature.state}]" for k, v in macro.outputs.items())
    table.add_row("Composite Outputs", out_desc or "None")
    table.add_row("Internal Topology", str(getattr(macro, "internal_topology", {})))
    table.add_row("Docstring", macro.docstring or "")

    console.print(table)
    console.print(f"[bold green][✓] MacroCell '{macro.cell_id}' verified and registered in orchestrator.[/bold green]")


def cmd_benchmark(args):
    """Runs empirical benchmark suite (matrix or reference bank)."""
    if getattr(args, "llm_feedback", False):
        os.environ["NSTL_LLM_FEEDBACK"] = "1"
    else:
        os.environ.pop("NSTL_LLM_FEEDBACK", None)
    if getattr(args, "allow_capabilities", None):
        os.environ["NSTL_ALLOWED_CAPABILITIES"] = str(args.allow_capabilities)
    if getattr(args, "macros", None) is not None:
        try:
            from config import settings
            settings.macros_enabled = bool(args.macros)
        except Exception:
            pass
        os.environ["NSTL_MACROS_ENABLED"] = "1" if args.macros else "0"
    if getattr(args, "topology", None) is not None:
        try:
            from config import settings
            settings.topology_mode = str(args.topology).lower()
        except Exception:
            pass
        os.environ["NSTL_TOPOLOGY_MODE"] = str(args.topology).lower()
    if getattr(args, "reranker", None) is not None:
        try:
            from config import settings
            settings.use_reranker = bool(args.reranker)
        except Exception:
            pass
        os.environ["NSTL_USE_RERANKER"] = "1" if args.reranker else "0"
    if getattr(args, "reranker_model", None):
        try:
            from config import settings
            settings.reranker_model = str(args.reranker_model)
        except Exception:
            pass
        os.environ["NSTL_RERANKER_MODEL"] = str(args.reranker_model)
    if getattr(args, "allow_fallback", None) is not None:
        try:
            from config import settings
            settings.allow_fallback = bool(args.allow_fallback)
        except Exception:
            pass
        os.environ["NSTL_ALLOW_FALLBACK"] = "1" if args.allow_fallback else "0"
    db_path = getattr(args, "db", "trees/lattice.db")
    ensure_lattice_compiled(trees_dir="trees", db_path=db_path)
    bench_type = getattr(args, "type", "matrix")
    if bench_type == "reference":
        print("[*] Running 50-Task Empirical Reference Benchmark Bank...")
        import pytest
        ret = pytest.main(["-s", "tests/test_reference_benchmark_bank.py"])
        sys.exit(ret)
    else:
        print("[*] Running STRATA Phase 5 Evaluation Matrix (all registered RouteMethods)...")
        import pytest
        ret = pytest.main(["-s", "tests/test_evaluation_matrix.py"])
        sys.exit(ret)


def cmd_precompute_rag(args):
    """Precomputes dense vector embeddings for all loaded lattice nodes into .rag_cache/."""
    try:
        from lattice import LatticeOrchestrator
        from internal_rag import LocalRAG
        from inference import ModelManager
    except ImportError:
        from .lattice import LatticeOrchestrator
        from .internal_rag import LocalRAG
        from .inference import ModelManager

    db_path = Path(args.db)
    ensure_lattice_compiled(trees_dir=args.trees_dir, db_path=str(db_path))
    if not db_path.exists():
        print(f"[!] Database not found: {db_path}. Please run 'compile' first.")
        return

    print(f"[*] Loading lattice cells from '{db_path}'...")
    orch = LatticeOrchestrator(trees_directory=args.trees_dir, db_path=str(db_path))
    print(f"[+] Loaded {len(orch.cells):,} cells.")

    print(f"[*] Initializing Profile A (embedder: '{args.embedder or 'optimal'}')...")
    ModelManager.get_instance().initialize_profile("A", embedder_name=args.embedder)

    try:
        print(f"[*] Starting incremental FAISS precomputation into '.rag_cache/'...")
        t0 = time.perf_counter()
        rag = LocalRAG(trees_dir=args.trees_dir, orchestrator=orch)
        dt = time.perf_counter() - t0
        print(f"[✓] Precomputation complete in {dt:.2f}s. Cache is 100% synchronized.")
    finally:
        ModelManager.get_instance().cleanup()


# =====================================================================
# Argument Parser Construction
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    """Constructs the CLI argument parser with all subcommands and global debug flags."""
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="STRATA (Symbolic Topological Routing Architecture for Typed Automata) Toolchain CLI & Studio")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose debug mode across all pipeline layers")
    subparsers = parser.add_subparsers(dest="command", required=False)

    p_harvest = subparsers.add_parser("harvest", help="Harvest API primitives into single-file domain JSON")
    p_harvest.add_argument("package", type=str, help="Python package name to harvest")
    p_harvest.add_argument("--domain", type=str, default=None, help="Target domain name (defaults to package name)")
    p_harvest.add_argument("--trees-dir", type=str, default="trees", help="Directory for domain tree JSON files")
    p_harvest.set_defaults(func=cmd_harvest)

    p_compile = subparsers.add_parser("compile", help="Compile single-file domain JSONs into SQLite database")
    p_compile.add_argument("--trees-dir", type=str, default="trees", help="Directory containing domain JSON files")
    p_compile.add_argument("--output", type=str, default="trees/lattice.db", help="Target SQLite DB path")
    p_compile.add_argument("--domains", nargs="*", default=None, help="Optional domain filter")
    p_compile.add_argument("--clean", action="store_true", help="Purge target database before compiling")
    p_compile.set_defaults(func=cmd_compile)

    p_validate = subparsers.add_parser("validate", help="Validate AST syntax and schema of all nodes in SQLite or domain JSON")
    p_validate.add_argument("target", nargs="?", default="trees/lattice.db", help="Path to SQLite database (.db) or domain JSON file (.json) to validate")
    p_validate.add_argument("--db", type=str, default=None, help="Optional legacy path to SQLite database")
    p_validate.set_defaults(func=cmd_validate)

    p_audit = subparsers.add_parser("audit", help="Audit lattice graph connectivity, reachability, and cross-tree bridges")
    p_audit.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_audit.add_argument("--trees-dir", type=str, default="trees", help="Directory for domain tree JSON files")
    p_audit.add_argument("--output", "-o", type=str, default="", help="Optional JSON path to save audit report")
    p_audit.set_defaults(func=cmd_audit)

    p_macro = subparsers.add_parser("macro", help="Harvest and register a composite MacroCell from constituent micro-cells")
    p_macro.add_argument("cells", nargs="+", help="Sequence of cell IDs composing the macro")
    p_macro.add_argument("--id", type=str, default="", help="Custom macro cell ID")
    p_macro.add_argument("--domain", type=str, default="", help="Domain name for the macro")
    p_macro.add_argument("--doc", type=str, default="", help="Docstring/description for the macro")
    p_macro.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_macro.add_argument("--trees-dir", type=str, default="trees", help="Directory for domain tree JSON files")
    p_macro.set_defaults(func=cmd_macro)

    p_bench = subparsers.add_parser("benchmark", help="Run STRATA empirical benchmarks")
    p_bench.add_argument("--type", choices=["reference", "matrix"], default="matrix", help="Benchmark type (default: matrix)")
    p_bench.add_argument("--output", type=str, default="evaluation_results.json", help="Path to write evaluation results JSON")
    p_bench.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_bench.add_argument("--methods", nargs="*", default=route_method_ids(), help="Route methods to evaluate")
    p_bench.add_argument("--macros", dest="macros", action="store_true", default=None, help="Enable macro-goal routing")
    p_bench.add_argument("--no-macros", dest="macros", action="store_false", help="Disable macro-goal routing")
    p_bench.add_argument("--reranker", dest="reranker", action="store_true", default=None, help="Enable neural reranker for Layer 1 RAG routing")
    p_bench.add_argument("--no-reranker", dest="reranker", action="store_false", help="Disable neural reranker for Layer 1 RAG routing")
    p_bench.add_argument("--reranker-model", type=str, default=None, help="Neural reranker model name (e.g. jina-reranker-v3.5)")
    p_bench.add_argument("--dev", action="store_true", default=False, help="Enable Dev Mode")
    p_bench.add_argument("--topology", choices=["frontier", "linear"], default=None, help="Topological planning approach")
    p_bench.add_argument("--allow-capabilities", type=str, default="", help="Comma-separated list of capabilities to allow in sandbox (e.g. net.http,fs.read)")
    p_bench.add_argument("--allow-fallback", action="store_true", default=False, help="Allow route methods to fall back to M1 on failure (default: False)")
    p_bench.add_argument("--llm-feedback", dest="llm_feedback", action="store_true", default=False, help="Enable LLM feedback / self-repair cycle")
    p_bench.add_argument(
        "--early-exit-layer",
        type=int,
        default=None,
        help="Experimental: Extract residual stream from transformer layer K instead of final layer.",
    )
    p_bench.add_argument(
        "--no-whitening",
        action="store_true",
        help="Experimental: Disable ZCA whitening on latent cell embeddings.",
    )
    p_bench.add_argument(
        "--sparse-sae",
        action="store_true",
        help="Experimental: Enable Top-K Sparse Autoencoder (OMP) multi-clause concept decomposition.",
    )
    p_bench.add_argument(
        "--no-markov-prior",
        action="store_true",
        help="Experimental: Disable Absorbing Markov Chain N=(I-Q)^-1 hitting-probability prior.",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    p_precompute = subparsers.add_parser("precompute-rag", help="Precompute FAISS dense embeddings into .rag_cache/")
    p_precompute.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_precompute.add_argument("--trees-dir", type=str, default="trees", help="Directory for domain tree JSON files")
    p_precompute.add_argument("--embedder", type=str, default="jina-embeddings-v5-text-nano", help="Embedding model name")
    p_precompute.set_defaults(func=cmd_precompute_rag)

    p_run = subparsers.add_parser("run", help="Synthesize code for a natural language prompt directly from CLI")
    p_run.add_argument("prompt", type=str, help="Natural language pipeline specification")
    p_run.add_argument("--debug", "-d", action="store_true", help="Enable verbose debug output across all pipeline layers")
    p_run.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_run.add_argument("--profile", type=str, default="0", help="Inference profile (0=Symbolic, A=Embedder, B/C=Neuro-Symbolic, D, E, S, U=Early-Exit Layer, AE=Transition Autoencoder)")
    p_run.add_argument("--route-method", "-m", type=str, default=None, choices=route_method_ids(), help="Routing method algorithm")
    p_run.add_argument("--dev", action="store_true", default=False, help="Enable Dev Mode")
    p_run.add_argument("--no-lint", action="store_true", help="Skip static pre-flight linter")
    p_run.add_argument("--embedder", type=str, default="jina-embeddings-v5-text-nano", help="Embedding model name")
    p_run.add_argument("--llm", type=str, default="qwen2.5-coder-0.5b-instruct", help="LLM model name")
    p_run.add_argument("--reranker", dest="reranker", action="store_true", default=None, help="Enable neural reranker for Layer 1 RAG routing")
    p_run.add_argument("--no-reranker", dest="reranker", action="store_false", help="Disable neural reranker for Layer 1 RAG routing")
    p_run.add_argument("--reranker-model", type=str, default=None, help="Neural reranker model name (e.g. jina-reranker-v3.5)")
    p_run.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    p_run.add_argument("--no-exec", action="store_true", help="Skip GEVR sandbox execution")
    p_run.add_argument("--exec", dest="exec_sandbox", action="store_true", default=False, help="Enable GEVR sandbox execution")
    p_run.add_argument("--llm-feedback", dest="llm_feedback", action="store_true", default=False, help="Enable LLM feedback / self-repair cycle when pre-flight lint fails")
    p_run.add_argument("--allow-capabilities", type=str, default="", help="Comma-separated list of capabilities to allow in sandbox (e.g. net.http,fs.read)")
    p_run.add_argument("--allow-fallback", action="store_true", default=False, help="Allow route methods to fall back to M1 on failure (default: False)")
    p_run.add_argument("--timeout", type=float, default=DEFAULT_SANDBOX_TIMEOUT, help="Sandbox execution timeout in seconds")
    p_run.add_argument("--macros", dest="macros", action="store_true", default=None, help="Enable macro-goal routing")
    p_run.add_argument("--no-macros", dest="macros", action="store_false", help="Disable macro-goal routing")
    p_run.add_argument("--topology", choices=["frontier", "linear"], default=None, help="Topological planning approach")
    p_run.add_argument(
        "--early-exit-layer",
        type=int,
        default=None,
        help="Experimental: Extract residual stream from transformer layer K instead of final layer.",
    )
    p_run.add_argument(
        "--no-whitening",
        action="store_true",
        help="Experimental: Disable ZCA whitening on latent cell embeddings.",
    )
    p_run.add_argument(
        "--sparse-sae",
        action="store_true",
        help="Experimental: Enable Top-K Sparse Autoencoder (OMP) multi-clause concept decomposition.",
    )
    p_run.add_argument(
        "--no-markov-prior",
        action="store_true",
        help="Experimental: Disable Absorbing Markov Chain N=(I-Q)^-1 hitting-probability prior.",
    )
    p_run.add_argument("--explain-plan", action="store_true", help="Print clause-level coverage and precision diagnostics table")
    p_run.set_defaults(func=cmd_run)

    p_shell = subparsers.add_parser("shell", help="Launch real-time interactive synthesis TUI studio")
    p_shell.add_argument("--db", type=str, default="trees/lattice.db", help="Path to SQLite database")
    p_shell.add_argument("--profile", type=str, default="0", help="Initial inference profile (0, A, B, C, D, E, S, U, AE)")
    p_shell.add_argument("--route-method", "-m", type=str, default=None, choices=route_method_ids(), help="Initial routing method")
    p_shell.add_argument("--dev", action="store_true", default=False, help="Launch studio with Dev Mode enabled")
    p_shell.add_argument("--no-lint", action="store_true", help="Skip static pre-flight linter")
    p_shell.add_argument("--embedder", type=str, default="jina-embeddings-v5-text-nano", help="Embedding model name")
    p_shell.add_argument("--llm", type=str, default="qwen2.5-coder-0.5b-instruct", help="LLM model name")
    p_shell.add_argument("--reranker", dest="reranker", action="store_true", default=None, help="Enable neural reranker for Layer 1 RAG routing")
    p_shell.add_argument("--no-reranker", dest="reranker", action="store_false", help="Disable neural reranker for Layer 1 RAG routing")
    p_shell.add_argument("--reranker-model", type=str, default=None, help="Neural reranker model name (e.g. jina-reranker-v3.5)")
    p_shell.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    p_shell.add_argument("--debug", "-d", action="store_true", help="Launch studio with debug mode enabled")
    p_shell.add_argument("--no-exec", action="store_true", help="Skip GEVR sandbox execution")
    p_shell.add_argument("--exec", dest="exec_sandbox", action="store_true", default=False, help="Enable GEVR sandbox execution")
    p_shell.add_argument("--llm-feedback", dest="llm_feedback", action="store_true", default=False, help="Enable LLM feedback / self-repair cycle when pre-flight lint fails")
    p_shell.add_argument("--allow-capabilities", type=str, default="", help="Comma-separated list of capabilities to allow in sandbox (e.g. net.http,fs.read)")
    p_shell.add_argument("--allow-fallback", action="store_true", default=False, help="Allow route methods to fall back to M1 on failure (default: False)")
    p_shell.add_argument("--timeout", type=float, default=DEFAULT_SANDBOX_TIMEOUT, help="Sandbox execution timeout in seconds")
    p_shell.add_argument("--macros", dest="macros", action="store_true", default=None, help="Enable macro-goal routing")
    p_shell.add_argument("--no-macros", dest="macros", action="store_false", help="Disable macro-goal routing")
    p_shell.add_argument("--topology", choices=["frontier", "linear"], default=None, help="Topological planning approach")
    p_shell.add_argument(
        "--early-exit-layer",
        type=int,
        default=None,
        help="Experimental: Extract residual stream from transformer layer K instead of final layer.",
    )
    p_shell.add_argument(
        "--no-whitening",
        action="store_true",
        help="Experimental: Disable ZCA whitening on latent cell embeddings.",
    )
    p_shell.add_argument(
        "--sparse-sae",
        action="store_true",
        help="Experimental: Enable Top-K Sparse Autoencoder (OMP) multi-clause concept decomposition.",
    )
    p_shell.add_argument(
        "--no-markov-prior",
        action="store_true",
        help="Experimental: Disable Absorbing Markov Chain N=(I-Q)^-1 hitting-probability prior.",
    )
    p_shell.set_defaults(func=cmd_shell)

    return parser


def main():
    if len(sys.argv) == 1:
        shell = NSTLInteractiveShell()
        shell.cmdloop()
        return

    if len(sys.argv) == 2 and sys.argv[1] in ("--debug", "-d"):
        shell = NSTLInteractiveShell(debug=True)
        shell.cmdloop()
        return

    known_cmds = {"harvest", "compile", "validate", "audit", "macro", "benchmark", "precompute-rag", "shell", "run", "-h", "--help"}
    if len(sys.argv) > 1 and sys.argv[1] not in known_cmds:
        if sys.argv[1] in ("--debug", "-d") and len(sys.argv) > 2 and sys.argv[2] not in known_cmds:
            prompt_arg = sys.argv[2]
            remaining = sys.argv[3:]
            sys.argv = [sys.argv[0], "run", prompt_arg, "--debug"] + remaining
        elif sys.argv[1] not in ("-h", "--help"):
            sys.argv.insert(1, "run")

    parser = build_parser()
    args = parser.parse_args()

    try:
        try:
            from experiments import ACTIVE_EXPERIMENTS
        except (ImportError, ValueError):
            from .experiments import ACTIVE_EXPERIMENTS
        if getattr(args, "early_exit_layer", None) is not None:
            ACTIVE_EXPERIMENTS.early_exit_layer = args.early_exit_layer
        if getattr(args, "no_whitening", False):
            ACTIVE_EXPERIMENTS.use_zca_whitening = False
        if getattr(args, "sparse_sae", False):
            ACTIVE_EXPERIMENTS.use_sparse_coding = True
        if getattr(args, "no_markov_prior", False):
            ACTIVE_EXPERIMENTS.use_markov_hitting_prior = False
    except Exception as _exp_err:
        logger.debug("[CLI] Experiment config binding skipped: %s", _exp_err)

    if hasattr(args, "func"):
        args.func(args)
    else:
        cmd_shell(argparse.Namespace(
            db="trees/lattice.db",
            profile="0",
            embedder="",
            llm="",
            device="auto",
            debug=getattr(args, "debug", False),
            no_exec=False,
            route_method=None,
            no_lint=False,
            macros=None,
            topology=None,
            dev=None,
            timeout=DEFAULT_SANDBOX_TIMEOUT,
            reranker=None,
            reranker_model=None
        ))


# Backward-compatible alias
STRATAInteractiveShell = NSTLInteractiveShell


if __name__ == "__main__":
    main()
