"""
ir_compiler.py — Prompt -> typed IR via the local LLM.

Robust JSON extraction:
  - Balanced-brace scanner (respects strings and escapes).
  - Strips markdown fences.
  - Retries once with a corrective prompt on parse failure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from log_config import get_logger

try:
    from .inference import ModelManager
    from .utils import extract_json_object
except (ImportError, ValueError):
    from inference import ModelManager
    from utils import extract_json_object

logger = get_logger("ir_compiler")

IR_SCHEMA = {
    "type": "object",
    "required": ["steps"],
    "properties": {
        "task_objective": {
            "type": "object",
            "properties": {
                "kind": {"type": "string"},
                "target_scale": {"type": "string"},
            },
        },
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "domain_representation": {"type": "string"},
                    "origin": {"type": "string"},
                },
            },
        },
        "libraries": {"type": "array", "items": {"type": "string"}},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["op"],
                "properties": {
                    "op":          {"type": "string"},
                    "step_kind":   {"type": "string"},
                    "condition":   {"type": "string"},
                    "params":      {"type": "object"},
                    "library":     {"type": "string"},
                    "inputs":      {"type": "array", "items": {"type": "string"}},
                    "outputs":     {"type": "array", "items": {"type": "string"}},
                    "body":        {"type": "array", "items": {"type": "object"}},
                    "then_branch": {"type": "array", "items": {"type": "object"}},
                    "else_branch": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
        "causal_dependencies": {"type": "array", "items": {"type": "array"}},
        "literals": {"type": "object"},
    },
}

PROMPT_TEMPLATE = """\
You are a semantic compiler. Convert the user request into a typed IR.

Allowed `op` values: {vocab}

Allowed libraries: {libs}

Rules:
- One `op` per semantic action. Do NOT split on commas inside lists.
- Use only ops from the allowed list.
- For iterative tasks or state loops (e.g. while loops, game loops, simulations), set `step_kind`: "loop" and place loop body steps into `body`.
- For conditional branches (if/else), set `step_kind`: "branch" with `condition`, placing branches into `then_branch` and `else_branch`.
- Extract EVERY literal into `literals`.
- Return ONLY a JSON object. No prose, no markdown.

User prompt:
{prompt}
"""

RETRY_TEMPLATE = """\
Your previous response was not valid JSON. Respond with ONLY a JSON object
matching this schema:

{{"steps": [{{"op": "str", "step_kind": "call|loop|branch", "params": {{}}, "library": "str", "body": []}}], "libraries": ["..."], "literals": {{}}}}

User prompt:
{prompt}
"""


def _extract_json_object(text: str) -> Optional[dict]:
    """Return the first balanced JSON object found in ``text``."""
    return extract_json_object(text)


@dataclass
class IR:
    steps: List[Dict[str, Any]]
    literals: Dict[str, Any] = field(default_factory=dict)
    libraries: List[str] = field(default_factory=list)
    task_objective: Dict[str, Any] = field(default_factory=dict)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    causal_dependencies: List[List[str]] = field(default_factory=list)
    raw: str = ""

    def all_steps(self) -> List[Dict[str, Any]]:
        """Recursively flattens all steps including nested body and branch steps."""
        res: List[Dict[str, Any]] = []
        def _collect(s_list: List[Dict[str, Any]]):
            for s in s_list:
                res.append(s)
                for k in ("body", "then_branch", "else_branch"):
                    sub = s.get(k)
                    if isinstance(sub, list):
                        _collect([x for x in sub if isinstance(x, dict)])
        _collect(self.steps)
        return res

    @classmethod
    def parse(cls, text: str) -> "IR":
        data = _extract_json_object(text)
        if data is None:
            raise ValueError("IR compiler: no valid JSON object in output")
        return cls(
            steps=data.get("steps", []),
            literals=dict(data.get("literals") or {}),
            libraries=list(data.get("libraries") or []),
            task_objective=dict(data.get("task_objective") or {}),
            variables=list(data.get("variables") or []),
            causal_dependencies=list(data.get("causal_dependencies") or []),
            raw=text,
        )

    def validate(self, vocab: Set[str]) -> List[str]:
        errs: List[str] = []
        vocab_norm = {str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in (vocab or ()) if v}
        vocab_subtokens = {p for v in vocab_norm for p in v.split("_") if len(p) >= 2}

        def _check_step(s: Dict[str, Any], path: str) -> None:
            op = str(s.get("op", "")).strip().lower().replace("-", "_").replace(" ", "_")
            if op:
                op_tokens = [t for t in op.split("_") if t]
                matched = sum(
                    1
                    for t in op_tokens
                    if t in vocab_norm or (len(t) >= 2 and t in vocab_subtokens)
                )
                prec = (matched / len(op_tokens)) if op_tokens else 0.0
                if op not in vocab_norm and prec < 0.6:
                    errs.append(f"{path}: unknown op {s.get('op')!r}")
            for sub_name in ("body", "then_branch", "else_branch"):
                sub_list = s.get(sub_name)
                if isinstance(sub_list, list):
                    for j, sub_s in enumerate(sub_list):
                        if isinstance(sub_s, dict):
                            _check_step(sub_s, f"{path}.{sub_name}[{j}]")

        for i, s in enumerate(self.steps):
            if isinstance(s, dict):
                _check_step(s, f"step {i}")
        return errs


class IRCompiler:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.vocab: Set[str] = self._derive_vocab()
        self._sorted_vocab: List[str] = sorted(self.vocab)
        self._vocab_tokens: Dict[str, Tuple[bool, Set[str]]] = {
            e: (
                len(e) >= 3,
                {p.lower() for p in e.replace("-", "_").split("_") if len(p) >= 3},
            )
            for e in self._sorted_vocab
        }
        self.libs: List[str] = self._derive_libs()

    def _derive_vocab(self) -> Set[str]:
        """Derives available operations dynamically from loaded cells or orchestrator configuration."""
        out: Set[str] = set()
        loaded = getattr(self.orchestrator, "loaded_cells", {}) or {}

        for c in loaded.values():
            # 1. Prefer explicit operation identifiers if exposed on the cell
            op_cand = (
                getattr(c, "op_name", None)
                or getattr(c, "operation", None)
                or getattr(c, "callable_name", None)
            )
            if op_cand:
                tail = str(op_cand).strip().lower()
            else:
                cid = str(getattr(c, "cell_id", "")).lower()
                dom = str(getattr(c, "domain_name", "") or "").lower()

                # Strip domain prefix across multiple possible delimiters (_, ., :, /, -)
                tail = cid
                if dom and tail.startswith(dom):
                    remainder = tail[len(dom):]
                    if remainder and remainder[0] in ("_", ".", ":", "/", "-"):
                        tail = remainder.lstrip("_.:/-")

                # If still delimited by hierarchical paths, take the terminal operation name
                for delim in (":", "/", "."):
                    if delim in tail:
                        tail = tail.split(delim)[-1]

            if 2 <= len(tail) <= 60 and " " not in tail and not tail.isdigit():
                out.add(tail)

            # Extract semantic annotations and keywords
            for src in (getattr(c, "keywords", []) or [],
                        getattr(c, "semantic_tags", []) or []):
                for kw in src:
                    k = str(kw).strip().lower()
                    if 2 <= len(k) < 40 and " " not in k and not k.isdigit():
                        out.add(k)

        if not out:
            # Fall back to orchestrator-level configured operation vocabulary if present
            orch_vocab = (
                getattr(self.orchestrator, "default_vocab", None)
                or getattr(self.orchestrator, "allowed_ops", None)
            )
            if orch_vocab:
                return set(orch_vocab)

        return out

    def _derive_libs(self) -> List[str]:
        loaded = getattr(self.orchestrator, "loaded_cells", {}) or {}
        return sorted({
            c.domain_name for c in loaded.values()
            if getattr(c, "domain_name", None)
        })

    def _vocab_for_prompt(self, prompt: str = "", max_entries: int = 300) -> str:
        """
        Formats vocabulary for the LLM prompt (Solution 7).

        Selection is PROMPT-RELEVANCE based, never stride sampling: the
        previous uniform stride sampling arbitrarily dropped up to ~70% of
        operations, so any operation falling between sample steps was
        unrepresentable in the compiled IR. Now:
          1. Entries whose declared tokens overlap (or prefix-match) the
             prompt's tokens always rank first — the operations the task
             needs are guaranteed to be offered to the LLM.
          2. Remaining budget is filled by stable, deterministic order so the
             prompt stays within the context window without semantic loss
             being concentrated on any lexical range.
        """
        if not self.vocab:
            return "(dynamically infer based on prompt and library context)"

        entries = sorted(self.vocab)
        if len(entries) <= max_entries:
            return ", ".join(entries)

        prompt_toks: Set[str] = set()
        try:
            from .tokenizer import CellTokenizer as _CT
        except (ImportError, ValueError):
            try:
                from tokenizer import CellTokenizer as _CT
            except Exception:
                _CT = None
        if _CT is not None and prompt:
            try:
                prompt_toks = {t.lower() for t in _CT.tokenize_prompt(prompt)}
            except Exception:
                prompt_toks = set()
        if not prompt_toks and prompt:
            prompt_toks = {
                w.strip("_'\",.;:()[]{}").lower()
                for w in prompt.split()
                if len(w) >= 3
            }

        vocab_tokens = getattr(self, "_vocab_tokens", {})

        def _relevance(entry: str) -> Tuple[int, int, str]:
            if not prompt_toks:
                return (0, 0, entry)
            cached = vocab_tokens.get(entry)
            if cached is not None:
                ge3, entry_toks = cached
            else:
                ge3 = len(entry) >= 3
                entry_toks = {p.lower() for p in entry.replace("-", "_").split("_") if len(p) >= 3}
            overlap = len(entry_toks & prompt_toks)
            prefix_hit = False
            if ge3:
                for pt in prompt_toks:
                    if pt.startswith(entry) or entry.startswith(pt):
                        prefix_hit = True
                        break
            return (overlap, 1 if prefix_hit else 0, entry)

        scored_items = sorted(((_relevance(e), e) for e in entries), key=lambda x: x[0], reverse=True)
        relevant = [e for sc, e in scored_items if sc[:2] != (0, 0)]
        filler = [e for sc, e in scored_items if sc[:2] == (0, 0)]
        selected = relevant[:max_entries]
        # Only pad with stratified background vocabulary when relevant hits are sparse,
        # avoiding hundreds of unrelated distractor op names in the LLM prompt.
        filler_target = max_entries if len(relevant) < 40 else min(max_entries, len(selected) + 40)
        if len(selected) < filler_target and filler:
            needed = filler_target - len(selected)
            if len(filler) <= needed:
                selected.extend(filler)
            else:
                sorted_filler = sorted(filler)
                step = len(sorted_filler) / float(needed)
                stratified_filler = [sorted_filler[int(i * step)] for i in range(needed)]
                selected.extend(stratified_filler)
        # Restore deterministic (alphabetical) order within the selected set
        # so the prompt is stable across runs for identical inputs.
        return ", ".join(sorted(selected))

    def compile(self, prompt: str) -> Optional[IR]:
        mm = ModelManager.get_instance()
        prof = getattr(mm, "profile", None)
        if prof is None or getattr(prof, "llm", None) is None:
            return None

        vocab_str = self._vocab_for_prompt(prompt)
        libs_str = ", ".join(sorted(self.libs))
        msg = PROMPT_TEMPLATE.format(vocab=vocab_str, libs=libs_str, prompt=prompt)

        try:
            raw = mm.generate_text(msg, max_tokens=768, schema=IR_SCHEMA)
            ir = IR.parse(raw)
            errs = ir.validate(self.vocab)
            if errs:
                logger.warning("[IR] vocab validation errors: %s", errs)
                raise ValueError(f"IR validation failed: {errs}")

            _ops = [s.get('op', '?') + '/' + str(s.get('library', '?')) for s in ir.steps]
            logger.info("[IR] compiled %d steps: %s", len(ir.steps), _ops)
            if ir.literals:
                logger.info("[IR] literals: %s", ir.literals)
            return ir
        except Exception as e1:
            logger.debug("[IR] first attempt failed: %s", e1)
            # Corrective retry
            try:
                raw2 = mm.generate_text(
                    RETRY_TEMPLATE.format(prompt=prompt),
                    max_tokens=768,
                    schema=IR_SCHEMA,
                )
                ir = IR.parse(raw2)
                errs2 = ir.validate(self.vocab)
                if errs2:
                    logger.warning("[IR] corrective retry validation errors: %s", errs2)
                    raise ValueError(f"IR validation failed after retry: {errs2}")
                logger.info("[IR] corrective retry succeeded")
                return ir
            except Exception as e2:
                logger.warning("[IR] compilation failed after retry: %s", e2)
                return None
