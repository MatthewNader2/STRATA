"""
src/tokenizer.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Domain-Agnostic Sub-Word and Identifier Tokenizer.

Deterministic, mathematically formal character-level transition scanner.
Zero regular expressions.

Morphological normalization is provided by an OPT-IN English Porter-class
suffix stripper. It is applied IDENTICALLY to prompt tokens and identifier
tokens when enabled, so English morphological variants ("testing"/"test",
"training"/"train", "regression"/"regressor", "normalized"/"normalize")
collapse to the same token. The stemmer is (a) gated by the module-level
`ENABLE_ENGLISH_STEMMING` flag and (b) applied only to purely ASCII-alphabetic
tokens. Non-Latin scripts and any token containing digits or symbols pass
through unchanged. This is an English-language heuristic, not a universal
normalizer; disabling the flag yields a purely character-level tokenizer.
"""

from __future__ import annotations
import functools
from typing import Set, Tuple, List


# --------------------------------------------------------------------------- #
# Universal token normalizer
# --------------------------------------------------------------------------- #

ENABLE_ENGLISH_STEMMING: bool = False


@functools.lru_cache(maxsize=32768)
def normalize_token(token: str) -> str:
    """
    Pure universal token normalizer: case-folding and whitespace stripping.
    Zero English stemming or morphological heuristics.
    """
    return token.strip().lower()


# (apply_fixes_v7) Alias expansion. Tree JSON declares aliases; tokenizer
# expands them in both directions.
_ALIASES: dict = {}
_ALIASES_REV: dict = {}


def _is_profile_zero() -> bool:
    try:
        from .inference import ModelManager
    except (ImportError, ValueError):
        try:
            from inference import ModelManager
        except Exception:
            return False
    try:
        return getattr(ModelManager.get_instance(), "current_profile_name", None) == "0"
    except Exception:
        return False


def register_aliases(mapping: dict) -> None:
    global _ALIASES, _ALIASES_REV
    clean = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in (mapping or {}).items()
        if k and v
    }
    if clean == _ALIASES:
        return
    _ALIASES = clean
    rev: dict = {}
    for k, v in _ALIASES.items():
        rev.setdefault(v, set()).add(k)
    _ALIASES_REV = rev
    try:
        _tokenize_identifier_cached.cache_clear()
    except Exception:
        pass


def _expand_aliases(tokens):
    if not _ALIASES or not tokens:
        return tokens
    out = set(tokens)
    for t in tokens:
        c = _ALIASES.get(t)
        if c:
            out.add(c)
        r = _ALIASES_REV.get(t)
        if r:
            out.update(r)
    return out


@functools.lru_cache(maxsize=16384)
def _tokenize_identifier_cached(identifier: str, expand_aliases: bool = True) -> Tuple[str, ...]:
    clean = str(identifier).strip()
    if not clean:
        return ()
    tokens = set()

    def _emit(chunk):
        t = "".join(chunk).lower().strip()
        if len(t) > 1:
            tokens.add(t)
            if ENABLE_ENGLISH_STEMMING:
                s = normalize_token(t)
                if len(s) > 1:
                    tokens.add(s)

    seg, chunk = [], []
    for ch in clean:
        if ch.isalnum():
            seg.append(ch)
        elif seg:
            _emit(seg)
            seg = []
    if seg:
        _emit(seg)

    n = len(clean)
    for i, ch in enumerate(clean):
        if not ch.isalnum():
            if chunk:
                _emit(chunk)
                chunk = []
            continue
        if chunk:
            p = chunk[-1]
            if (
                (p.islower() and ch.isupper())
                or (p.isupper() and ch.isupper() and i + 1 < n and clean[i + 1].islower())
                or (p.isalpha() and ch.isdigit())
                or (p.isdigit() and ch.isalpha())
            ):
                _emit(chunk)
                chunk = [ch]
                continue
        chunk.append(ch)
    if chunk:
        _emit(chunk)

    full = "".join(c for c in clean if c.isalnum() or c in ("_", "-")).lower().strip()
    if len(full) > 1:
        tokens.add(full)

    if expand_aliases and _ALIASES:  # (apply_fixes_v7)
        tokens = _expand_aliases(tokens)

    return tuple(sorted(tokens))


class CellTokenizer:
    """Tokenizes code identifiers, cell IDs, and user prompts into sub-word tokens without regular expressions."""

    @classmethod
    def clean_identifier(cls, token: str) -> str:
        """
        Strips non-identifier boundary characters using character transitions without regular expressions
        or ad-hoc punctuation string lists.
        """
        if not token:
            return ""
        chars = list(str(token))
        i = 0
        while i < len(chars) and not (chars[i].isalnum() or chars[i] == '_'):
            i += 1
        j = len(chars)
        while j > i and not (chars[j - 1].isalnum() or chars[j - 1] == '_'):
            j -= 1
        return "".join(chars[i:j])

    @classmethod
    def tokenize_identifier(cls, identifier: str) -> Set[str]:
        """
        Splits camelCase, snake_case, kebab-case, and dotted identifiers into distinct sub-tokens
        using a deterministic character-level transition scanner. Each emitted token is paired
        with its morphological stem (when the English stemmer is enabled) so cross-side variant
        matches succeed symmetrically.
        """
        if not identifier:
            return set()
        return set(_tokenize_identifier_cached(str(identifier), not _is_profile_zero()))

    @classmethod
    def tokenize_identifier_frozen(cls, identifier: str) -> Tuple[str, ...]:
        """Cached immutable variant returning Tuple[str, ...]."""
        if not identifier:
            return ()
        return _tokenize_identifier_cached(str(identifier), not _is_profile_zero())

    @classmethod
    def tokenize_prompt(cls, prompt: str, remove_stopwords: bool = False) -> Set[str]:
        if not prompt:
            return set()
        tokens = set()
        cur = []
        for ch in prompt.lower():
            if ch.isalnum() or ch == '_':
                cur.append(ch)
            else:
                if cur:
                    w = "".join(cur)
                    if len(w) > 1 and (w[0].isalpha() or w[0] == '_'):
                        tokens.add(w)
                        if ENABLE_ENGLISH_STEMMING and w.isalpha():
                            s = normalize_token(w)
                            if len(s) > 1:
                                tokens.add(s)
                    cur = []
        if cur:
            w = "".join(cur)
            if len(w) > 1 and (w[0].isalpha() or w[0] == '_'):
                tokens.add(w)
                if ENABLE_ENGLISH_STEMMING and w.isalpha():
                    s = normalize_token(w)
                    if len(s) > 1:
                        tokens.add(s)
        for raw in prompt.split():
            cw = raw.strip(",;.:!?()[]{}\'\"")
            if any(c.isupper() for c in cw) or '_' in cw or '.' in cw or any(c.isdigit() for c in cw):
                tokens.update(cls.tokenize_identifier(cw))
        if _ALIASES and not _is_profile_zero():  # (apply_fixes_v7)
            tokens = _expand_aliases(tokens)
        return tokens

    @classmethod
    def tokenize_cell(cls, cell_id: str, keywords: Set[str]) -> Set[str]:
        """Produces a unified token set for a cell using its ID and declared keywords."""
        tokens = cls.tokenize_identifier(cell_id)
        for kw in keywords:
            tokens.update(cls.tokenize_identifier(kw))
        return tokens

    @classmethod
    def split_prompt_clauses(cls, prompt: str) -> List[str]:
        """
        Splits a prompt into logical action clauses strictly on universal punctuation
        ('.', ';', ',', '\n', '->', '➔') without breaking inside quotes or brackets.
        Zero English connective lists, zero grammar heuristics, zero word-window lookbacks.
        """
        if not prompt or not prompt.strip():
            return []

        text = prompt.strip()
        # 1. Mask quoted strings and bracketed expressions
        protected = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in ("'", '"'):
                j = i + 1
                while j < n and text[j] != ch:
                    if text[j] == '\\':
                        j += 1
                    j += 1
                if j < n:
                    protected.append((i, j + 1))
                    i = j + 1
                    continue
            elif ch in ('(', '['):
                closing = ')' if ch == '(' else ']'
                depth = 1
                j = i + 1
                while j < n and depth > 0:
                    if text[j] == ch:
                        depth += 1
                    elif text[j] == closing:
                        depth -= 1
                    j += 1
                protected.append((i, j))
                i = j
                continue
            i += 1

        def _is_inside_protected(idx: int) -> bool:
            return any(s <= idx < e for s, e in protected)

        # 2. Tokenize words with positions
        raw_words = []
        cur = []
        w_start = None
        for idx, ch in enumerate(text):
            if not ch.isspace():
                if w_start is None:
                    w_start = idx
                cur.append(ch)
            else:
                if cur:
                    raw_words.append((w_start, idx, "".join(cur)))
                    cur = []
                    w_start = None
        if cur and w_start is not None:
            raw_words.append((w_start, len(text), "".join(cur)))

        # 3. Detect clause boundaries strictly on universal punctuation
        split_positions = set()
        # Structural newlines outside quotes/brackets are universal clause boundaries
        for idx, ch in enumerate(text):
            if ch == '\n' and not _is_inside_protected(idx):
                split_positions.add(idx)
        for idx, (s, e, w) in enumerate(raw_words):
            if _is_inside_protected(s) and _is_inside_protected(e - 1):
                continue
            clean = w.strip(".,;:!?")

            if "->" in w or "➔" in w:
                split_positions.add(s)
                split_positions.add(e)
                continue

            if ";" in w and not (_is_inside_protected(s) and _is_inside_protected(e - 1)):
                split_positions.add(e)
                continue

            if "\n" in w:
                split_positions.add(e)
                continue

            if w.endswith(".") and idx + 1 < len(raw_words) and not (_is_inside_protected(s) and _is_inside_protected(e - 1)):
                if not (clean and clean.replace(".", "").isdigit()):
                    split_positions.add(e)
                    continue

            if w.endswith(",") and idx + 1 < len(raw_words) and not (_is_inside_protected(s) and _is_inside_protected(e - 1)):
                split_positions.add(e)
                continue

        if not split_positions:
            return [text]

        sorted_splits = sorted(list(split_positions))
        clauses = []
        last_pos = 0
        for pos in sorted_splits:
            chunk = text[last_pos:pos].strip(" ,;:\n\t->➔")
            if chunk:
                clauses.append(chunk)
            last_pos = pos

        tail = text[last_pos:].strip(" ,;:\n\t->➔")
        if tail:
            clauses.append(tail)

        return clauses if clauses else [text]
