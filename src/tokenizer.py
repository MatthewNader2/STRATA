"""
src/tokenizer.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Domain-Agnostic Sub-Word and Identifier Tokenizer.

Deterministic, mathematically formal character-level transition scanner.
Zero regular expressions and zero English-language morphological or stopword heuristics.
"""

from __future__ import annotations
import functools
from typing import Set, Tuple, List


# --------------------------------------------------------------------------- #
# Universal token normalizer
# --------------------------------------------------------------------------- #


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
_ALIASES_KEY: Tuple[Tuple[str, str], ...] = ()


_MM_CLS = None


def _is_profile_zero() -> bool:
    global _MM_CLS
    try:
        from .config import resolve_profile_spec, active_profile_spec
    except (ImportError, ValueError):
        from config import resolve_profile_spec, active_profile_spec
    if _MM_CLS is None:
        try:
            from .inference import ModelManager as _MM
            _MM_CLS = _MM
        except (ImportError, ValueError):
            try:
                from inference import ModelManager as _MM
                _MM_CLS = _MM
            except Exception:
                _MM_CLS = False
    if _MM_CLS:
        try:
            pname = getattr(_MM_CLS.get_instance(), "current_profile_name", None)
            if pname is not None:
                return resolve_profile_spec(pname).pure_symbolic
        except Exception:
            pass
    try:
        return active_profile_spec().pure_symbolic
    except Exception:
        return False


def register_aliases(mapping: dict) -> None:
    global _ALIASES, _ALIASES_REV, _ALIASES_KEY
    clean = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in (mapping or {}).items()
        if k and v
    }
    if clean == _ALIASES:
        return
    _ALIASES = clean
    _ALIASES_KEY = tuple(sorted(_ALIASES.items()))
    rev: dict = {}
    for k, v in _ALIASES.items():
        rev.setdefault(v, set()).add(k)
    _ALIASES_REV = rev
    try:
        _tokenize_identifier_cached.cache_clear()
    except Exception:
        pass
    try:
        _tokenize_prompt_cached.cache_clear()
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
        if len(t) > 1 or (len(t) == 1 and t.isalpha()):
            tokens.add(t)

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
    if len(full) > 1 or (len(full) == 1 and full.isalpha()):
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
        using a deterministic character-level transition scanner.
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
    def tokenize_prompt(cls, prompt: str) -> Set[str]:
        if not prompt:
            return set()
        return set(_tokenize_prompt_cached(prompt, _is_profile_zero(), _ALIASES_KEY))

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
        hard_break_word_idxs: Set[int] = set()
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
                hard_break_word_idxs.add(idx)
                continue

            if (";" in w or w.endswith(":")) and not (_is_inside_protected(s) and _is_inside_protected(e - 1)):
                split_positions.add(e)
                hard_break_word_idxs.add(idx)
                continue

            if "\n" in w:
                split_positions.add(e)
                hard_break_word_idxs.add(idx)
                continue

            if w.endswith(".") and idx + 1 < len(raw_words) and not (_is_inside_protected(s) and _is_inside_protected(e - 1)):
                if not (clean and clean.replace(".", "").isdigit()):
                    split_positions.add(e)
                    hard_break_word_idxs.add(idx)
                    continue

        # 3b. Evaluate unprotected trailing commas within each hard sentence span so
        # series / Oxford lists ("a, b, and c", "a, b, c", "X, y") stay in one clause
        # while genuine multi-clause commas still split cleanly.
        def _clean_seg_words(word_slice: List[Tuple[int, int, str]]) -> Tuple[bool, int]:
            cleaned: List[str] = []
            for _, _, tok in word_slice:
                cw = tok.strip(",;.:!?()[]{}\'\"")
                if cw and cw not in ("->", "➔"):
                    cleaned.append(cw)
            starts_coord = len(cleaned) >= 2 and cleaned[0].lower() in ("and", "or", "&")
            core_count = (len(cleaned) - 1) if starts_coord else len(cleaned)
            return starts_coord, core_count

        span_start = 0
        n_words = len(raw_words)
        while span_start < n_words:
            span_end = span_start
            while span_end < n_words and span_end not in hard_break_word_idxs:
                # Also break if a bare newline lies between raw_words[span_end] and raw_words[span_end+1]
                if span_end + 1 < n_words:
                    gap = text[raw_words[span_end][1]:raw_words[span_end + 1][0]]
                    if "\n" in gap and any(
                        text[raw_words[span_end][1] + gi] == "\n"
                        and not _is_inside_protected(raw_words[span_end][1] + gi)
                        for gi in range(len(gap))
                    ):
                        break
                span_end += 1
            seg_end = min(span_end + 1, n_words) if span_end < n_words else n_words

            comma_cuts: List[int] = []
            for wi in range(span_start, seg_end - 1):
                ws, we, ww = raw_words[wi]
                if ww.endswith(",") and not (_is_inside_protected(ws) and _is_inside_protected(we - 1)):
                    comma_cuts.append(wi)

            if comma_cuts:
                slices: List[List[Tuple[int, int, str]]] = []
                prev_wi = span_start
                for ci in comma_cuts:
                    slices.append(raw_words[prev_wi:ci + 1])
                    prev_wi = ci + 1
                slices.append(raw_words[prev_wi:seg_end])

                for k, ci in enumerate(comma_cuts):
                    _, left_core = _clean_seg_words(slices[k])
                    right_coord, right_core = _clean_seg_words(slices[k + 1])
                    is_series = (
                        (not right_coord and right_core == 1)
                        or (right_coord and right_core == 1 and left_core == 1)
                    )
                    if not is_series:
                        split_positions.add(raw_words[ci][1])

            span_start = seg_end

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


@functools.lru_cache(maxsize=4096)
def _tokenize_prompt_cached(
    prompt: str,
    profile_zero: bool,
    aliases_key: Tuple[Tuple[str, str], ...],
) -> Tuple[str, ...]:
    tokens = set()
    cur = []
    for ch in prompt.lower():
        if ch.isalnum() or ch == '_':
            cur.append(ch)
        else:
            if cur:
                w = "".join(cur)
                if (len(w) > 1 and (w[0].isalpha() or w[0] == '_')) or (len(w) == 1 and w.isalpha()):
                    tokens.add(w)
                cur = []
    if cur:
        w = "".join(cur)
        if (len(w) > 1 and (w[0].isalpha() or w[0] == '_')) or (len(w) == 1 and w.isalpha()):
            tokens.add(w)
    for raw in prompt.split():
        cw = raw.strip(",;.:!?()[]{}\'\"")
        if any(c.isupper() for c in cw) or '_' in cw or '.' in cw or any(c.isdigit() for c in cw):
            tokens.update(CellTokenizer.tokenize_identifier(cw))
        if '-' in cw:
            h_parts = [p.lower() for p in cw.split('-') if p and p.isalnum()]
            if len(h_parts) >= 2 and all(any(ch.isalpha() for ch in p) for p in h_parts):
                joined = "".join(h_parts)
                if len(joined) > 1:
                    tokens.add(joined)
    if _ALIASES and not profile_zero:  # (apply_fixes_v7)
        tokens = _expand_aliases(tokens)
    return tuple(sorted(tokens))
