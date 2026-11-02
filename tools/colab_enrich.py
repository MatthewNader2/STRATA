"""
colab_enrich.py — Tier 3 LLM enrichment for STRATA trees/*.json
Optimized for Google Colab Free Tier (NVIDIA T4 16GB VRAM).

Key Features:
  - 100% GPU Offload: Utilizes T4 Tensor Cores + FlashAttention.
  - Fail-safe Checkpointing: Uses append-only JSONL deltas on Drive per batch,
    periodically synchronizing the full tree JSON. Disconnects never lose work.
  - Compact Prompt Serialization: Minifies prompt payloads to minimize latency.
"""

from __future__ import annotations
import atexit
import json
import os
import re
import signal
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================== CONFIG ======================================

REPO_URL = "https://github.com/MatthewNader2/STRATA_test.git"
IS_COLAB = Path("/content").exists()
LOCAL_REPO_DIR = Path(__file__).resolve().parent
LOCAL_CLONE_DIR = Path("/content/STRATA_test") if IS_COLAB else LOCAL_REPO_DIR
DRIVE_ROOT = Path("/content/drive/MyDrive/strata_enrichment") if IS_COLAB else LOCAL_REPO_DIR / "enrichment_checkpoints"
DRIVE_CHECKPOINTS = DRIVE_ROOT / "checkpoints"
DRIVE_LOGS = DRIVE_ROOT / "logs"

# Domains to enrich in sequential order
DOMAINS = ["pandas", "sklearn", "numpy", "cv2", "matplotlib", "scipy", "python_core"]

# Default high-efficiency model for T4 (Dense 9B, fits comfortably in ~6GB VRAM)
MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"

# Alternative ultra-fast code-specialized model if throughput is top priority:
# MODEL_REPO = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
# MODEL_FILE = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"

MODEL_LOCAL_DIR = Path("/content/models")
N_CTX = 4096
N_GPU_LAYERS = -1   # Offload all layers to GPU
N_BATCH = 1024      # Prompt processing batch size (speeds up prefill on T4)
N_UBATCH = 512      # Physical compute micro-batch size

# Inference & Checkpointing Batch Settings
BATCH_SIZE = 20                 # Items per LLM prompt
SAVE_FULL_JSON_EVERY_N = 10     # Flush consolidated tree JSON every N batches (200 cells)
MAX_RETRIES_PER_BATCH = 2

# ============================================================================


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def mount_drive() -> None:
    if IS_COLAB and not Path("/content/drive/MyDrive").exists():
        raise RuntimeError(
            "Google Drive is not mounted. Run this in a notebook cell first:\n\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
        )
    DRIVE_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    DRIVE_LOGS.mkdir(parents=True, exist_ok=True)


def sync_repo() -> None:
    if not IS_COLAB:
        log("Running locally — using existing tree files.")
        return

    if LOCAL_CLONE_DIR.exists():
        log("Checking existing repository clone...")
        ret = os.system(f"git -C {LOCAL_CLONE_DIR} pull --depth 1")
        if ret != 0 or not (LOCAL_CLONE_DIR / "trees").exists():
            log("Git pull failed; re-cloning fresh...")
            shutil.rmtree(LOCAL_CLONE_DIR, ignore_errors=True)
            os.system(f"git clone --depth 1 {REPO_URL} {LOCAL_CLONE_DIR}")
    else:
        log(f"Cloning {REPO_URL}...")
        os.system(f"git clone --depth 1 {REPO_URL} {LOCAL_CLONE_DIR}")

    if not (LOCAL_CLONE_DIR / "trees").exists():
        raise RuntimeError(f"Clone failed — no 'trees/' directory at {LOCAL_CLONE_DIR / 'trees'}")


def get_hf_token() -> Optional[str]:
    if IS_COLAB:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
            if token:
                return token
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")


def load_model():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    token = get_hf_token()
    if token:
        log("HF_TOKEN detected — using authenticated download.")
    else:
        log("No HF_TOKEN detected — proceeding with unauthenticated download.")

    MODEL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Resolving model weights: {MODEL_REPO}/{MODEL_FILE}...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_LOCAL_DIR),
        token=token,
    )

    log(f"Initializing Llama engine on GPU with FlashAttention enabled...")
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=N_BATCH,
        n_ubatch=N_UBATCH,
        flash_attn=True,
        n_threads=2,  # Matches Colab free tier 2-vCPU allocation
        verbose=False,
    )

    # Diagnostic GPU memory check
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_free, vram_total = torch.cuda.mem_get_info()
            log(f"Hardware Verified: {dev_name} | VRAM: {(vram_total - vram_free)/(1024**2):.0f}MB used / {vram_total/(1024**2):.0f}MB total")
    except Exception:
        pass

    return llm


# --------------------------- Resilient I/O ----------------------------------

def checkpoint_path(domain: str) -> Path:
    return DRIVE_CHECKPOINTS / f"{domain}.json"


def jsonl_progress_path(domain: str) -> Path:
    return DRIVE_CHECKPOINTS / f"{domain}_progress.jsonl"


def load_working_tree(domain: str) -> Dict[str, Any]:
    """Loads domain tree, applying any existing full checkpoint and replaying delta logs."""
    ckpt = checkpoint_path(domain)
    if ckpt.exists():
        log(f"[{domain}] Loading base checkpoint from Drive.")
        with open(ckpt, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        src = LOCAL_CLONE_DIR / "trees" / f"{domain}.json"
        if not src.exists():
            raise FileNotFoundError(f"No tree found for domain '{domain}' at {src}")
        log(f"[{domain}] Seeding base tree from repo clone.")
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Replay any uncommitted JSONL delta entries
    jpath = jsonl_progress_path(domain)
    if jpath.exists():
        by_id = {c["cell_id"]: c for c in data.get("cells", []) if "cell_id" in c}
        replayed = 0
        with open(jpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    cid = entry.get("cell_id")
                    if cid in by_id and not by_id[cid].get("docstring"):
                        by_id[cid]["docstring"] = entry.get("docstring")
                        by_id[cid]["enrichment_source"] = entry.get("enrichment_source", "llm")
                        by_id[cid]["enriched_at"] = entry.get("enriched_at")
                        replayed += 1
                except Exception:
                    continue
        if replayed > 0:
            log(f"[{domain}] Replayed {replayed} incremental updates from {jpath.name}.")

    return data


def atomic_save_json(domain: str, data: Dict[str, Any]) -> None:
    """Atomic write for full JSON snapshot."""
    ckpt = checkpoint_path(domain)
    tmp = ckpt.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ckpt)


def append_progress_delta(domain: str, entries: List[Dict[str, Any]]) -> None:
    """Fast, append-only disk write per batch (near-zero latency over Drive FUSE)."""
    jpath = jsonl_progress_path(domain)
    with open(jpath, "a", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# --------------------------- Prompting & Parsing -----------------------------

SYSTEM_PROMPT = (
    "You are documenting a library's functions for a code-synthesis tool.\n"
    "For each function given, write ONE short, plain, factual sentence describing what it does "
    "(a one-line docstring summary).\n"
    "Base your summary ONLY on the function name, its code template, and its input/output types.\n"
    "Do NOT invent parameter behavior or edge cases. If unsure, provide a clean generic description.\n"
    "Additionally, for each function declare a semantic ROLE for any input port whose value is "
    "a textual REFERENCE chosen by the caller — the kind of thing a user names in prose "
    "(a column/field name, a file path, a label, a key, an option name). Use exactly one of "
    "these role strings per port, or omit the port: 'column_reference' (a column or field "
    "name), 'file_path' (a filesystem path), 'label' (a display/title label), 'key_reference' "
    "(a dict/key lookup), 'option_reference' (an option/setting name). Omit ports that take "
    "data objects, numerics, booleans, or free-form content values.\n"
    "Respond strictly with a JSON object containing an 'items' array:\n"
    '{"items": [{"cell_id": "<id>", "docstring": "<one line summary>", '
    '"roles": {"<port_name>": "<role>"}}, ...]}'
)

#: roles the enrichment LLM may declare (data-side vocabulary; the engine is
#: vocabulary-free — it only stem-matches a DECLARED role against the prompt)
VALID_ROLES = {
    "column_reference", "file_path", "label", "key_reference", "option_reference",
}


def build_batch_prompt(cells: List[Dict[str, Any]]) -> str:
    """Builds a minified JSON array omitting empty/null keys to conserve prompt tokens."""
    items = []
    for c in cells:
        item = {
            "cell_id": c["cell_id"],
            "code_template": c.get("code_template", ""),
        }
        inputs = {k: v.get("type_name") for k, v in c.get("inputs", {}).items() if v.get("type_name")}
        if inputs:
            item["inputs"] = inputs
        outputs = {k: v.get("type_name") for k, v in c.get("outputs", {}).items() if v.get("type_name")}
        if outputs:
            item["outputs"] = outputs
        tags = c.get("semantic_tags", [])
        if tags:
            item["tags"] = tags[:6]
        if c.get("docstring"):
            item["partial_docs"] = c.get("docstring")
        items.append(item)
    return json.dumps(items, separators=(",", ":"))


def extract_json_items(raw: str) -> Optional[List[Dict[str, Any]]]:
    """Tolerant parser handling raw lists, wrapped objects, or fenced markdown."""
    if not raw:
        return None
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["items", "cells", "results", "data", "functions"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
    except Exception:
        pass

    # Regex search for fallback substring JSON
    start_arr, end_arr = text.find("["), text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        try:
            data = json.loads(text[start_arr:end_arr + 1])
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return None


def call_model(llm, cells: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    prompt = build_batch_prompt(cells)
    expected_ids = set(c["cell_id"] for c in cells)
    # 50 tokens per cell + 150 token margin is ideal for one-sentence outputs
    max_tokens = min(50 * len(cells) + 150, 2048)

    for attempt in range(1, MAX_RETRIES_PER_BATCH + 1):
        try:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = resp["choices"][0]["message"]["content"]
            parsed = extract_json_items(raw)

            if parsed is None:
                log(f"  Attempt {attempt}: output not parseable as JSON, retrying...")
                continue

            result_map = {}
            for item in parsed:
                if isinstance(item, dict) and "cell_id" in item and "docstring" in item:
                    cid = str(item["cell_id"]).strip()
                    doc = str(item["docstring"]).strip()
                    if cid in expected_ids and doc:
                        clean_roles: Dict[str, str] = {}
                        roles = item.get("roles")
                        if isinstance(roles, dict):
                            for p_name, role in roles.items():
                                if (
                                    isinstance(p_name, str) and isinstance(role, str)
                                    and role.strip().lower() in VALID_ROLES
                                ):
                                    clean_roles[p_name.strip()] = role.strip().lower()
                        result_map[cid] = {"docstring": doc, "roles": clean_roles}

            if len(result_map) == 0:
                log(f"  Attempt {attempt}: no matching cell IDs returned, retrying...")
                continue

            return result_map
        except Exception as e:
            log(f"  Attempt {attempt}: inference exception: {e}")
            time.sleep(1)

    return None


# --------------------------------- Runner ------------------------------------

_CURRENT_DOMAIN: Optional[str] = None
_CURRENT_DATA: Optional[Dict[str, Any]] = None


def emergency_flush():
    """Flushes active state if session is abruptly interrupted or terminated."""
    if _CURRENT_DOMAIN and _CURRENT_DATA:
        log(f"[{_CURRENT_DOMAIN}] Emergency snapshot save triggered...")
        atomic_save_json(_CURRENT_DOMAIN, _CURRENT_DATA)


atexit.register(emergency_flush)


def enrich_domain(llm, domain: str) -> None:
    global _CURRENT_DOMAIN, _CURRENT_DATA
    _CURRENT_DOMAIN = domain
    data = load_working_tree(domain)
    _CURRENT_DATA = data

    cells = data.get("cells", [])
    pending = [c for c in cells if not c.get("docstring") and c.get("enrichment_source") != "llm"]
    already_done = len(cells) - len(pending)
    total_cells = len(cells)

    log(f"[{domain}] Total: {total_cells} | Already Enriched: {already_done} | Remaining: {len(pending)}")
    if not pending:
        log(f"[{domain}] Domain is completely enriched. Nothing to do.")
        return

    by_id = {c["cell_id"]: c for c in cells}
    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    domain_start_time = time.time()
    enriched_this_run = 0

    for idx in range(0, len(pending), BATCH_SIZE):
        batch_start = time.time()
        batch = pending[idx:idx + BATCH_SIZE]
        batch_num = (idx // BATCH_SIZE) + 1

        result = call_model(llm, batch)
        if not result:
            log(f"[{domain}] Batch {batch_num}/{n_batches} failed after retries. Logging IDs and continuing...")
            with open(DRIVE_LOGS / "failed_batches.log", "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} {domain} {[c['cell_id'] for c in batch]}\n")
            continue

        now = datetime.now(timezone.utc).isoformat()
        delta_entries = []
        for cid, payload in result.items():
            doc = payload["docstring"] if isinstance(payload, dict) else payload
            roles = payload.get("roles", {}) if isinstance(payload, dict) else {}
            cell = by_id.get(cid)
            if cell is not None:
                cell["docstring"] = doc
                cell["enrichment_source"] = "llm"
                cell["enriched_at"] = now
                # Declared semantic roles: DATA, not engine logic. Only applied
                # to wildcard-state input ports — harvester-declared constraints
                # (mutator receivers, typestates) are never regressed.
                declared_roles = {}
                if roles:
                    for p_name, role in roles.items():
                        port = (cell.get("inputs") or {}).get(p_name)
                        if isinstance(port, dict) and str(port.get("state", "")).lower() in ("any", "", "default"):
                            port["state"] = role
                            declared_roles[p_name] = role
                delta_entries.append({
                    "cell_id": cid,
                    "docstring": doc,
                    "roles": declared_roles,
                    "enrichment_source": "llm",
                    "enriched_at": now
                })
                enriched_this_run += 1

        # Fast append-only write to avoid Colab Drive I/O hangs
        append_progress_delta(domain, delta_entries)

        # Periodic full JSON flush & ETA telemetry
        elapsed = time.time() - domain_start_time
        cells_per_sec = enriched_this_run / max(elapsed, 0.001)
        remaining_cells = len(pending) - (idx + len(batch))
        eta_sec = int(remaining_cells / max(cells_per_sec, 0.001)) if cells_per_sec > 0 else 0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

        if batch_num % SAVE_FULL_JSON_EVERY_N == 0 or idx + BATCH_SIZE >= len(pending):
            atomic_save_json(domain, data)
            log(f"[{domain}] Batch {batch_num}/{n_batches} | Speed: {cells_per_sec:.1f} cells/s | ETA: {eta_str} [Checkpoint Synced]")
        else:
            log(f"[{domain}] Batch {batch_num}/{n_batches} | Speed: {cells_per_sec:.1f} cells/s | ETA: {eta_str}")

    # Final consolidated flush
    atomic_save_json(domain, data)
    _CURRENT_DOMAIN = None
    _CURRENT_DATA = None
    log(f"[{domain}] Completed all pending batches. Checkpoint finalized at {checkpoint_path(domain)}")


def main() -> None:
    mount_drive()
    sync_repo()
    llm = load_model()

    for domain in DOMAINS:
        try:
            enrich_domain(llm, domain)
        except KeyboardInterrupt:
            log(f"Interrupted by user during {domain}. State has been saved to Drive.")
            emergency_flush()
            sys.exit(0)
        except Exception as e:
            log(f"[{domain}] ERROR: {e} — state preserved. Moving to next domain.")

    log(f"\nAll domains processed! Checkpoint files are saved at: {DRIVE_CHECKPOINTS}")


if __name__ == "__main__":
    main()
