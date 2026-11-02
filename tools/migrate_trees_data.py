#!/usr/bin/env python3
"""
tools/migrate_trees_data.py

Systematic migration script for STRATA trees (PR 2: trees - the data side):
  1. Unifies all `shape_contract` into the canonical structured schema:
     {"ndim": int | None, "shape": list | None, "rank_polymorphic": bool, "channel_count": int | None, "channel_axis": int | None, "len": int | None}
  2. Fixes Pillow RGB/RGBA/CMYK/etc. rank convention to rank 3, grayscale to rank 2, and polymorphic ops to rank_polymorphic.
  3. Corrects structure_kind and rank anomalies (e.g. List[Contour] to collection/r1, Axes in seaborn to handle, List[T] in control_flow to collection/r1).
  4. Gives every required scalar port a port_role and a default value or explicit sentinel.
  5. Registers all undeclared port typestates into domain typestates blocks formatted as TypestateVocabularySchema.
"""

import json
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Parameter defaults mapping
PARAM_DEFAULTS = {
    # audio
    "sample_rate": 44100,
    "orig_sr": 44100,
    "target_sr": 22050,
    "gain_db": 0.0,
    # cryptography
    "current_time": 0,
    "ttl": 3600,
    "digestmod": "sha256",
    "n_bytes": 32,
    "length": 32,
    "iterations": 100000,
    "info": "",
    "n": 16384,
    "r": 8,
    "p": 1,
    "maxmem": 0,
    "dklen": 64,
    "key_size": 2048,
    # pathlib
    "name": "file.txt",
    "suffix": ".txt",
    "stem": "file",
    "pattern": "*",
    "mode": 420,  # 0o644
    "child_name": "",
    "sub_path": "",
    # pygame
    "title": "Pygame Window",
    "milliseconds": 10,
    "target_type": 0,
    "dest_pos": [0, 0],
    "x": 0,
    "y": 0,
    "dx": 1,
    "dy": 1,
    "point": [0, 0],
    "center_pos": [0, 0],
    "center": [0, 0],
    "radius": 10,
    "start_pos": [0, 0],
    "end_pos": [100, 100],
    "points": [[0, 0], [100, 100]],
    "flip_x": False,
    "flip_y": False,
    "factor": 1.0,
    # numpy
    "rows": 10,
    "cols": 10,
    # sqlite3
    "pragma_name": "foreign_keys",
    "pragma_value": "ON",
    "function_name": "custom_fn",
    "num_params": 1,
    "aggregate_name": "custom_agg",
    "collation_name": "nocase",
    "n_instructions": 100,
    "enabled": True,
    "extension_path": "ext.so",
    "target_db_path": "backup.db",
    "isolation_level": "DEFERRED",
    "timeout_seconds": 5.0,
    "table_name": "data",
    "schema_ddl": "id INTEGER PRIMARY KEY, value TEXT",
    "index_name": "idx_data",
    "columns": "id",
    "column_def": "col TEXT",
    "sql": "SELECT 1",
    "sql_script": "BEGIN; COMMIT;",
    "set_clause": "value = ?",
    "where_clause": "id = ?",
    "size": 10,
    # pillow
    "degrees": 90.0,
    "crop_box": [0, 0, 100, 100],
    "channel_identifier": "R",
    # generic
    "scale": 1.0,
    "angle": 0.0,
    "winname": "preview",
}

CELL_PARAM_DEFAULTS = {
    ("pandas.json", "PD_SELECT_COLUMNS", "port_1"): ([], "column_projection"),
    ("pandas.json", "PD_SELECT_COLUMN", "port_1"): ("", "column_projection"),
    ("pandas.json", "PD_SET_COLUMN", "port_1"): ("", "column_projection"),
    ("pandas.json", "PD_FILLNA_CONST", "port_1"): (0.0, "configuration"),
    ("pandas.json", "PD_READ_SQL", "port_0"): ("SELECT * FROM data", "configuration"),
    ("pygame.json", "PYGAME_SURFACE_CREATE", "size"): ([800, 600], "configuration"),
    ("pygame.json", "PYGAME_SURFACE_FILL", "color"): ([0, 0, 0], "configuration"),
    ("pygame.json", "PYGAME_RECT_INIT", "width"): (100, "configuration"),
    ("pygame.json", "PYGAME_RECT_INIT", "height"): (100, "configuration"),
    ("pygame.json", "PYGAME_IMAGE_LOAD", "image_path"): ("image.png", "configuration"),
    ("pygame.json", "PYGAME_IMAGE_SAVE", "file_path"): ("output.png", "configuration"),
    ("pygame.json", "PYGAME_SOUND_LOAD", "file_path"): ("sound.wav", "configuration"),
    ("pygame.json", "PYGAME_TRANSFORM_SCALE", "size"): ([100, 100], "configuration"),
    ("pygame.json", "PYGAME_TRANSFORM_SMOOTHSCALE", "size"): ([100, 100], "configuration"),
    ("numpy.json", "NUMPY_ZEROS_1D", "length"): (10, "configuration"),
    ("numpy.json", "NUMPY_ONES_1D", "length"): (10, "configuration"),
    ("numpy.json", "NUMPY_ARANGE", "stop"): (10, "configuration"),
    ("numpy.json", "NUMPY_LINSPACE", "start"): (0.0, "configuration"),
    ("numpy.json", "NUMPY_LINSPACE", "stop"): (1.0, "configuration"),
    ("matplotlib.json", "matplotlib_ax_annotate", "text"): ("", "configuration"),
    ("matplotlib.json", "matplotlib_ax_text", "x"): (0.0, "configuration"),
    ("matplotlib.json", "matplotlib_ax_text", "y"): (0.0, "configuration"),
    ("matplotlib.json", "matplotlib_ax_text", "s"): ("", "configuration"),
    ("matplotlib.json", "matplotlib_ax_set_title", "label"): ("Title", "configuration"),
    ("matplotlib.json", "matplotlib_ax_set_xlabel", "label"): ("X", "configuration"),
    ("matplotlib.json", "matplotlib_ax_set_ylabel", "label"): ("Y", "configuration"),
    ("matplotlib.json", "matplotlib_fig_suptitle", "t"): ("Title", "configuration"),
    ("pillow.json", "PILLOW_ROTATE", "angle"): (90.0, "configuration"),
    ("pillow.json", "PILLOW_ENHANCE_CONTRAST", "factor"): (1.5, "configuration"),
    ("pillow.json", "PILLOW_ENHANCE_BRIGHTNESS", "factor"): (1.2, "configuration"),
    ("pillow.json", "PILLOW_ENHANCE_COLOR", "factor"): (1.3, "configuration"),
    ("pillow.json", "PILLOW_ENHANCE_SHARPNESS", "factor"): (2.0, "configuration"),
    ("pillow.json", "PILLOW_IMAGE_NEW_RGB", "size"): ([256, 256], "configuration"),
    ("pillow.json", "PILLOW_IMAGE_NEW_RGBA", "size"): ([256, 256], "configuration"),
    ("pillow.json", "PILLOW_IMAGE_NEW_GRAYSCALE", "size"): ([256, 256], "configuration"),
    ("pillow.json", "PILLOW_RESIZE_LANCZOS", "size"): ([224, 224], "configuration"),
    ("pillow.json", "PILLOW_RESIZE_BILINEAR", "size"): ([128, 128], "configuration"),
    ("pillow.json", "PILLOW_RESIZE_NEAREST", "size"): ([64, 64], "configuration"),
    ("pillow.json", "PILLOW_CROP", "box"): ([0, 0, 100, 100], "configuration"),
    ("pillow.json", "PILLOW_IMAGE_GETCHANNEL", "channel"): ("A", "configuration"),
    ("requests.json", "REQ_SESSION_MOUNT_ADAPTER", "prefix"): ("https://", "configuration"),
    ("requests.json", "REQ_SESSION_SET_PROXIES", "http_proxy"): ("", "configuration"),
    ("requests.json", "REQ_SESSION_SET_PROXIES", "https_proxy"): ("", "configuration"),
    ("requests.json", "REQ_SESSION_SET_VERIFY_SSL", "verify"): (True, "configuration"),
    ("requests.json", "REQ_SET_TIMEOUT_TUPLE", "connect_timeout"): (5.0, "configuration"),
    ("requests.json", "REQ_SET_TIMEOUT_TUPLE", "read_timeout"): (30.0, "configuration"),
    ("requests.json", "REQ_SET_USER_AGENT", "user_agent"): ("python-requests", "configuration"),
    ("requests.json", "REQ_URL_JOIN", "base_url"): ("https://api.example.com/", "configuration"),
    ("requests.json", "REQ_SET_BEARER_AUTH", "token"): ("", "configuration"),
    ("requests.json", "REQ_SET_BASIC_AUTH", "username"): ("", "configuration"),
    ("requests.json", "REQ_SET_BASIC_AUTH", "password"): ("", "configuration"),
    ("requests.json", "REQ_SET_API_KEY_HEADER", "api_key"): ("", "configuration"),
    ("requests.json", "REQ_SET_DIGEST_AUTH", "username"): ("", "configuration"),
    ("requests.json", "REQ_SET_DIGEST_AUTH", "password"): ("", "configuration"),
    ("requests.json", "REQ_SET_PROXY_AUTH", "username"): ("", "configuration"),
    ("requests.json", "REQ_SET_PROXY_AUTH", "password"): ("", "configuration"),
    ("requests.json", "REQ_BUILD_CUSTOM_HEADER", "value"): ("", "configuration"),
}

DATA_INPUT_CELL_PORTS = {
    ("builtins.json", "PY_ABS", "data"),
    ("builtins.json", "PY_ROUND", "data"),
    ("builtins.json", "PY_RANGE", "stop"),
    ("builtins.json", "PY_INT", "data"),
    ("builtins.json", "PY_FLOAT", "data"),
    ("cryptography.json", "CRYPTO_FERNET_INIT", "key"),
    ("cryptography.json", "CRYPTO_FERNET_ENCRYPT", "data"),
    ("cryptography.json", "CRYPTO_FERNET_DECRYPT", "ciphertext"),
    ("cryptography.json", "CRYPTO_B64_DECODE", "b64_data"),
    ("cryptography.json", "CRYPTO_URLSAFE_B64_DECODE", "b64_data"),
    ("cryptography.json", "CRYPTO_HEX_TO_BYTES", "hex_str"),
    ("cryptography.json", "CRYPTO_PBKDF2_DERIVE", "password"),
    ("cryptography.json", "CRYPTO_PBKDF2_VERIFY", "password"),
    ("cryptography.json", "CRYPTO_PBKDF2_VERIFY", "expected_key"),
    ("cryptography.json", "CRYPTO_HKDF_DERIVE", "key_material"),
    ("cryptography.json", "CRYPTO_HKDF_VERIFY", "key_material"),
    ("cryptography.json", "CRYPTO_HKDF_VERIFY", "expected_key"),
    ("cryptography.json", "CRYPTO_COMPARE_DIGEST", "a"),
    ("cryptography.json", "CRYPTO_COMPARE_DIGEST", "b"),
    ("cryptography.json", "CRYPTO_SCRYPT_DERIVE", "password"),
    ("cryptography.json", "CRYPTO_SCRYPT_DERIVE", "salt"),
    ("requests.json", "REQ_IS_STATUS_200", "status_code"),
    ("requests.json", "REQ_IS_CLIENT_ERROR", "status_code"),
    ("requests.json", "REQ_IS_SERVER_ERROR", "status_code"),
    ("audio.json", "AUDIO_GET_SAMPLE_RATE", "sample_rate_in"),
    ("pandas.json", "PD_READ_SQL", "port_1"),
    ("pathlib.json", "PATH_WRITE_TEXT", "content"),
    ("pathlib.json", "PATH_WRITE_BYTES", "content"),
    ("pathlib.json", "PATH_APPEND_TEXT", "content"),
    ("pathlib.json", "SHUTIL_WHICH", "cmd"),
    ("pygame.json", "PYGAME_FONT_RENDER", "text"),
    ("pygame.json", "PYGAME_GAME_LOOP", "running"),
}


def parse_shape_contract(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        ndim = raw.get("ndim")
        shape = raw.get("shape")
        poly = bool(raw.get("rank_polymorphic", False))
        ch = raw.get("channel_count")
        ch_ax = raw.get("channel_axis")
        l = raw.get("len")
        if shape is not None and ndim is None and not poly:
            ndim = len(shape)
        if l is not None and ndim is None:
            ndim = 1
            shape = [l]
        if shape is not None:
            shape = list(shape)
        return {
            "ndim": ndim,
            "shape": shape,
            "rank_polymorphic": poly,
            "channel_count": ch,
            "channel_axis": ch_ax,
            "len": l,
        }
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in ("none", "null", "any", "*"):
            return {
                "ndim": None,
                "shape": None,
                "rank_polymorphic": True,
                "channel_count": None,
                "channel_axis": None,
                "len": None,
            }
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip()]
        shape_elements: List[Union[int, str]] = []
        for p in parts:
            if p.isdigit():
                shape_elements.append(int(p))
            else:
                shape_elements.append(p)
        ndim = len(shape_elements)
        poly = any(p in ("*", "...", "any") for p in shape_elements)
        ch = None
        ch_ax = None
        l = None
        if ndim == 3 and shape_elements[0] in ("H", "height", "rows") and isinstance(shape_elements[2], int):
            ch = shape_elements[2]
            ch_ax = -1
        elif ndim == 1 and isinstance(shape_elements[0], int):
            l = shape_elements[0]
        return {
            "ndim": None if poly else ndim,
            "shape": shape_elements,
            "rank_polymorphic": poly,
            "channel_count": ch,
            "channel_axis": ch_ax,
            "len": l,
        }
    return None


def migrate_all():
    tree_files = sorted(glob.glob("trees/*.json"))
    print(f"Migrating {len(tree_files)} tree files...")

    for tree_path in tree_files:
        filename = os.path.basename(tree_path)
        with open(tree_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        domain = data.get("domain", filename.replace(".json", ""))

        # 1. Normalize existing typestates to {"domain": ..., "states": [...]}
        existing_states_map: Dict[str, Dict[str, Any]] = {}
        ts_block = data.get("typestates")
        if isinstance(ts_block, dict):
            if "states" in ts_block:
                for s in ts_block.get("states", []):
                    if isinstance(s, dict) and "name" in s:
                        existing_states_map[s["name"]] = s
                    elif isinstance(s, str):
                        existing_states_map[s] = {"name": s}
            else:
                for k, v in ts_block.items():
                    if isinstance(v, dict):
                        entry = {"name": k}
                        entry.update(v)
                        existing_states_map[k] = entry
                    else:
                        existing_states_map[k] = {"name": k, "description": str(v)}
        elif isinstance(ts_block, list):
            for s in ts_block:
                if isinstance(s, dict) and "name" in s:
                    existing_states_map[s["name"]] = s
                elif isinstance(s, str):
                    existing_states_map[s] = {"name": s}

        # 2. Iterate cells and ports to apply corrections
        for cell in data.get("cells", []):
            cid = cell.get("cell_id", "")
            stage = cell.get("stage", 2)

            for port_type, ports in [("inputs", cell.get("inputs", {})), ("outputs", cell.get("outputs", {}))]:
                for p_name, port in ports.items():
                    st = port.get("state")
                    tn = port.get("type_name", "")
                    sk = port.get("structure_kind")
                    r = port.get("rank")
                    sc = port.get("shape_contract")
                    role = port.get("port_role") or port.get("role")
                    default = port.get("default_value")
                    req = port.get("required", True)

                    # --- Pillow RGB/RGBA rank convention ---
                    if domain == "pillow":
                        if cid in (
                            "PILLOW_RESIZE_LANCZOS", "PILLOW_RESIZE_BILINEAR", "PILLOW_RESIZE_NEAREST",
                            "PILLOW_CROP", "PILLOW_ROTATE", "PILLOW_TRANSPOSE_FLIP_LEFT_RIGHT",
                            "PILLOW_TRANSPOSE_FLIP_TOP_BOTTOM", "PILLOW_TRANSPOSE_ROTATE_90",
                            "PILLOW_FILTER_GAUSSIAN_BLUR", "PILLOW_FILTER_BOX_BLUR", "PILLOW_FILTER_FIND_EDGES",
                            "PILLOW_FILTER_CONTOUR", "PILLOW_FILTER_SHARPEN", "PILLOW_FILTER_UNSHARP_MASK",
                            "PILLOW_FILTER_MEDIAN", "PILLOW_FILTER_SMOOTH", "PILLOW_IMAGE_SAVE",
                            "PILLOW_IMAGE_SAVE_JPEG", "PILLOW_IMAGE_SAVE_PNG", "PILLOW_IMAGE_SHOW",
                            "PILLOW_IMAGEOPS_EXIF_TRANSPOSE", "PILLOW_IMAGEOPS_INVERT",
                            "PILLOW_IMAGEOPS_AUTOCONTRAST", "PILLOW_IMAGEOPS_PAD", "PILLOW_IMAGEOPS_FIT",
                            "PILLOW_IMAGEOPS_CONTAIN", "PILLOW_IMAGEOPS_COVER"
                        ) and (p_name in ("image", "image_src", "image_dst", "output_var") or tn == "Image"):
                            # Rank-polymorphic operations
                            port["rank"] = None
                            port["shape_contract"] = {
                                "ndim": None,
                                "shape": None,
                                "rank_polymorphic": True,
                                "channel_count": None,
                                "channel_axis": None,
                                "len": None,
                            }
                            if port_type == "inputs" and p_name in ("image", "image_src", "image_dst"):
                                port["accepted_states"] = [
                                    "mode_rgb", "mode_rgba", "mode_grayscale", "mode_binary_1bit",
                                    "mode_cmyk", "mode_palette", "mode_hsv", "mode_lab",
                                    "mode_int32", "mode_float32", "resized", "cropped", "rotated"
                                ]
                        # Multi-channel color images are rank 3
                        elif st in ("mode_rgb", "mode_bgr", "mode_hsv", "mode_lab", "mode_ycbcr"):
                            port["rank"] = 3
                            port["shape_contract"] = {
                                "ndim": 3,
                                "shape": ["H", "W", 3],
                                "rank_polymorphic": False,
                                "channel_count": 3,
                                "channel_axis": -1,
                                "len": None,
                            }
                        elif st in ("mode_rgba", "mode_cmyk"):
                            port["rank"] = 3
                            port["shape_contract"] = {
                                "ndim": 3,
                                "shape": ["H", "W", 4],
                                "rank_polymorphic": False,
                                "channel_count": 4,
                                "channel_axis": -1,
                                "len": None,
                            }
                        elif st in ("mode_grayscale", "mode_binary_1bit", "mode_palette", "mode_int32", "mode_float32"):
                            port["rank"] = 2
                            port["shape_contract"] = {
                                "ndim": 2,
                                "shape": ["H", "W"],
                                "rank_polymorphic": False,
                                "channel_count": 1,
                                "channel_axis": None,
                                "len": None,
                            }

                    # --- Structural Kind & Rank Anomalies Across Domains ---
                    # 1. Any List[...] or collection/sequence port cannot have rank 0 or structure_kind handle
                    if tn.startswith("List[") or tn in ("list", "sequence", "collection", "set"):
                        if sk in ("handle", "composite", "scalar", None):
                            port["structure_kind"] = "collection"
                        if not port.get("abstract_type") or port.get("abstract_type") in ("scalar", "handle", "standard"):
                            port["abstract_type"] = "collection"
                        if r == 0 or r is None:
                            port["rank"] = 1
                        if port.get("shape_contract") is None:
                            port["shape_contract"] = {
                                "ndim": 1,
                                "shape": ["N"],
                                "rank_polymorphic": False,
                                "channel_count": None,
                                "channel_axis": None,
                                "len": None,
                            }
                    # 2. Contours in cv2: List[Contour] is a 1D collection of 2D/3D point arrays
                    elif tn == "List[Contour]":
                        port["structure_kind"] = "collection"
                        port["abstract_type"] = "collection"
                        port["rank"] = 1
                        port["element_type"] = "ndarray"
                        port["shape_contract"] = {
                            "ndim": 1,
                            "shape": ["N"],
                            "rank_polymorphic": False,
                            "channel_count": None,
                            "channel_axis": None,
                            "len": None,
                        }
                    # 3. Axes and Figure in seaborn are handles
                    elif domain == "seaborn" and tn in ("Axes", "Figure"):
                        port["structure_kind"] = "handle"
                        port["abstract_type"] = "plot_handle"
                        port["rank"] = None
                    # 4. Builtins list/range
                    elif domain == "builtins" and tn in ("list", "tuple") and port_type == "outputs":
                        port["structure_kind"] = "collection"
                        port["abstract_type"] = "collection"
                        port["rank"] = 1
                    # 5. cv2 handle objects misclassified as scalar
                    elif domain == "cv2" and tn in ("cv2.CLAHE", "cv2.VideoCapture", "cv2.VideoWriter", "cv2.SIFT", "cv2.ORB", "cv2.BFMatcher"):
                        port["structure_kind"] = "handle"
                        port["abstract_type"] = "handle"
                        port["rank"] = None
                    # 6. numpy frombuffer
                    elif domain == "numpy" and cid == "NUMPY_FROMBUFFER" and p_name == "buffer":
                        port["structure_kind"] = "tensor"
                        port["abstract_type"] = "tensor"
                        port["rank"] = 1

                    # --- Shape Contract Canonicalization ---
                    # If not already unified, parse it now
                    cur_sc = port.get("shape_contract")
                    if cur_sc is not None and (isinstance(cur_sc, str) or (isinstance(cur_sc, dict) and "rank_polymorphic" not in cur_sc)):
                        port["shape_contract"] = parse_shape_contract(cur_sc)

                    # --- Required Scalar Port Roles & Defaults ---
                    is_scalar = (port.get("structure_kind") == "scalar") or (port.get("rank") == 0 and port.get("abstract_type") in ("scalar", "primitive", "standard", "logical", "text"))
                    if port_type == "inputs" and req and is_scalar:
                        # Ingress path/file in S1
                        if role in ("source_data", "source_identifier") or (stage == 1 and p_name in ("filepath", "file_path", "database_path", "image_path")):
                            if not role:
                                port["port_role"] = "source_data"
                                port["role"] = "source_data"
                        elif (filename, cid, p_name) in CELL_PARAM_DEFAULTS:
                            c_def, c_role = CELL_PARAM_DEFAULTS[(filename, cid, p_name)]
                            port["port_role"] = c_role
                            port["role"] = c_role
                            if default is None:
                                port["default_value"] = c_def
                        elif (filename, cid, p_name) in DATA_INPUT_CELL_PORTS:
                            port["port_role"] = "data_input"
                            port["role"] = "data_input"
                        elif p_name in PARAM_DEFAULTS:
                            if not role:
                                port["port_role"] = "configuration"
                                port["role"] = "configuration"
                            if default is None:
                                port["default_value"] = PARAM_DEFAULTS[p_name]
                        elif domain == "seaborn" and p_name.startswith("port_"):
                            # Seaborn column projections or style configs
                            desc = (port.get("description") or "").lower()
                            if "column" in desc or "variable" in desc:
                                port["port_role"] = "column_projection"
                                port["role"] = "column_projection"
                                if default is None:
                                    port["default_value"] = ""
                            else:
                                port["port_role"] = "configuration"
                                port["role"] = "configuration"
                                if default is None:
                                    port["default_value"] = "default"
                        else:
                            # General configuration fallback
                            if not role:
                                port["port_role"] = "configuration"
                                port["role"] = "configuration"
                            if default is None:
                                if port.get("type_name") in ("int", "int32", "int64"):
                                    port["default_value"] = 0
                                elif port.get("type_name") in ("float", "float32", "float64"):
                                    port["default_value"] = 0.0
                                elif port.get("type_name") in ("bool", "boolean"):
                                    port["default_value"] = False
                                else:
                                    port["default_value"] = ""

                    # Collect typestate if present and not yet declared
                    cur_st = port.get("state")
                    if cur_st and cur_st not in ("any", "*", "default", "unregistered"):
                        if cur_st not in existing_states_map:
                            existing_states_map[cur_st] = {
                                "name": cur_st,
                                "parent_state": port.get("parent_state"),
                                "carrier_type": port.get("type_name"),
                                "description": port.get("description") or f"{cur_st} typestate",
                                "properties": port.get("value_constraints") or {},
                            }

        # 3. Assemble normalized typestates block
        sorted_states = [existing_states_map[k] for k in sorted(existing_states_map.keys())]
        data["typestates"] = {
            "domain": domain,
            "states": sorted_states,
            "transitions": ts_block.get("transitions", []) if isinstance(ts_block, dict) else [],
            "initial_state": ts_block.get("initial_state") if isinstance(ts_block, dict) else None,
            "terminal_states": ts_block.get("terminal_states", []) if isinstance(ts_block, dict) else [],
        }

        # 4. Save updated tree
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    print("Migration complete for all trees!")


if __name__ == "__main__":
    migrate_all()
