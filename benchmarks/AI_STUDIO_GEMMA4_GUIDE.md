# Testing Gemma 4 31B (`gemma-4-31b-it`) in Google AI Studio (v2 Protocol)

This guide runs a **controlled 3-arm comparison** in Google AI Studio (**Gemma 4 31B**) against local/Colab models (**STRATA v2**, **Pure GGUF**, **SKILL.md GGUF**, and **SKILL.md + Haystack**) over the expanded 19-task v2 suite (`v2_expanded`) plus optional stress tasks (`v2_full`).

---

## Step 1: Google AI Studio Configuration (CRITICAL — Strict Fairness Rules)

1. Open [Google AI Studio](https://aistudio.google.com/) -> **Create new prompt**.
2. **Model**: **`Gemma 4 31B`** (`gemma-4-31b-it`).
3. **Tools (Disable ALL Tools!)**:
   - **Code Execution**: **OFF** *(must NOT execute Python in Google's cloud or self-correct errors)*.
   - **Grounding with Google Search**: **OFF**.
   - **URL Context**: **OFF**.
   - **Function Calling**: **OFF**.
   - **Structured Outputs**: **OFF**.
4. **Generation Parameters**:
   - **Temperature**: `0.0`
   - **Top P**: `0.95`
   - **Max Output Tokens**: `4096`
   - **Thinking**: Leave default or Minimum (`benchmarks/ingest_aistudio.py` automatically strips `<think>...</think>` blocks).
5. **Fresh Chat Rule (Mandatory)**: Open a **brand-new chat** for every single prompt! Sharing chat history leaks previous SDKs (`cryopulse`, `aetheris`), wrong-domain tasks, or earlier solutions across tasks.

---

## Step 2: Generate Ready-to-Paste System + Prompt Packs (Recommended)

Instead of manually stitching domain docs, generate exact markdown packs matching Colab/local runs:

```bash
# Arm A — Pure Zero-Shot (no .md docs; includes only env library notes for wrong_domain/domain_spec)
python benchmarks/ingest_aistudio.py --export-prompts benchmarks/aistudio_packs --suite v2_expanded --context-mode pure

# Arm B — Exact Loaded Domain SKILL.md in Chat (fair SKILL.md vs STRATA v2 comparison)
python benchmarks/ingest_aistudio.py --export-prompts benchmarks/aistudio_packs --suite v2_expanded --context-mode skill_md

# Arm C — Exact SKILL.md + Full 1,049-Cell 15-Domain Haystack (uses AI Studio's long context without truncation!)
python benchmarks/ingest_aistudio.py --export-prompts benchmarks/aistudio_packs --suite v2_haystack --context-mode skill_md_haystack --full-haystack
```

Each task generates `benchmarks/aistudio_packs/<mode>/<idx>_<task_id>.md` with exact copy-paste blocks for **System Instructions** and **User Prompt**.

---

## Step 3: The 3-Arm Benchmark Suite (`v2_expanded` — 19 Tasks)

### Group 1: `v2_nl_core` (`10` Tasks — Detailed Unvague vs. Underspecified Refusals)

**Arm A (`pure`) System Instruction** (all 10 tasks use `default` full environment):
```text
You are a concise, precise Python developer. Write ONLY clean executable Python code inside a single ```python ... ``` code block solving the user's request. Use standard/installed libraries only. Preserve every literal value, filename, column name, and parameter specified in the prompt. If the prompt is genuinely too underspecified to write concrete working code without guessing critical missing rules, missing required file paths, or requesting an unavailable library, ask 2 concise clarifying questions or state the missing requirement instead of writing speculative or substitute code.
```
**Arm B (`skill_md`) System Instruction:** Copy the generated System Instruction from `benchmarks/aistudio_packs/skill_md/01_v2_nl_cv2_params.md` (same 15-domain `SKILL.md` header applies to all 10 `v2_nl_core` tasks).

Run each prompt in a **fresh chat**:
1. `v2_nl_cv2_params` *(expected: synthesize, cv2)*:
   > `read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"`
2. `v2_nl_sqlite_pandas` *(expected: synthesize, sqlite3 + pandas)*:
   > `open an in-memory sqlite database connection, run SQL query "SELECT 1 AS a, 2 AS b" into a pandas dataframe, and close the sqlite connection`
3. `v2_nl_cv2_implicit_params` *(expected: synthesize, cv2 Canny)*:
   > `read image "input.png", convert to grayscale, and detect Canny edges`
4. `v2_nl_unvague_pillow` *(expected: synthesize, strictly `PIL`, zero `cv2`)*:
   > `using Pillow, read image "input.png", convert to grayscale, and save to "pil_gray.png"`
5. `v2_nl_unvague_pygame` *(expected: synthesize, pygame window lifecycle)*:
   > `initialize pygame, create a display window of size (640, 480), flip the display surface, and quit pygame`
6. `v2_nl_unvague_pandas_etl` *(expected: synthesize, pandas CSV ETL)*:
   > `load csv file "input.csv", drop missing rows, and save the dataframe to "clean_out.csv"`
7. `v2_nl_vague_game` *(expected: clarify / no code)*:
   > `make a 2d platformer game`
8. `v2_nl_vague_ml` *(expected: clarify / no code)*:
   > `load dataset and train a machine learning model to predict target`
9. `v2_nl_audio_missing_file` *(expected: clarify / no code)*:
   > `load audio file, apply high-pass filter, compute spectrogram, and export filtered audio`
10. `v2_nl_bound_ambiguous` *(expected: clarify / no code)*:
    > `build an end-to-end anomaly detection microservice with alerts and automatic retraining`

---

### Group 2: `v2_unseen` (`2` Tasks — Zero-Pretraining Blind SDKs)

- **Arm A (`pure`):** Use the plain System Instruction above with **ZERO `.md` docs** (measures whether zero-shot LLM memorization fails on novel SDKs).
- **Arm B (`skill_md`):** Paste [`benchmarks/fixtures/cryopulse/cryopulse_docs.md`](fixtures/cryopulse/cryopulse_docs.md) for task 11, and [`benchmarks/fixtures/aetheris/aetheris_docs.md`](fixtures/aetheris/aetheris_docs.md) for task 12 inside System Instructions (using the strict `Use ONLY the API documented below...` wrapper from the generated pack):

11. `v2_blind_cryopulse_nl` (`cryopulse_only`):
    > `open cryo vessel 'DR-BLUE-04' at 12.5 millikelvin, energize flux coil on rail 'RAIL-X2' at 38.0 microamps, sample squid flux for 512 ns with 2048 shots, demodulate quadrature at 6.15 GHz, quench tls defects with damping gamma 0.18, compile pi schedule with target fidelity 0.998, emit cryo manifest to 'calibration.cym', and vent cryo vessel`
12. `v2_blind_aetheris_nl` (`aetheris_only`, note alphabetical ordering & 4 trap APIs):
    > `mount optical table 'OPT-NORTH-2' at wavelength 1550.0 nm, lock reference laser on cavity mode 'TEM00' at 18.5 mW, arm phase modulator with 16 lanes and vpi 3.6 volts, sweep fringe interferogram over 65.0 pm with 1024 frames, unwrap phase residues with 'goldstein' branch cut, solve zernike wavefront at order 8, export phase lut to 'zernike_n2.ael', and park optical table`

---

### Group 3: `v2_haystack` (`2` Tasks — Blind SDKs Inside 15-Domain Catalog Noise)

Use the System Instructions generated by `--context-mode skill_md_haystack` (`benchmarks/aistudio_packs/skill_md_haystack/01_v2_haystack_cryopulse_1049_noise.md` and `02_v2_haystack_aetheris_1049_noise.md` — embed the target SDK docs at the midpoint of all 15 standard domains / `1,049` distractor cells):

13. `v2_haystack_cryopulse_1049_noise`: Same prompt as task 11 (`v2_blind_cryopulse_nl`), evaluated with `strict_sdk_only=True` (any distractor import fails `noise_imports_used`).
14. `v2_haystack_aetheris_1049_noise`: Same prompt as task 12 (`v2_blind_aetheris_nl`), evaluated with `strict_sdk_only=True`.

---

### Group 4: `v2_wrong_domain` (`2` Tasks — Wrong / Missing Domain Refusal Controls)

Expected outcome: **refuse cleanly without writing code** (`REFUSED_OK` if no code block emitted; emitting executable code fails with `DOMAIN_VIOLATION`). Use the exact System Instruction from each generated pack so the available/missing domain list is explicit:

15. `v2_wrong_domain_image_in_sqlite` *(only `python_core` + `sqlite3` loaded)*:
    > `read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"`
16. `v2_wrong_domain_cryo_not_loaded` *(15 standard domains loaded; `cryopulse` NOT loaded)*:
    > Same prompt as task 11 (`v2_blind_cryopulse_nl`).

---

### Group 5: `v2_domain_spec` (`3` Tasks — Explicit Library Specification Adherence)

17. `v2_spec_pillow_only_cv2_nl` *(only `python_core + numpy + cv2` loaded — **expected: refuse, must NOT substitute `cv2`**)*:
    > `using Pillow, read image "input.png", convert to grayscale, and save to "pil_gray.png"`
18. `v2_spec_pillow_both_loaded_nl` *(both `cv2` & `pillow` loaded — **expected: strictly `PIL`, zero `cv2`**)*:
    > `using Pillow, read image "input.png", convert to grayscale, and save to "pil_gray.png"`
19. `v2_spec_cv2_both_loaded_nl` *(both `cv2` & `pillow` loaded — **expected: strictly `cv2`, zero `PIL`**)*:
    > `using OpenCV (cv2), read image "input.png", convert to grayscale, and save to "cv2_gray.png"`

*(Optional extra tasks in `--suite v2_full`: `cryo_implicit`, `cryo_diag`, `aetheris_implicit`, `aetheris_diag`, `user_etl_sqlite`, `user_cv_contour`, `user_audio_fft_export`, `bound_custom_algo`.)*

---

## Step 4: Save, Verify & Visualize All Arms Together

Save each arm's raw replies into separate text files (`gemma4_v2_pure.txt`, `gemma4_v2_skill_md.txt`, `gemma4_v2_haystack.txt`) using exact task IDs:

```text
=== TASK: v2_nl_cv2_params ===
<paste Gemma 4 31B raw response here>
=== END ===

=== TASK: v2_spec_pillow_only_cv2_nl ===
<paste Gemma 4 31B raw response here>
=== END ===
```

Then verify every arm against STRATA's live GEVR sandbox & SDK typestate auditors:

```bash
python benchmarks/ingest_aistudio.py --input gemma4_v2_pure.txt     --suite v2_expanded --context-mode pure              --out test_logs/aistudio_v2/results_gemma4_31b_pure.json
python benchmarks/ingest_aistudio.py --input gemma4_v2_skill_md.txt --suite v2_expanded --context-mode skill_md          --out test_logs/aistudio_v2/results_gemma4_31b_skill_md.json
python benchmarks/ingest_aistudio.py --input gemma4_v2_haystack.txt --suite v2_haystack --context-mode skill_md_haystack --out test_logs/aistudio_v2/results_gemma4_31b_haystack.json
```

Compare **STRATA v2** + **Local GGUF** + **Colab** + **Gemma 4 AI Studio** across all categories in one table + 4 PNG figures:

```bash
python benchmarks/visualize_results.py \
  test_logs/local_bench test_logs/colab_v2_pure test_logs/colab_v2_skill_md test_logs/colab_v2_haystack test_logs/aistudio_v2 \
  --out-dir test_logs/visualizations
```
