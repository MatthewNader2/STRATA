# Testing Gemma 4 31B (`gemma-4-31b-it`) in Google AI Studio

This guide ensures a **100% controlled, honest apples-to-apples comparison** between Google AI Studio (**Gemma 4 31B**) and local/Colab models (**STRATA O+OMEGA 0.8B**, **Pure 0.8B**, **Pure 7B**, **Qwen3.8-27B**, **GPT-OSS 20B**, **Qwen3-Coder-30B-A3B**, **Qwen3.5-35B-A3B**).

---

## Step 1: Google AI Studio Configuration (Left/Right Sidebar)

1. Open [Google AI Studio](https://aistudio.google.com/) -> **Create new prompt** (Chat / Freeform).
2. **Model**: Select **`Gemma 4 31B`** (`gemma-4-31b-it`).
3. **Tools (CRITICAL — Disable ALL Tools!)**:
   - **Code Execution**: **OFF** *(must NOT run Python in Google's cloud and self-correct errors — STRATA's pure-model baseline evaluates raw one-shot code synthesis against our local GEVR sandbox)*.
   - **Grounding with Google Search**: **OFF** *(unseen SDKs like `cryopulse`/`aetheris` must rely only on the prompt + `.md` doc)*.
   - **URL Context**: **OFF**.
   - **Function Calling**: **OFF**.
   - **Structured Outputs**: **OFF**.
4. **Generation Parameters**:
   - **Temperature**: `0.0` (deterministic greedy decoding, matching local benchmarks).
   - **Top P**: `0.95`.
   - **Max Output Tokens**: `4096` (prevents truncation if Gemma 4 uses thinking tokens).
   - **Thinking / Reasoning**: If a **Thinking** toggle is shown, you may leave it **Default/Auto** (or **Minimum/Low** if you want faster runs) — our ingestion parser (`benchmarks/ingest_aistudio.py`) automatically strips `<think>...</think>` blocks and extracts the final ```python ... ``` block.
5. **Fresh Chat Rule**: Start a **new chat** (or clear history) for each task so earlier prompts/docs never leak context into subsequent tasks!

---

## Step 2: Core 9 Suites (11 Prompts Total)

### Part A — Tasks 1 to 8 (`core8`: Standard Python / Sandbox Tasks)

**Paste this into "System Instructions":**
```text
You are a concise, precise Python developer. Write ONLY clean executable Python code inside a single ```python ... ``` code block solving the user's request. Use standard/installed libraries only. Preserve every literal value, filename, column name, and parameter specified in the prompt. If the prompt is genuinely too underspecified to write concrete working code without guessing critical missing rules, ask 2 concise clarifying questions instead of writing speculative code.
```

Run each of these 8 prompts in a **fresh chat** (copy-paste exact string into User box):

1. `canonical`:
   > `load a csv file named "input.csv", normalize X column, drop null values, get the mean of the Y column, and perform FFT and write it to a new column called Z, then make a regression model trained on X and Y to predict Z`
2. `vague_game`:
   > `make a 2D game in pygame with a display window, game loop, clock, and quit`
3. `unvague_game`:
   > `initialize pygame and font system, create an 800 by 600 display window with title 'Snake Game', initialize clock, create an Arial size 24 font, create snake head rect at x 400 y 300 width 20 height 20, create food rect at x 200 y 200 width 20 height 20, create boundary rect at x 0 y 0 width 800 height 600, create background color black 0 0 0 255, green snake color 0 255 0 255, red food color 255 0 0 255, and white text color 255 255 255 255, run main game loop while running is True: get events and key presses, move snake rect by dx 20 and dy 0, clamp snake rect to boundary rect, check rect collision between snake rect and food rect, fill screen surface with black color, draw green snake rect and red food rect on screen surface, render score text surface 'Score: 0' with white color and blit it at position (10, 10) onto screen, tick clock at 15 fps, flip display, and finally quit font and quit pygame`
4. `vague_ml`:
   > `load dataset and train a machine learning model to predict target`
5. `cv2_params`:
   > `read image "input.png", convert to grayscale, apply Gaussian blur with kernel size 5 and sigma 1.5, detect Canny edges with threshold1 50 and threshold2 150, and save to "edges.png"`
6. `control_flow`:
   > `define a function that processes data with a while loop, branches on condition with if else, and handles exceptions with try except`
7. `crypto`:
   > `read file 'input.csv', hash data using sha256, encrypt payload with aes cipher, and encode with base64`
8. `audio`:
   > `load audio file, apply high-pass filter, compute spectrogram, and export filtered audio`

---

### Part B — Task 9 (`cryopulse`: Unseen Invented Domain — 3 Sub-Tasks)

**Replace "System Instructions" with:**
```text
You are a Python coding assistant. Use ONLY the API documented below to write a single executable Python script inside a ```python ... ``` code block solving the user's request.
```
And immediately append the full contents of [`benchmarks/fixtures/cryopulse/cryopulse_docs.md`](fixtures/cryopulse/cryopulse_docs.md) inside System Instructions.

Run these 3 prompts (fresh chat for each):

9a. `cryo_explicit`:
> `open cryo vessel 'DR-BLUE-04' at 12.5 millikelvin, energize flux coil on rail 'RAIL-X2' at 38.0 microamps, sample squid flux for 512 ns with 2048 shots, demodulate quadrature at 6.15 GHz, quench tls defects with damping gamma 0.18, compile pi schedule with target fidelity 0.998, and emit cryo manifest to 'calibration.cym'`

9b. `cryo_implicit`:
> `initialize cryo vessel 'DR-RED-09', capture squid flux trace, demodulate quadrature at 5.90 GHz, compile pi pulse schedule at fidelity 0.997, and emit cryo manifest to 'red09.cym'`

9c. `cryo_diag`:
> `open cryo vessel 'QX-ALPHA', measure mixing plate temperature, energize flux coil on rail 'FLUX-01', sample squid flux, audit squid snr, and demodulate quadrature`

---

## Step 3 (Optional): Tier-2 Stress Suite Prompts (`stress`)

If you also want to test **Gemma 4 31B** on the harder STRATA theory stress suite:
- **Realistic User + Boundary tasks (`user_etl_sqlite`, `user_cv_contour`, `user_audio_fft_export`, `bound_ambiguous`, `bound_custom_algo`)**: use the **Part A** System Instruction; exact prompts are listed in [`benchmarks/suites.py`](suites.py).
- **Aetheris (`aetheris_explicit`, `aetheris_implicit`, `aetheris_diag`)**: use **Part B** System Instruction + [`benchmarks/fixtures/aetheris/aetheris_docs.md`](fixtures/aetheris/aetheris_docs.md) (alphabetical reference with strict preconditions & trap APIs).

---

## Step 4: Save & Ingest Responses Automatically

Save your Gemma 4 31B outputs into a file `gemma4_responses.txt` using this exact delimiter format:

```text
=== TASK: canonical ===
<paste Gemma 4 31B raw response here>
=== END ===

=== TASK: vague_game ===
<paste Gemma 4 31B raw response here>
=== END ===
...
```

Then verify & score all responses against STRATA's sandbox and SDK auditors:
```bash
python benchmarks/ingest_aistudio.py --input gemma4_responses.txt --suite all --out test_logs/colab_bench/results_gemma4_31b.json
```
And regenerate all comparison tables & PNG charts:
```bash
python benchmarks/visualize_results.py test_logs/bench_round2_part1_v2 test_logs/bench_round2_part1/results_pure_*.json test_logs/bench_round2_part2_omega_08b_v2 test_logs/bench_round2_part2_pure_* test_logs/colab_bench --out-dir test_logs/visualizations
```
