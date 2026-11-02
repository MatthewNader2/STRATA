# debug_engine.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import asyncio
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from main import initialize_engine, run_prompt, InitRequest, RunRequest

async def debug():
    req = InitRequest(profile="C", embedder_model="jina-embeddings-v5-text-small",
                      llm_model="qwen2.5-coder-1.5b-instruct")
    init_result = initialize_engine(req)
    print(f"Init type: {type(init_result)}, keys: {list(init_result.keys())}")
    print("=== Engine initialized ===")

    for prompt in [
        "Read data.csv, drop rows with missing values, sort by the 'age' column descending, and save to output.csv",
    ]:
        print(f"\n>>> PROMPT: {prompt[:60]}...")
        try:
            raw = run_prompt(RunRequest(prompt=prompt))
            print(f"run_prompt returned type: {type(raw)}")
            print(f"All keys: {list(raw.keys())}")
            print(f"CODE (repr): {repr(raw.get('code', 'NO_CODE_KEY'))}")
            print(f"message: {raw.get('message', 'NO_MSG')}")
            print(f"status: {raw.get('status', 'NO_STATUS')}")
            print(f"generated_code: {repr(raw.get('generated_code', 'NO_GEN'))}")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            traceback.print_exc()

asyncio.run(debug())
