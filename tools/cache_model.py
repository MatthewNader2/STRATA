import os
import sys
from huggingface_hub import snapshot_download

def download_and_save():
    model_name = "jinaai/jina-embeddings-v5-text-nano"
    print(f"Downloading model files for '{model_name}'...")
    try:
        # Fetch the entire repository snapshot directly to model_cache
        snapshot_download(
            repo_id=model_name,
            local_dir="model_cache",
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        print("Success! Model files saved locally in model_cache.")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    download_and_save()
