#!/usr/bin/env python3
"""Download official Mistral-7B-Instruct weights into the project models/ tree."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_OUT = ROOT / "models" / "mistral" / "Mistral-7B-Instruct-v0.3"


def main() -> int:
    p = argparse.ArgumentParser(description="Download Mistral weights to models/mistral/")
    p.add_argument("--model-id", default=os.environ.get("MISTRAL_MODEL_ID", DEFAULT_MODEL_ID))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    args = p.parse_args()

    out = args.out_dir.resolve()

    def _weights_complete() -> bool:
        if not (out / "config.json").is_file():
            return False
        idx = out / "model.safetensors.index.json"
        if idx.is_file():
            import json

            meta = json.loads(idx.read_text(encoding="utf-8"))
            shards = sorted(set(meta.get("weight_map", {}).values()))
            return shards and all((out / s).is_file() for s in shards)
        for name in ("model.safetensors", "consolidated.safetensors"):
            if (out / name).is_file():
                return True
        return False

    if _weights_complete():
        print(f"Already present: {out}")
        print(f"Set lora.local_model_path in configs/model_config.yaml to:\n  {out}")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model_id} -> {out}")
    if not args.token:
        print(
            "Note: If the repo is gated, set HF_TOKEN or run: huggingface-cli login",
            file=sys.stderr,
        )

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(out),
        token=args.token,
    )
    if not _weights_complete():
        print(f"Download finished but weight shards missing under {out}", file=sys.stderr)
        return 1

    print(f"Done: {out}")
    print(f"configs/model_config.yaml -> lora.local_model_path: \"{out.as_posix()}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
