from __future__ import annotations

"""
Zero-shot LLM classification on real JSONL (train_mistral.jsonl).

Backends:
  - ollama (default): local Ollama API — use OLLAMA_MODEL or config mistral_infer.ollama_model
  - hf: Hugging Face transformers (Mistral-7B etc., needs HF_TOKEN if gated)

Outputs:
  data/processed/mistral/zero_shot_results.jsonl
  data/processed/mistral/metrics.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wave_llm.evaluation.mistral_plots import extract_json_object  # noqa: E402
from wave_llm.models.ollama_infer import ollama_chat  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _maybe_hf_login() -> None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        return
    try:
        from huggingface_hub import login

        login(token=tok, add_to_git_credential=False)
        logging.info("HF_TOKEN present: logged in to Hugging Face Hub")
    except Exception as e:
        logging.warning("HF login skipped/failed: %s", e)


def build_user_prompt(rec: dict) -> str:
    instr = rec.get("instruction", "")
    inp = rec.get("input", {})
    return (
        f"{instr}\n\n"
        "下面是结构化输入（JSON）。请只输出一个 JSON 对象，不要 Markdown，不要解释文字。"
        "字段必须包含：wave_regime（字符串）, predictability_24h（high|medium|low 之一）, notes（简短中文）。\n\n"
        f"INPUT_JSON:\n{json.dumps(inp, ensure_ascii=False)}\n"
    )


def chat_messages(rec: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You classify buoy wave regimes. Reply with JSON only, no markdown fences.",
        },
        {"role": "user", "content": build_user_prompt(rec)},
    ]


def infer_hf(
    rec: dict,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    _model_cache: dict,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "model" not in _model_cache:
        logging.info("Loading HF model %s", model_id)
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs: dict = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not torch.cuda.is_available():
            model.to("cpu")
        _model_cache["model"] = model
        _model_cache["tok"] = tok

    model = _model_cache["model"]
    tok = _model_cache["tok"]
    messages = chat_messages(rec)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(1e-4, temperature),
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--backend", choices=("ollama", "hf"), default=None)
    args = ap.parse_args()

    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    mic = mcfg.get("mistral_infer", {}) or {}
    backend = args.backend or os.environ.get("MISTRAL_BACKEND") or mic.get("backend", "ollama")
    ollama_model = os.environ.get("OLLAMA_MODEL", mic.get("ollama_model", "deepseek-r1:14b"))
    model_id = os.environ.get("MISTRAL_MODEL_ID", mic.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3"))
    max_samples = int(args.max_samples or mic.get("max_samples", 8))
    max_new_tokens = int(mic.get("max_new_tokens", 320))
    temperature = float(mic.get("temperature", 0.1))

    jsonl = ROOT / "data" / "processed" / "llm" / "train_mistral.jsonl"
    if not jsonl.exists():
        logging.error("Missing %s — run scripts/06_export_llm_jsonl.py first", jsonl)
        return 1

    out_dir = ensure_dir(ROOT / "data" / "processed" / "mistral")
    hf_cache: dict = {}

    rows_out: list[dict] = []
    y_true_r: list[str] = []
    y_pred_r: list[str] = []
    y_true_p: list[str] = []
    y_pred_p: list[str] = []

    with jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            rec = json.loads(line)
            out0 = rec.get("output", {}) or {}
            true_r = str(out0.get("wave_regime", "")).strip()
            true_p = str(out0.get("predictability_24h", "")).strip().lower()

            try:
                if backend == "ollama":
                    logging.info("Ollama infer sample %s/%s model=%s", i + 1, max_samples, ollama_model)
                    text = ollama_chat(chat_messages(rec), model=ollama_model, temperature=temperature)
                else:
                    _maybe_hf_login()
                    text = infer_hf(rec, model_id, max_new_tokens, temperature, hf_cache)
            except Exception as e:
                logging.error("Inference failed sample %s: %s", i, e)
                text = ""

            parsed = extract_json_object(text) or {}
            pred_r = str(parsed.get("wave_regime", "")).strip()
            pred_p = str(parsed.get("predictability_24h", "")).strip().lower()

            rows_out.append(
                {
                    "sample_idx": i,
                    "station_id": (rec.get("input") or {}).get("station_id"),
                    "true_wave_regime": true_r,
                    "pred_wave_regime": pred_r,
                    "true_predictability_24h": true_p,
                    "pred_predictability_24h": pred_p,
                    "raw_model_text": text,
                    "parsed_json": parsed,
                }
            )
            if true_r:
                y_true_r.append(true_r)
                y_pred_r.append(pred_r or "parse_failed")
            if true_p:
                y_true_p.append(true_p)
                y_pred_p.append(pred_p or "parse_failed")

    (out_dir / "zero_shot_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out) + "\n",
        encoding="utf-8",
    )

    acc_r = float(np.mean([a == b for a, b in zip(y_true_r, y_pred_r)])) if y_true_r else None
    acc_p = float(np.mean([a == b for a, b in zip(y_true_p, y_pred_p)])) if y_true_p else None
    reported_model = ollama_model if backend == "ollama" else model_id
    metrics = {
        "backend": backend,
        "model_id": reported_model,
        "n_samples": len(rows_out),
        "regime_accuracy": acc_r,
        "predictability_accuracy": acc_p,
        "y_true_regime": y_true_r,
        "y_pred_regime": y_pred_r,
        "y_true_predictability": y_true_p,
        "y_pred_predictability": y_pred_p,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "Wrote %s (%s %s) regime_acc=%s pred_acc=%s",
        out_dir / "zero_shot_results.jsonl",
        backend,
        reported_model,
        acc_r,
        acc_p,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
