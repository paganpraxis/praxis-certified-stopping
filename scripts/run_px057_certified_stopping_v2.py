#!/usr/bin/env python
"""Collect matched rolling and context-reset traces for PX-057 v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_px057_certified_stopping_v2 import normalize_numeric


def rolling_prompt(question: str, previous: str | None, round_index: int) -> str:
    if previous is None:
        return first_round_prompt(question)
    return (
        "Reconsider the problem independently. Check the previous proposed solution "
        "for arithmetic or reasoning mistakes. You may keep or change the answer. "
        "End with exactly 'Final answer: <number>'.\n\nProblem: "
        f"{question}\n\nPrevious proposed solution:\n{previous}\n\n"
        f"Reconsideration round: {round_index}"
    )


def first_round_prompt(question: str) -> str:
    return (
        "Solve the arithmetic word problem carefully. End with exactly "
        f"'Final answer: <number>'.\n\nProblem: {question}"
    )


def load_dataset(config: dict[str, Any], output_dir: Path) -> list[dict[str, str]]:
    import requests

    response = requests.get(config["dataset_url"], timeout=60)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != config["dataset_sha256"]:
        raise ValueError(f"dataset hash mismatch: {digest}")
    source = output_dir / "dataset_source.jsonl"
    source.write_bytes(response.content)
    all_rows = [json.loads(line) for line in response.text.splitlines() if line]
    rng = random.Random(int(config["sample_seed"]))
    indices = sorted(rng.sample(range(len(all_rows)), int(config["sample_size"])))
    selected = []
    for index in indices:
        answer = all_rows[index]["answer"].rsplit("####", 1)
        if len(answer) != 2:
            raise ValueError(f"missing GSM8K marker at source row {index}")
        selected.append(
            {
                "question_id": f"gsm8k-test-{index}",
                "question": all_rows[index]["question"],
                "gold_answer": normalize_numeric(answer[1]),
                "source_index": index,
            }
        )
    return selected


def load_backend(model_id: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, device_map="auto", torch_dtype="auto"
    )
    model.eval()
    return torch, tokenizer, model


def generate(torch, tokenizer, model, prompt: str, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
    generated = output.sequences[0][prompt_tokens:]
    return (
        tokenizer.decode(generated, skip_special_tokens=True),
        prompt_tokens,
        int(len(generated)),
    )


def collect(config: dict[str, Any]) -> Path:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_dataset(config, output_dir)
    (output_dir / "selected_rows.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    torch, tokenizer, model = load_backend(config["model_id"], config["model_revision"])
    raw_path = output_dir / "raw_generations.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            for arm in ("rolling", "context_reset"):
                previous = None
                for round_index in range(1, int(config["rounds"]) + 1):
                    prompt = (
                        rolling_prompt(row["question"], previous, round_index)
                        if arm == "rolling"
                        else first_round_prompt(row["question"])
                    )
                    response, prompt_tokens, generated_tokens = generate(
                        torch,
                        tokenizer,
                        model,
                        prompt,
                        int(config["max_new_tokens"]),
                    )
                    handle.write(
                        json.dumps(
                            {
                                "question_id": row["question_id"],
                                "arm": arm,
                                "round": round_index,
                                "prompt": prompt,
                                "response": response,
                                "gold_answer": row["gold_answer"],
                                "prompt_tokens": prompt_tokens,
                                "generated_tokens": generated_tokens,
                                "total_tokens": prompt_tokens + generated_tokens,
                            }
                        )
                        + "\n"
                    )
                    if arm == "rolling":
                        previous = response
    return raw_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(collect(config))


if __name__ == "__main__":
    main()
