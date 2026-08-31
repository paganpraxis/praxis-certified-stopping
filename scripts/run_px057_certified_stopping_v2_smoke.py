#!/usr/bin/env python
"""Run the deterministic end-to-end harness smoke for PX-057 v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_px057_certified_stopping_v2 import audit_raw_generations
from scripts.run_px057_certified_stopping_v2 import first_round_prompt, rolling_prompt


SEQUENCES = {
    "rescue": ["1", "1", "1", "2", "2", "2", "2", "2"],
    "harm": ["2", "2", "2", "1", "1", "1", "1", "1"],
    "both_correct": ["1"] * 8,
    "both_wrong": ["2"] * 8,
}


def build_fixture() -> list[dict]:
    rows = []
    for question_id, rolling_answers in SEQUENCES.items():
        previous = None
        for arm in ("rolling", "context_reset"):
            for round_index in range(1, 9):
                prompt = (
                    rolling_prompt("Fixture: what value is correct?", previous, round_index)
                    if arm == "rolling"
                    else first_round_prompt("Fixture: what value is correct?")
                )
                answer = rolling_answers[round_index - 1] if arm == "rolling" else "1"
                if question_id == "both_wrong" and arm == "context_reset":
                    answer = "2"
                response = f"Final answer: {answer}"
                rows.append(
                    {
                        "question_id": question_id,
                        "arm": arm,
                        "round": round_index,
                        "prompt": prompt,
                        "response": response,
                        "gold_answer": "1",
                        "prompt_tokens": len(prompt.split()),
                        "generated_tokens": len(response.split()),
                        "total_tokens": len(prompt.split()) + len(response.split()),
                    }
                )
                if arm == "rolling":
                    previous = response
            previous = None
    return rows


def validate(result: dict, rows: list[dict]) -> dict[str, bool]:
    rolling = [row for row in rows if row["arm"] == "rolling"]
    reset = [row for row in rows if row["arm"] == "context_reset"]
    checks = {
        "two_arms_present": len(rolling) == 32 and len(reset) == 32,
        "eight_rounds_per_item_arm": len(rows) == 4 * 2 * 8,
        "rolling_carries_previous_only": all(
            "Previous proposed solution" in row["prompt"]
            for row in rolling
            if row["round"] > 1
        ),
        "reset_has_no_previous_solution": all(
            "Previous proposed solution" not in row["prompt"] for row in reset
        ),
        "token_accounting_available": result["cost_accounting"][
            "prompt_tokens_available"
        ]
        and result["cost_accounting"]["billed_token_saving"] is not None,
        "paired_table_reconstructed": result["stopped_vs_full_table"]
        == {
            "both_wrong": 1,
            "first_wrong_second_correct": 1,
            "first_correct_second_wrong": 1,
            "both_correct": 1,
        },
        "context_reset_analyzed": result["context_reset_causal_split"] is not None,
        "extractors_exercised": result["extraction_audit"]["method_counts"]
        == {"strict_explicit": 64},
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = build_fixture()
    result = audit_raw_generations(rows, config)
    checks = validate(result, rows)
    report = {
        "experiment_id": config["experiment_id"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scientific_claim_allowed": False,
        "checks": checks,
        "analysis": result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_generations.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
