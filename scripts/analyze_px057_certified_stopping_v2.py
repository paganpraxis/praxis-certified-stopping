#!/usr/bin/env python
"""Independent paired certification and protocol audit for PX-057 v2.

The analyzer deliberately does not trust stored ``correct`` flags or the old
Gate 2 summary.  It reconstructs correctness from the archived raw generations
and gold answers, replays the frozen stopping policy, and reports one-sided
finite-sample certificates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_numeric(value: str) -> str:
    value = value.replace(",", "").strip()
    try:
        number = float(value)
    except ValueError:
        return value.lower()
    return str(int(number)) if number.is_integer() else f"{number:.10g}"


def extraction_candidates(text: str) -> dict[str, str]:
    explicit = re.findall(
        r"(?:final answer|answer)\s*(?:is|:|=)\s*"
        r"([-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
        text,
        flags=re.I,
    )
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    boxed_numbers = NUMBER_RE.findall(boxed[-1]) if boxed else []
    all_numbers = NUMBER_RE.findall(text)
    return {
        "strict_explicit": normalize_numeric(explicit[-1]) if explicit else "",
        "boxed": normalize_numeric(boxed_numbers[-1]) if boxed_numbers else "",
        "fallback_last_number": normalize_numeric(all_numbers[-1]) if all_numbers else "",
    }


def resolved_answer(
    text: str, precedence: Iterable[str] = (
        "boxed",
        "strict_explicit",
        "fallback_last_number",
    )
) -> tuple[str, str]:
    candidates = extraction_candidates(text)
    for method in precedence:
        if method not in candidates:
            raise ValueError(f"unknown extraction method: {method}")
        if candidates[method]:
            return candidates[method], method
    return "", "empty"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    term = (1.0 - probability) ** n
    total = term
    ratio = probability / (1.0 - probability)
    for index in range(k):
        term *= (n - index) / (index + 1) * ratio
        total += term
    return min(1.0, max(0.0, total))


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """One-sided exact upper confidence limit for a binomial proportion."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("successes must be between zero and positive trials")
    if successes == trials:
        return 1.0
    low, high = successes / trials, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if binomial_cdf(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    if successes == 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(trials - successes, trials, alpha)


def select_stability_stop(
    rounds: list[dict[str, Any]], *, min_round: int, patience: int
) -> dict[str, Any]:
    for index, current in enumerate(rounds):
        if current["round"] < min_round or index + 1 < patience:
            continue
        window = rounds[index + 1 - patience : index + 1]
        if len({row["answer"] for row in window}) == 1:
            return current
    return rounds[-1]


def paired_table(
    first: Iterable[bool], second: Iterable[bool]
) -> dict[str, int]:
    counts = Counter(zip(first, second))
    return {
        "both_wrong": counts[(False, False)],
        "first_wrong_second_correct": counts[(False, True)],
        "first_correct_second_wrong": counts[(True, False)],
        "both_correct": counts[(True, True)],
    }


def paired_net_effect_certificate(
    table: dict[str, int], *, trials: int, alpha: float
) -> dict[str, float | int]:
    rescues = table["first_correct_second_wrong"]
    losses = table["first_wrong_second_correct"]
    discordant = rescues + losses
    point = (rescues - losses) / trials
    if discordant == 0:
        lower = 0.0
    else:
        rescue_share_lower = clopper_pearson_lower(rescues, discordant, alpha)
        lower = discordant / trials * (2.0 * rescue_share_lower - 1.0)
    return {
        "discordant_pairs": discordant,
        "rescues": rescues,
        "losses": losses,
        "point": point,
        "lower_confidence_bound": lower,
    }


def audit_raw_generations(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    method_counts: Counter[str] = Counter()
    disagreements = 0
    precedence = config.get("extraction", {}).get(
        "primary_precedence",
        ["boxed", "strict_explicit", "fallback_last_number"],
    )
    for raw in rows:
        candidates = extraction_candidates(str(raw["response"]))
        answer, method = resolved_answer(str(raw["response"]), precedence)
        nonempty = {value for value in candidates.values() if value}
        disagreements += len(nonempty) > 1
        method_counts[method] += 1
        arm = str(raw.get("arm", "rolling"))
        grouped[(arm, str(raw["question_id"]))].append(
            {
                "round": int(raw["round"]),
                "answer": answer,
                "gold": normalize_numeric(str(raw["gold_answer"])),
                "correct": answer == normalize_numeric(str(raw["gold_answer"])),
                "generated_tokens": int(raw.get("generated_tokens", 0)),
                "prompt_tokens": raw.get("prompt_tokens"),
            }
        )

    expected_rounds = int(config["protocol"]["rounds"])
    traces_by_arm: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (arm, question_id), rounds in grouped.items():
        rounds.sort(key=lambda row: row["round"])
        if [row["round"] for row in rounds] != list(range(1, expected_rounds + 1)):
            raise ValueError(f"incomplete rounds for {question_id}")
        traces_by_arm[arm].append((question_id, rounds))
    for traces_for_arm in traces_by_arm.values():
        traces_for_arm.sort()
    traces = traces_by_arm["rolling"]
    if not traces:
        raise ValueError("no rolling-arm traces found")

    n = len(traces)
    policy = config["policy"]
    stopped = [
        select_stability_stop(
            rounds,
            min_round=int(policy["min_round"]),
            patience=int(policy["patience"]),
        )
        for _, rounds in traces
    ]
    full = [rounds[-1] for _, rounds in traces]
    reference_round = int(config["protocol_admissibility"]["reference_round"])
    reference = [rounds[reference_round - 1] for _, rounds in traces]

    stopped_vs_full = paired_table(
        (row["correct"] for row in stopped), (row["correct"] for row in full)
    )
    reference_vs_full = paired_table(
        (row["correct"] for row in reference), (row["correct"] for row in full)
    )
    harm_count = stopped_vs_full["first_wrong_second_correct"]
    alpha = float(config["inference"]["alpha"])
    joint_alpha = alpha / 2.0
    harm_upper = clopper_pearson_upper(harm_count, n, alpha)
    joint_harm_upper = clopper_pearson_upper(harm_count, n, joint_alpha)
    net = paired_net_effect_certificate(stopped_vs_full, trials=n, alpha=alpha)
    joint_net = paired_net_effect_certificate(
        stopped_vs_full, trials=n, alpha=joint_alpha
    )
    admissibility = paired_net_effect_certificate(
        reference_vs_full, trials=n, alpha=alpha
    )

    generated_stopped = sum(
        sum(row["generated_tokens"] for row in rounds[: stop["round"]])
        for (_, rounds), stop in zip(traces, stopped)
    )
    generated_full = sum(
        sum(row["generated_tokens"] for row in rounds) for _, rounds in traces
    )
    prompt_tokens_available = all(
        row["prompt_tokens"] is not None for _, rounds in traces for row in rounds
    )
    billed_saving = None
    if prompt_tokens_available:
        billed_stopped = sum(
            sum(
                int(row["prompt_tokens"]) + row["generated_tokens"]
                for row in rounds[: stop["round"]]
            )
            for (_, rounds), stop in zip(traces, stopped)
        )
        billed_full = sum(
            sum(int(row["prompt_tokens"]) + row["generated_tokens"] for row in rounds)
            for _, rounds in traces
        )
        billed_saving = 1.0 - billed_stopped / billed_full

    harm_ceiling = float(config["certificates"]["harm_ceiling"])
    net_floor = float(config["certificates"]["net_effect_floor"])
    margin = float(config["protocol_admissibility"]["margin"])
    reset_result = None
    reset_traces = traces_by_arm.get("context_reset", [])
    if reset_traces:
        if [item[0] for item in reset_traces] != [item[0] for item in traces]:
            raise ValueError("rolling and context-reset item IDs do not match")
        reset_final = [rounds[-1] for _, rounds in reset_traces]
        reset_vs_rolling = paired_table(
            (row["correct"] for row in reset_final),
            (row["correct"] for row in full),
        )
        reset_result = {
            "accuracy": sum(row["correct"] for row in reset_final) / n,
            "paired_table_vs_rolling_full": reset_vs_rolling,
            "within_reference_tolerance": abs(
                sum(row["correct"] for row in reset_final) / n
                - sum(row["correct"] for row in reference) / n
            )
            <= float(config.get("causal_split", {}).get("reference_tolerance", 0.03)),
        }
    return {
        "n_items": n,
        "policy": policy,
        "accuracy": {
            "stopped": sum(row["correct"] for row in stopped) / n,
            "full": sum(row["correct"] for row in full) / n,
            "reference_round": sum(row["correct"] for row in reference) / n,
            "by_round": [
                sum(rounds[index]["correct"] for _, rounds in traces) / n
                for index in range(expected_rounds)
            ],
        },
        "stopped_vs_full_table": stopped_vs_full,
        "reference_vs_full_table": reference_vs_full,
        "stopping_harm_certificate": {
            "count": harm_count,
            "point": harm_count / n,
            "upper_confidence_bound": harm_upper,
            "ceiling": harm_ceiling,
            "passes": harm_upper <= harm_ceiling,
        },
        "paired_net_effect_certificate": {
            **net,
            "floor": net_floor,
            "passes": net["lower_confidence_bound"] >= net_floor,
        },
        "dual_certificate": {
            "shared_error_budget": alpha,
            "per_component_alpha": joint_alpha,
            "harm_upper_confidence_bound": joint_harm_upper,
            "net_effect_lower_confidence_bound": joint_net[
                "lower_confidence_bound"
            ],
            "passes": joint_harm_upper <= harm_ceiling
            and joint_net["lower_confidence_bound"] >= net_floor,
        },
        "protocol_admissibility": {
            **admissibility,
            "reference_round": reference_round,
            "full_round": expected_rounds,
            "margin": margin,
            "protocol_inadmissible": admissibility["lower_confidence_bound"] > margin,
            "causal_interpretation": "pending_context_reset_arm",
        },
        "context_reset_causal_split": reset_result,
        "extraction_audit": {
            "primary_precedence": precedence,
            "method_counts": dict(method_counts),
            "template_disagreements": disagreements,
            "requires_manual_review": disagreements + method_counts["empty"],
        },
        "cost_accounting": {
            "generated_token_saving": 1.0 - generated_stopped / generated_full,
            "prompt_tokens_available": prompt_tokens_available,
            "billed_token_saving": billed_saving,
            "claim_allowed": (
                "generated-token saving only"
                if not prompt_tokens_available
                else "billed-token saving may be computed in prospective run"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit_raw_generations(read_jsonl(args.raw_generations), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
