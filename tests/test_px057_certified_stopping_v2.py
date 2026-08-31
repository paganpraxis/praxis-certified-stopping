import pytest

from scripts.analyze_px057_certified_stopping_v2 import (
    audit_raw_generations,
    clopper_pearson_upper,
    extraction_candidates,
    paired_net_effect_certificate,
    paired_table,
)
from scripts.run_px057_certified_stopping_v2 import (
    first_round_prompt,
    rolling_prompt,
)


def test_exact_harm_bound_shows_original_point_gate_does_not_certify() -> None:
    upper = clopper_pearson_upper(1, 200, 0.05)
    assert upper == pytest.approx(0.02347, abs=0.00002)
    assert upper > 0.02


def test_known_px057_paired_tables() -> None:
    adaptive = [True] * 122 + [True] * 60 + [False] + [False] * 17
    full = [True] * 122 + [False] * 60 + [True] + [False] * 17
    assert paired_table(adaptive, full) == {
        "both_wrong": 17,
        "first_wrong_second_correct": 1,
        "first_correct_second_wrong": 60,
        "both_correct": 122,
    }
    certificate = paired_net_effect_certificate(
        paired_table(adaptive, full), trials=200, alpha=0.05
    )
    assert certificate["point"] == pytest.approx(0.295)
    assert certificate["lower_confidence_bound"] > 0.25


def test_extraction_audit_detects_template_disagreement() -> None:
    candidates = extraction_candidates(
        r"A stray 7 appears. Therefore \boxed{12}. Final answer: 13"
    )
    assert candidates == {
        "strict_explicit": "13",
        "boxed": "12",
        "fallback_last_number": "13",
    }


def test_protocol_prompts_do_not_accumulate_transcript() -> None:
    first = first_round_prompt("What is 2+2?")
    rolling = rolling_prompt("What is 2+2?", "Final answer: 5", 3)
    reset = first_round_prompt("What is 2+2?")
    assert "Final answer: 5" in rolling
    assert "Final answer: 5" not in reset
    assert reset == first


def test_end_to_end_recomputes_correctness_and_certificates() -> None:
    rows = []
    answers = {
        "rescue": ["1", "1", "2"],
        "harm": ["2", "2", "1"],
        "both_correct": ["1", "1", "1"],
        "both_wrong": ["2", "2", "2"],
    }
    for question_id, sequence in answers.items():
        for round_index, answer in enumerate(sequence, 1):
            rows.append(
                {
                    "question_id": question_id,
                    "round": round_index,
                    "response": f"Final answer: {answer}",
                    "extracted_answer": "untrusted",
                    "gold_answer": "1",
                    "correct": False,
                    "generated_tokens": 10,
                }
            )
    config = {
        "protocol": {"rounds": 3},
        "policy": {"min_round": 2, "patience": 2},
        "inference": {"alpha": 0.05},
        "certificates": {"harm_ceiling": 0.8, "net_effect_floor": -0.8},
        "protocol_admissibility": {"reference_round": 2, "margin": 0.05},
    }
    result = audit_raw_generations(rows, config)
    assert result["stopped_vs_full_table"] == {
        "both_wrong": 1,
        "first_wrong_second_correct": 1,
        "first_correct_second_wrong": 1,
        "both_correct": 1,
    }
    assert result["extraction_audit"]["method_counts"] == {"strict_explicit": 12}
    assert result["cost_accounting"]["billed_token_saving"] is None
