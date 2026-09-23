"""Auditable conversions and dataset inspection for Laya training decisions."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from typing import Any


def level_to_score_index(level: int) -> int:
    """Map an ordinal prosody level in ``[-3, 3]`` to Laya's ``[0, 6]`` index."""

    if isinstance(level, bool) or not isinstance(level, int) or not -3 <= level <= 3:
        raise ValueError("prosody level must be an integer in [-3, 3]")
    return level + 3


def score_index_to_level(score: float) -> int:
    """Map Laya's expected score in ``[0, 6]`` to a rounded prosody level."""

    value = float(score)
    if not math.isfinite(value):
        raise ValueError("score index must be finite")
    return max(-3, min(3, round(value) - 3))


def _to_internal(question: dict[str, Any]) -> dict[str, Any]:
    qtype = question["type"]
    criteria = question.get("criteria")
    if qtype == "choice" and isinstance(criteria, list):
        criteria = {value: None for value in criteria}
    instructions = question["instructions"]
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return {"t": qtype, "ins": instructions, "crit": criteria}


def _ordered_probabilities(
    question: dict[str, Any], gold_question: dict[str, Any]
) -> tuple[list[str], list[float]]:
    qtype = question["type"]
    criteria = question.get("criteria")
    probabilities = gold_question.get("probabilities", {})
    if qtype == "choice":
        keys = list(criteria if isinstance(criteria, dict) else criteria or [])
    elif qtype == "score":
        keys = [str(index) for index in range(len(criteria or []))]
    elif qtype == "noul":
        keys = ["false", "true"]
    else:
        raise ValueError(f"unsupported Laya question type: {qtype!r}")
    values = [float(probabilities.get(key, 0.0)) for key in keys]
    if not values or sum(values) <= 0:
        raise ValueError("gold probabilities must have positive mass")
    return keys, values


def audit_training_rows(
    rows: Iterable[dict[str, Any]],
    agent: Any,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Render training decisions and expose the exact label/token marker contract."""

    from laya.common import build_sequence, render_options

    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")

    cfg = agent.cfg
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        state = row["state"]
        questions = row["questions"]
        gold = row["gold"]
        for qid, question in questions.items():
            if qid not in gold:
                raise ValueError(f"question {qid!r} has no gold target")
            internal = _to_internal(question)
            option_keys, probabilities = _ordered_probabilities(question, gold[qid])
            ids, markers = build_sequence(
                agent.tok,
                state,
                internal,
                cfg.get("max_len", 1024),
                cfg.get("head_max_len", 256),
            )
            expected = len(render_options(internal))
            if len(markers) != expected:
                raise ValueError(
                    f"question {qid!r} marker/option mismatch: "
                    f"got {len(markers)}, expected {expected}"
                )
            marker_tokens = None
            if hasattr(agent.tok, "convert_ids_to_tokens"):
                marker_tokens = agent.tok.convert_ids_to_tokens([ids[pos] for pos in markers])
            records.append(
                {
                    "row_index": row_index,
                    "state": state,
                    "qid": qid,
                    "qtype": question["type"],
                    "criteria": question.get("criteria"),
                    "gold_probabilities": dict(gold[qid].get("probabilities", {})),
                    "option_keys": option_keys,
                    "label": probabilities.index(max(probabilities)),
                    "ids": ids,
                    "markers": markers,
                    "marker_tokens": marker_tokens,
                    "marker_count": len(markers),
                    "expected_option_count": expected,
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records
