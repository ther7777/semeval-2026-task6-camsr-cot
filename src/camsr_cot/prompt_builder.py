from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


LABEL_TO_DIR = {
    "Declining to answer": "declining",
    "Claims ignorance": "claims_ignorance",
    "Clarification": "clarification",
    "OTHER": "ambivalent",
}


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _format_fewshot(item: Dict[str, Any], idx: int) -> str:
    label = str(item.get("label") or item.get("gold_label") or "").strip()
    fs_type = str(item.get("type") or "").strip().lower()
    if fs_type == "anchor":
        header = f"Example {idx} (Correct: {label})"
    else:
        confused_as = str(item.get("confused_as") or "").strip()
        header = f"Example {idx} (Correct: {label}, often confused as {confused_as})" if confused_as else f"Example {idx} (Correct: {label})"

    iq = str(item.get("interview_question") or "").strip()
    ia = str(item.get("interview_answer") or "").strip()
    sq = str(item.get("sub_question") or item.get("question") or "").strip()
    reasoning = str(item.get("reasoning") or "").strip()

    return (
        f"{header}:\n"
        f"Interview Question: {iq}\n"
        f"Interview Answer: {ia}\n"
        f"Sub-question: {sq}\n"
        f"Reasoning: {reasoning}\n"
        f"<label>{label}</label>\n"
    )


def build_gate_prompt(*, gate_template: str, sample: Dict[str, Any]) -> str:
    """构建 Step1 Gate Prompt（模板为 .txt，使用 format 占位符）。"""
    return gate_template.format(
        interview_question=sample.get("interview_question", ""),
        interview_answer=sample.get("interview_answer", ""),
        sub_question=sample.get("question", sample.get("sub_question", "")),
    )


def build_fewshots_block(
    *,
    gate_label: str,
    fewshots_dir: Path,
    num_anchors: int,
    num_boundaries: int,
) -> str:
    if int(num_anchors) <= 0 and int(num_boundaries) <= 0:
        return ""

    if gate_label not in LABEL_TO_DIR:
        return "### Examples ###\n(No specific examples for this case.)"

    label_dir = LABEL_TO_DIR[gate_label]
    base = fewshots_dir / label_dir
    anchors = _load_jsonl(base / "anchors.jsonl")[: int(num_anchors)]
    boundaries = _load_jsonl(base / "boundaries.jsonl")[: int(num_boundaries)]

    lines: List[str] = ["### Examples ###", ""]
    idx = 1
    for item in anchors:
        lines.append(_format_fewshot(item, idx))
        idx += 1
    for item in boundaries:
        lines.append(_format_fewshot(item, idx))
        idx += 1

    return "\n".join(lines).strip() + "\n"


def build_confusion_guidelines(*, gate_label: str, templates_dir: Path) -> str:
    if gate_label == "Declining to answer":
        path = templates_dir / "confusion_guide_declining.txt"
    elif gate_label == "Claims ignorance":
        path = templates_dir / "confusion_guide_claims.txt"
    elif gate_label == "Clarification":
        path = templates_dir / "confusion_guide_clarification.txt"
    elif gate_label == "OTHER":
        path = templates_dir / "confusion_guide_ambivalent.txt"
    else:
        return ""

    return _load_text(path) if path.exists() else ""


def build_prior_analysis_block(*, gate_label: str, gate_confidence: str, templates_dir: Path) -> str:
    path = templates_dir / "prior_analysis_block.txt"
    if not path.exists():
        return ""
    template = _load_text(path)
    return template.format(gate_label=gate_label, gate_confidence=gate_confidence).strip()


def build_stage_prompt(
    *,
    sample: Dict[str, Any],
    gate_label: str,
    gate_confidence: str,
    template_path: Path,
    definitions_path: Path,
    templates_dir: Path,
    fewshots_dir: Path,
    include_prior_analysis: bool,
    num_anchors: int,
    num_boundaries: int,
) -> str:
    """构建 Step2/Step3 的动态 Prompt。"""
    template = _load_text(template_path)
    definitions_block = _load_text(definitions_path) if definitions_path.exists() else ""

    prior_analysis_block = ""
    if include_prior_analysis:
        # Step2：仅对 Medium/Low 的 Non-Reply 做 prior analysis；Step3 通常关闭
        if gate_label in ("Declining to answer", "Claims ignorance", "Clarification") and gate_confidence in ("Medium", "Low"):
            prior_analysis_block = build_prior_analysis_block(
                gate_label=gate_label,
                gate_confidence=gate_confidence,
                templates_dir=templates_dir,
            )

    confusion_guidelines_block = build_confusion_guidelines(gate_label=gate_label, templates_dir=templates_dir)
    fewshots_block = build_fewshots_block(
        gate_label=gate_label,
        fewshots_dir=fewshots_dir,
        num_anchors=int(num_anchors),
        num_boundaries=int(num_boundaries),
    )

    prompt = template.format(
        prior_analysis_block=prior_analysis_block,
        definitions_block=definitions_block,
        confusion_guidelines_block=confusion_guidelines_block,
        fewshots_block=fewshots_block,
        interview_question=sample.get("interview_question", ""),
        interview_answer=sample.get("interview_answer", ""),
        sub_question=sample.get("question", sample.get("sub_question", "")),
    )
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip() + "\n"
    return prompt

