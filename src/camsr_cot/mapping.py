from __future__ import annotations

from typing import Dict


# Task2 -> Task1 映射（论文中固定的 deterministic mapping）
TASK2_TO_TASK1: Dict[str, str] = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Dodging": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}


def task2_to_task1(task2_label: str, *, mapping: Dict[str, str] | None = None) -> str:
    m = mapping or TASK2_TO_TASK1
    if task2_label not in m:
        # 防御式兜底：未知标签按 Ambivalent 处理，避免脚本直接崩溃
        return "Ambivalent"
    return m[task2_label]

