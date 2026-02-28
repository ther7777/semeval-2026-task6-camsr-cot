"""
用途：用本地 gold 对 assets/reported_runs/ 的“论文报告 run（脱敏预测）”做口径校准。

说明：
- 默认只打印“计算值 vs 期望值（0.811759 / 0.617374）”与差异，不强制失败；
- 加 --strict 时才做精确断言（用于 CI 或强一致场景）。

输入：
- --gold：dev gold JSONL（需要包含 clarity_label 与 annotator1/2/3 或等价字段）

输出：
- 终端打印 Task1/Task2 macro-F1、期望值、差异

运行示例（Windows / Linux 单行）：
  python scripts/verify_reported_dev.py --gold your/path/to/dev.jsonl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_TASK1 = 0.811759
EXPECTED_TASK2 = 0.617374


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True, help="dev gold JSONL 路径")
    p.add_argument("--strict", action="store_true", help="严格断言 macro-F1 必须精确匹配")
    return p.parse_args(argv)


def _run_eval(task: str, gold_path: Path, pred_jsonl_path: Path) -> Tuple[float, str]:
    cmd = [
        sys.executable,
        str((_REPO_ROOT / "scripts" / "eval_competition.py").resolve()),
        "--task",
        task,
        "--gold",
        str(gold_path),
        "--pred_jsonl",
        str(pred_jsonl_path),
        "--no_record",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"Eval failed (task={task}). stderr:\n{err}")

    macro_f1 = None
    for line in out.splitlines():
        if line.startswith("macro_f1\t"):
            try:
                macro_f1 = float(line.split("\t", 1)[1].strip())
            except Exception:
                macro_f1 = None
            break
    if macro_f1 is None:
        raise RuntimeError(f"Cannot parse macro_f1 from output (task={task}).\n{out}")
    return float(macro_f1), out


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    gold_path = Path(args.gold).expanduser()
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    pred_path = (_REPO_ROOT / "assets" / "reported_runs" / "dev_predictions.jsonl").resolve()
    if not pred_path.exists():
        raise FileNotFoundError(f"Reported predictions not found: {pred_path}")

    t1, _ = _run_eval("task1", gold_path, pred_path)
    t2, _ = _run_eval("task2", gold_path, pred_path)

    d1 = t1 - EXPECTED_TASK1
    d2 = t2 - EXPECTED_TASK2

    print(f"Task1 macro_f1: {t1:.6f} | expected: {EXPECTED_TASK1:.6f} | diff: {d1:+.6f}")
    print(f"Task2 macro_f1: {t2:.6f} | expected: {EXPECTED_TASK2:.6f} | diff: {d2:+.6f}")

    if args.strict and (abs(d1) > 1e-12 or abs(d2) > 1e-12):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
