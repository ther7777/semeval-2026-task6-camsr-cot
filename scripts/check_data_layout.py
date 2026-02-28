"""
用途：检查数据 JSONL 的字段契约是否满足 CAMSR-CoT pipeline 的最小要求，并打印数据签名。

输入：
- --data_path：JSONL 文件路径（每行一个样本）

输出：
- 终端打印：文件大小、sha256、前若干行字段缺失情况

运行示例（Windows / Linux 单行）：
  python scripts/check_data_layout.py --data_path your/path/to/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from camsr_cot.io import sha256_file  # noqa: E402


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="数据 JSONL 路径")
    p.add_argument("--max_lines", type=int, default=50, help="最多检查前 N 行（默认 50）")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    data_path = Path(args.data_path).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    size = data_path.stat().st_size
    h = sha256_file(data_path)
    print(f"data_path: {data_path}")
    print(f"bytes: {size}")
    print(f"sha256: {h}")

    required = ["interview_question", "interview_answer", "question"]
    optional_id = ["index", "id"]

    missing_counts: Dict[str, int] = {k: 0 for k in required}
    missing_id = 0
    checked = 0

    with open(data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            checked += 1
            for k in required:
                if k not in obj or obj.get(k) is None or str(obj.get(k)).strip() == "":
                    missing_counts[k] += 1
            if not any((kid in obj and obj.get(kid) is not None) for kid in optional_id):
                missing_id += 1
            if checked >= int(args.max_lines):
                break

    print(f"checked_lines: {checked}")
    for k in required:
        print(f"missing[{k}]: {missing_counts[k]}")
    print(f"missing[id/index]: {missing_id}")

    if checked == 0:
        print("WARNING: no valid JSON lines found.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

