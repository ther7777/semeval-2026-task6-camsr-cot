"""
用途：从开发用 predictions.jsonl 导出 Codabench 提交文件（prediction / prediction.zip）。

输入（JSONL）：每行一个 JSON object，通常由 pipeline 生成。期望字段：
- id / index：样本 id（默认优先使用 index；缺失时回退到 id）
- task1_prediction / task2_prediction：预测标签

输出：
- <output_dir>/prediction：每行一个标签
- <output_dir>/prediction.zip（可选）：zip 根目录仅包含 prediction 文件

运行示例（Windows / Linux 单行）：
  python scripts/make_submission.py --task task2 --input outputs/2026-02-26_120000_camsr_cot_dev/dev/predictions.jsonl --output_dir outputs/2026-02-26_120000_camsr_cot_dev/codabench_task2 --zip

说明：
- 若你有官方 test/evaltest JSONL（用于确定导出顺序），可传 --test_file；
  否则默认使用输入 JSONL 的行顺序（或 --order_by index 时按 id 升序）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Args:
    task: str
    input_path: str
    output_dir: str
    test_file: str
    id_key: str
    prediction_key: Optional[str]
    order_by: str
    zip_output: bool


def _iter_jsonl(path: str) -> Iterable[Tuple[int, dict]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}, got {type(obj).__name__}")
            yield line_number, obj


def _coerce_int(value, *, context: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Expected int-like id at {context}, got {value!r}")


def _read_id(obj: dict, *, id_key: str, context: str) -> int:
    if id_key in obj:
        raw_id = obj[id_key]
    elif id_key == "index" and "id" in obj:
        raw_id = obj["id"]
    elif id_key == "id" and "index" in obj:
        raw_id = obj["index"]
    else:
        raise ValueError(f"Missing '{id_key}' at {context}")
    return _coerce_int(raw_id, context=f"{context}:{id_key}")


def _load_predictions_by_id(input_path: str, *, id_key: str, prediction_key: str) -> Dict[int, str]:
    predictions_by_id: Dict[int, str] = {}
    for line_number, obj in _iter_jsonl(input_path):
        context = f"{input_path}:{line_number}"
        sample_id = _read_id(obj, id_key=id_key, context=context)

        if prediction_key not in obj:
            raise ValueError(f"Missing '{prediction_key}' at {context}")
        prediction = obj[prediction_key]
        if prediction is None or str(prediction).strip() == "":
            raise ValueError(f"Empty '{prediction_key}' at {context}")
        prediction_str = str(prediction).strip()

        if sample_id in predictions_by_id:
            raise ValueError(f"Duplicate id={sample_id} found in {input_path} (at line {line_number})")
        predictions_by_id[sample_id] = prediction_str

    if not predictions_by_id:
        raise ValueError(f"No predictions found in {input_path}")
    return predictions_by_id


def _load_predictions_in_order(input_path: str, *, id_key: str, prediction_key: str) -> List[Tuple[int, str]]:
    ordered: List[Tuple[int, str]] = []
    seen: set[int] = set()

    for line_number, obj in _iter_jsonl(input_path):
        context = f"{input_path}:{line_number}"
        sample_id = _read_id(obj, id_key=id_key, context=context)
        if sample_id in seen:
            raise ValueError(f"Duplicate id={sample_id} found in {context}")
        seen.add(sample_id)

        if prediction_key not in obj:
            raise ValueError(f"Missing '{prediction_key}' at {context}")
        prediction = obj[prediction_key]
        if prediction is None or str(prediction).strip() == "":
            raise ValueError(f"Empty '{prediction_key}' at {context}")
        ordered.append((sample_id, str(prediction).strip()))

    if not ordered:
        raise ValueError(f"No predictions found in {input_path}")
    return ordered


def _load_test_order(test_file: str, *, order_by: str) -> List[int]:
    ordered_ids: List[int] = []
    for line_number, obj in _iter_jsonl(test_file):
        context = f"{test_file}:{line_number}"
        if "index" in obj and obj["index"] is not None:
            sample_id = _coerce_int(obj["index"], context=f"{context}:index")
        else:
            sample_id = line_number - 1
        ordered_ids.append(sample_id)

    if not ordered_ids:
        raise ValueError(f"No rows found in test file {test_file}")

    if order_by not in {"keep", "index"}:
        raise ValueError(f"Unknown order_by: {order_by!r}")
    if order_by == "index":
        ordered_ids = sorted(ordered_ids)
    return ordered_ids


def _write_prediction_file(labels: List[str], *, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "prediction")
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(f"{label}\n")
    return output_path


def _zip_prediction_file(prediction_path: str) -> str:
    zip_path = prediction_path + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        zip_handle.write(prediction_path, arcname="prediction")
    return zip_path


def _parse_args(argv: List[str]) -> Args:
    parser = argparse.ArgumentParser(description="从 predictions.jsonl 生成 Codabench 提交文件")
    parser.add_argument("--task", required=True, choices=["task1", "task2"], help="导出哪个任务的 prediction 文件")
    parser.add_argument("--input", required=True, dest="input_path", help="predictions.jsonl 路径（JSONL，每行一个 object）")
    parser.add_argument("--output_dir", required=True, help="输出目录（写入 prediction 与可选的 prediction.zip）")
    parser.add_argument(
        "--test_file",
        default="",
        help="可选：官方 test/evaltest JSONL（用于确定导出顺序）。不提供则使用输入 JSONL 行顺序（或 --order_by index 时按 id 升序）。",
    )
    parser.add_argument(
        "--id_key",
        default="index",
        help="输入 JSONL 中样本 id 的字段名（默认 index；缺失时会回退到 id）。",
    )
    parser.add_argument("--prediction_key", default=None, help="预测字段名；默认随 --task 使用 task1_prediction/task2_prediction。")
    parser.add_argument("--order_by", default="keep", choices=["keep", "index"], help="导出顺序：keep=保持原顺序；index=按 id 升序")
    parser.add_argument("--zip", action="store_true", dest="zip_output", help="同时生成 prediction.zip")
    ns = parser.parse_args(argv)
    return Args(
        task=ns.task,
        input_path=ns.input_path,
        output_dir=ns.output_dir,
        test_file=ns.test_file,
        id_key=ns.id_key,
        prediction_key=ns.prediction_key,
        order_by=ns.order_by,
        zip_output=ns.zip_output,
    )


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    prediction_key = args.prediction_key or ("task1_prediction" if args.task == "task1" else "task2_prediction")

    if args.test_file:
        ordered_ids = _load_test_order(args.test_file, order_by=args.order_by)
        predictions_by_id = _load_predictions_by_id(args.input_path, id_key=args.id_key, prediction_key=prediction_key)

        missing_ids = [sample_id for sample_id in ordered_ids if sample_id not in predictions_by_id]
        if missing_ids:
            preview = ", ".join(map(str, missing_ids[:20]))
            suffix = "" if len(missing_ids) <= 20 else f" ... (+{len(missing_ids) - 20} more)"
            raise ValueError(f"Missing predictions for {len(missing_ids)} ids from test order: {preview}{suffix}")

        labels = [predictions_by_id[sample_id] for sample_id in ordered_ids]
    else:
        ordered = _load_predictions_in_order(args.input_path, id_key=args.id_key, prediction_key=prediction_key)
        if args.order_by == "index":
            ordered = sorted(ordered, key=lambda t: t[0])
        labels = [lab for _, lab in ordered]

    prediction_path = _write_prediction_file(labels, output_dir=args.output_dir)
    print(f"Wrote {len(labels)} lines to {prediction_path}")

    if args.zip_output:
        zip_path = _zip_prediction_file(prediction_path)
        print("Wrote {} (zip root contains only 'prediction')".format(zip_path))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

