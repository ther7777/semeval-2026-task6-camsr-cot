"""
使用比赛口径的宏平均 F1（macro F1）评估预测结果。

支持两个任务：
- Task 1：在 clarity_label 上计算单标签 macro F1
- Task 2：与 Codabench scorer 一致的 multi-reference macro F1。
  样本 i 的参考集合 G_i = {annotator1, annotator2, annotator3}（去重），预测为 p_i。
  在每个类 c 的 one-vs-rest 计数中：
    - TP：p_i == c 且 c in G_i
    - FP：p_i == c 且 c not in G_i
    - FN：p_i != c 且 c in G_i 且 (p_i not in G_i)
  即：当 p_i 命中任一参考标签时，不会因为 G_i 还有其他参考标签而额外计 FN；但若 p_i 完全不命中 G_i，则 G_i 中的每个类都会产生 FN。

输入：
- gold：HF JSONL（train/test）
- preds：二选一
  - 提交格式文件 'prediction'（每行一个标签，顺序与 gold 一致），或
  - 开发用 JSONL，包含 id + task*_prediction（按 gold 的 'index' 对齐）

示例：
python scripts/eval/eval_competition.py --task task1 --gold HF/data/test-00000-of-00001.jsonl --pred prediction
python scripts/eval/eval_competition.py --task task2 --gold HF/data/test-00000-of-00001.jsonl --pred prediction
python scripts/eval/eval_competition.py --task task1 --gold HF/data/test-00000-of-00001.jsonl --pred_jsonl outputs/2026-01-07_direct_fs_9/predictions_dev.jsonl --per_class
python scripts/eval/eval_competition.py --task task2 --gold HF/data/test-00000-of-00001.jsonl --pred_jsonl outputs/2026-01-07_direct_fs_9/predictions_dev.jsonl --per_class
python scripts/eval/eval_competition.py --task task1 --gold HF/data/test-00000-of-00001.jsonl --pred_jsonl outputs/2026-01-08_exp2_cot_Non-Reply/predictions_dev.jsonl --per_class
python scripts/eval/eval_competition.py --task task1 --gold HF/data/test-00000-of-00001.jsonl --pred_jsonl outputs/2026-01-08_exp2_cot_Non-Reply/predictions_dev.jsonl --per_class
python scripts/eval/eval_competition.py --task task1 --gold HF/data/test-00000-of-00001.jsonl --pred_jsonl outputs/2026-01-08_exp3_exp2_ANCHORS_TOPIC_MATCH/predictions_dev.jsonl --per_class
python scripts/eval/eval_competition.py --task task2 --gold HF/data/test-00000-of-00001.jsonl --pred_jsonl outputs/2026-01-08_exp2_cot_Non-Reply/predictions_dev.jsonl --per_class
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


TASK1_LABELS: Tuple[str, ...] = ("Ambivalent", "Clear Non-Reply", "Clear Reply")
TASK2_LABELS: Tuple[str, ...] = (
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
)

@dataclass(frozen=True)
class Args:
    task: str
    gold_path: str
    pred_path: Optional[str]
    pred_jsonl_path: Optional[str]
    id_key: str
    order_by: str
    strict: bool
    per_class: bool
    no_record: bool


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _strip_trailing_punct(text: str) -> str:
    return text.rstrip(" \t\r\n.;:，,")


def _norm_key(text: str) -> str:
    return _normalize_spaces(_strip_trailing_punct(text)).lower()


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


def _normalize_task1(label: str) -> str:
    raw = str(label)
    key = _norm_key(raw)
    if key in {"ambivalent", "ambivalent reply"}:
        return "Ambivalent"
    if key in {"clear reply"}:
        return "Clear Reply"
    if key in {"clear non-reply", "clear nonreply", "clear non reply"}:
        return "Clear Non-Reply"
    return _normalize_spaces(_strip_trailing_punct(raw))


def _normalize_task2(label: str) -> str:
    raw = str(label).replace("\\/", "/")
    key = _norm_key(raw)
    canonical_by_key = {
        _norm_key(v): v for v in TASK2_LABELS
    }
    # 常见别名归一化（尽量保持精简；需要时再扩展）。
    aliases = {
        "partial/half answer": "Partial/half-answer",
        "partial / half-answer": "Partial/half-answer",
        "partial / half answer": "Partial/half-answer",
    }
    if key in canonical_by_key:
        return canonical_by_key[key]
    if key in aliases:
        return aliases[key]
    return _normalize_spaces(_strip_trailing_punct(raw))


def _load_gold(task: str, gold_path: str, *, order_by: str) -> Tuple[List[int], List[Set[str]], List[List[str]]]:
    ordered_ids: List[int] = []
    gold_sets: List[Set[str]] = []
    gold_refs: List[List[str]] = []

    for line_number, obj in _iter_jsonl(gold_path):
        sample_id = obj.get("index", line_number - 1)
        if not isinstance(sample_id, int):
            try:
                sample_id = int(str(sample_id).strip())
            except Exception as exc:
                raise ValueError(f"Bad gold index at {gold_path}:{line_number}") from exc

        if task == "task1":
            raw = obj.get("clarity_label")
            if raw is None or str(raw).strip() == "":
                raise ValueError(f"Missing clarity_label at {gold_path}:{line_number}")
            gold_label = _normalize_task1(str(raw))
            refs = [gold_label]
            gold_set = {gold_label}
        else:
            # 优先使用 evasion_label（训练集通常提供）。
            evasion = obj.get("evasion_label")
            if evasion is not None and str(evasion).strip() != "":
                ev = _normalize_task2(str(evasion))
                refs = [ev]
                gold_set = {ev}
            else:
                raw_refs = [
                    obj.get("annotator1"),
                    obj.get("annotator2"),
                    obj.get("annotator3"),
                ]
                refs = [_normalize_task2(str(r)) for r in raw_refs if r is not None and str(r).strip() != ""]
                gold_set = set(refs)
                if not gold_set:
                    # Fallbacks for pre-processed / non-HF gold formats.
                    raw_task2_labels = obj.get("task2_labels")
                    if isinstance(raw_task2_labels, list):
                        refs = [
                            _normalize_task2(str(r))
                            for r in raw_task2_labels
                            if r is not None and str(r).strip() != ""
                        ]
                        gold_set = set(refs)

                if not gold_set:
                    raw_label_set = obj.get("label_task2_set")
                    if isinstance(raw_label_set, list):
                        refs = [
                            _normalize_task2(str(r))
                            for r in raw_label_set
                            if r is not None and str(r).strip() != ""
                        ]
                        gold_set = set(refs)

                if not gold_set:
                    raw_label_refs = obj.get("label_task2_refs")
                    if isinstance(raw_label_refs, list):
                        refs = [
                            _normalize_task2(str(r))
                            for r in raw_label_refs
                            if r is not None and str(r).strip() != ""
                        ]
                        gold_set = set(refs)

                if not gold_set:
                    raw_label = obj.get("label_task2")
                    if raw_label is not None and str(raw_label).strip() != "":
                        ev = _normalize_task2(str(raw_label))
                        refs = [ev]
                        gold_set = {ev}

                if not gold_set:
                    raise ValueError(
                        "Missing task2 gold labels (expected evasion_label/annotator1-3/"
                        f"task2_labels/label_task2*) at {gold_path}:{line_number}"
                    )

        ordered_ids.append(sample_id)
        gold_sets.append(gold_set)
        gold_refs.append(refs)

    if not ordered_ids:
        raise ValueError(f"No rows found in gold file {gold_path}")

    if order_by not in {"keep", "index"}:
        raise ValueError(f"Unknown order_by: {order_by!r}")

    if order_by == "index":
        # 按 sample_id 升序重排所有数组，保证顺序稳定。
        zipped = list(zip(ordered_ids, gold_sets, gold_refs))
        zipped.sort(key=lambda t: t[0])
        ordered_ids = [t[0] for t in zipped]
        gold_sets = [t[1] for t in zipped]
        gold_refs = [t[2] for t in zipped]

    return ordered_ids, gold_sets, gold_refs


def _load_predictions_submission(pred_path: str, *, normalize) -> List[str]:
    labels: List[str] = []
    with open(pred_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            label = line.strip()
            if label == "":
                raise ValueError(f"Empty prediction at {pred_path}:{line_number}")
            labels.append(normalize(label))
    if not labels:
        raise ValueError(f"No predictions found in {pred_path}")
    return labels


def _load_predictions_jsonl(
    pred_jsonl_path: str,
    *,
    ordered_ids: Sequence[int],
    id_key: str,
    prediction_key: str,
    normalize,
) -> List[str]:
    predictions_by_id: Dict[int, str] = {}
    for line_number, obj in _iter_jsonl(pred_jsonl_path):
        if id_key in obj:
            sample_id = obj[id_key]
        elif id_key == "index" and "id" in obj:
            sample_id = obj["id"]
        elif id_key == "id" and "index" in obj:
            sample_id = obj["index"]
        else:
            raise ValueError(f"Missing '{id_key}' at {pred_jsonl_path}:{line_number}")

        if not isinstance(sample_id, int):
            try:
                sample_id = int(str(sample_id).strip())
            except Exception as exc:
                raise ValueError(f"Bad id at {pred_jsonl_path}:{line_number}:{id_key}") from exc

        if prediction_key not in obj:
            raise ValueError(f"Missing '{prediction_key}' at {pred_jsonl_path}:{line_number}")
        pred = obj[prediction_key]
        if pred is None or str(pred).strip() == "":
            raise ValueError(f"Empty '{prediction_key}' at {pred_jsonl_path}:{line_number}")
        pred_str = normalize(str(pred))

        if sample_id in predictions_by_id:
            raise ValueError(f"Duplicate id={sample_id} in {pred_jsonl_path} (line {line_number})")
        predictions_by_id[sample_id] = pred_str

    missing = [i for i in ordered_ids if i not in predictions_by_id]
    if missing:
        preview = ", ".join(map(str, missing[:20]))
        suffix = "" if len(missing) <= 20 else f" ... (+{len(missing) - 20} more)"
        raise ValueError(f"Missing predictions for {len(missing)} ids: {preview}{suffix}")

    return [predictions_by_id[i] for i in ordered_ids]


def _compute_macro_f1_multilabel_ovr(
    *,
    classes: Sequence[str],
    y_pred: Sequence[str],
    y_gold_sets: Sequence[Set[str]],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    if len(y_pred) != len(y_gold_sets):
        raise ValueError(f"Pred/gold length mismatch: {len(y_pred)} vs {len(y_gold_sets)}")

    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []

    for c in classes:
        tp = fp = fn = 0
        for pred, gold_set in zip(y_pred, y_gold_sets):
            pred_is_c = pred == c
            gold_has_c = c in gold_set
            if pred_is_c and gold_has_c:
                tp += 1
            elif pred_is_c and not gold_has_c:
                fp += 1
            elif (not pred_is_c) and gold_has_c:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        support = tp + fn

        per_class[c] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return macro_f1, per_class


def _compute_macro_f1_multiref_official(
    *,
    classes: Sequence[str],
    y_pred: Sequence[str],
    y_gold_sets: Sequence[Set[str]],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    if len(y_pred) != len(y_gold_sets):
        raise ValueError(f"Pred/gold length mismatch: {len(y_pred)} vs {len(y_gold_sets)}")

    # Codabench multi-reference scoring (Task2).
    # For each class c, one-vs-rest counts:
    #   TP: pred==c and c in gold_set
    #   FP: pred==c and c not in gold_set
    #   FN: pred!=c and c in gold_set and (pred not in gold_set)
    # i.e., if prediction hits ANY reference label, don't count FN for the other references.
    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []

    for c in classes:
        tp = fp = fn = 0
        for pred, gold_set in zip(y_pred, y_gold_sets):
            pred_hits_any_ref = pred in gold_set
            pred_is_c = pred == c
            gold_has_c = c in gold_set

            if pred_is_c and gold_has_c:
                tp += 1
            elif pred_is_c and not gold_has_c:
                fp += 1
            elif (not pred_is_c) and gold_has_c and (not pred_hits_any_ref):
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        support = tp + fn

        per_class[c] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return macro_f1, per_class


def _parse_args(argv: List[str]) -> Args:
    parser = argparse.ArgumentParser(description="按竞赛口径评估 macro F1（本地）")
    parser.add_argument("--task", required=True, choices=["task1", "task2"])
    parser.add_argument("--gold", required=True, dest="gold_path", help="gold 文件：HF JSONL（train/test）。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred", dest="pred_path", help="提交格式 prediction 文件（每行一个标签）。")
    group.add_argument("--pred_jsonl", dest="pred_jsonl_path", help="开发用 JSONL（包含 index/id + task*_prediction）。")
    parser.add_argument(
        "--id_key",
        default="index",
        help="--pred_jsonl 中样本 id 的字段名（默认 index；缺失时回退到 id）。",
    )
    parser.add_argument(
        "--order_by",
        default="keep",
        choices=["keep", "index"],
        help="对齐/评测的顺序（默认 keep：保持 gold 文件顺序；index：按 index 升序）。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="预测包含非官方标签时直接报错。",
    )
    parser.add_argument(
        "--per_class",
        action="store_true",
        help="打印每个类别的 TP/FP/FN/precision/recall/F1。",
    )
    parser.add_argument(
        "--no_record",
        action="store_true",
        help="不在预测文件同目录下追加 eval_results.jsonl 记录（默认会保存）。",
    )
    ns = parser.parse_args(argv)
    return Args(
        task=ns.task,
        gold_path=ns.gold_path,
        pred_path=ns.pred_path,
        pred_jsonl_path=ns.pred_jsonl_path,
        id_key=ns.id_key,
        order_by=ns.order_by,
        strict=ns.strict,
        per_class=ns.per_class,
        no_record=bool(ns.no_record),
    )


def _append_eval_record(*, record_dir: str, record: dict) -> str:
    os.makedirs(record_dir, exist_ok=True)
    path = os.path.join(record_dir, "eval_results.jsonl")
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        # 为了可读性：每条记录按结构缩进（多行），记录之间用空行分隔。
        handle.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")
    return path


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    if args.task == "task1":
        classes = TASK1_LABELS
        normalize = _normalize_task1
        prediction_key = "task1_prediction"
    else:
        classes = TASK2_LABELS
        normalize = _normalize_task2
        prediction_key = "task2_prediction"

    ordered_ids, gold_sets, gold_refs = _load_gold(args.task, args.gold_path, order_by=args.order_by)

    if args.pred_path:
        y_pred = _load_predictions_submission(args.pred_path, normalize=normalize)
        if len(y_pred) != len(gold_sets):
            raise ValueError(f"Expected {len(gold_sets)} lines in {args.pred_path}, got {len(y_pred)}")
    else:
        y_pred = _load_predictions_jsonl(
            args.pred_jsonl_path,  # type: ignore[arg-type]
            ordered_ids=ordered_ids,
            id_key=args.id_key,
            prediction_key=prediction_key,
            normalize=normalize,
        )

    if args.strict:
        unknown = sorted(set(y_pred) - set(classes))
        if unknown:
            raise ValueError(f"Unknown labels in predictions: {unknown}")

    if args.task == "task2":
        macro_f1, per_class = _compute_macro_f1_multiref_official(
            classes=classes,
            y_pred=y_pred,
            y_gold_sets=gold_sets,
        )
    else:
        # Task1 始终使用标准的单标签 macro F1（gold_sets 为单元素集合）。
        macro_f1, per_class = _compute_macro_f1_multilabel_ovr(classes=classes, y_pred=y_pred, y_gold_sets=gold_sets)

    print(f"macro_f1\t{macro_f1:.6f}")

    if args.per_class:
        for c in classes:
            m = per_class[c]
            print(
                f"{c}\t"
                f"support={int(m['support'])}\t"
                f"tp={int(m['tp'])}\tfp={int(m['fp'])}\tfn={int(m['fn'])}\t"
                f"p={m['precision']:.6f}\tr={m['recall']:.6f}\tf1={m['f1']:.6f}"
            )

    if not args.no_record:
        pred_input = args.pred_jsonl_path or args.pred_path
        assert pred_input is not None
        record_dir = os.path.dirname(os.path.abspath(pred_input)) or os.getcwd()
        record = {
            "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "task": args.task,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "task2_metric": "multiref_official" if args.task == "task2" else None,
            "args": {
                "gold": args.gold_path,
                "pred": args.pred_path,
                "pred_jsonl": args.pred_jsonl_path,
                "id_key": args.id_key,
                "order_by": args.order_by,
                "strict": args.strict,
                "per_class": args.per_class,
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
            },
        }
        try:
            record_path = _append_eval_record(record_dir=record_dir, record=record)
            print(f"saved_record\t{record_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to write eval record: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
