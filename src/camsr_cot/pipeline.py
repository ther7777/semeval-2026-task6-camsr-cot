from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .io import append_jsonl, read_jsonl, read_text, write_jsonl
from .llm_client import ChatConfig, OpenAIChatClient
from .mapping import task2_to_task1
from .prompt_builder import build_gate_prompt, build_stage_prompt


TASK2_LABELS_9 = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]

TASK2_LABELS_6 = [
    "Explicit",
    "Implicit",
    "Partial/half-answer",
    "General",
    "Deflection",
    "Dodging",
]


@dataclass(frozen=True)
class InferencePaths:
    gate_results_jsonl: Path
    step2_results_jsonl: Path
    step3_results_jsonl: Path
    predictions_jsonl: Path


def _parse_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*([^<]+?)\s*</{tag}>", text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_confidence(text: str) -> str:
    raw = _parse_tag(text, "confidence")
    key = raw.strip().lower()
    if key in {"high", "medium", "low"}:
        return key.capitalize()
    return "Unknown"


def _parse_gate_label(text: str, *, non_reply_labels: List[str]) -> str:
    raw = _parse_tag(text, "label")
    if not raw:
        return "OTHER"
    raw_l = raw.strip().lower()
    for nr in non_reply_labels:
        if raw_l == nr.lower() or nr.lower() in raw_l:
            return nr
    return "OTHER"


def _normalize_task2_label(raw: str) -> str:
    s = (raw or "").replace("\\/", "/").strip()
    key = " ".join(s.split()).lower().rstrip(" \t\r\n.;:")
    aliases = {
        "partial/half answer": "Partial/half-answer",
        "partial / half-answer": "Partial/half-answer",
        "partial / half answer": "Partial/half-answer",
    }
    if key in aliases:
        return aliases[key]
    return " ".join(s.split()).rstrip(" \t\r\n.;:")


def _parse_task2_label(text: str, *, restrict_to_6class: bool) -> str:
    raw = _parse_tag(text, "label")
    raw_norm = _normalize_task2_label(raw)
    candidates = TASK2_LABELS_6 if restrict_to_6class else TASK2_LABELS_9

    if raw_norm:
        for c in candidates:
            if raw_norm.lower() == c.lower():
                return c
        # 宽松：包含关系
        for c in candidates:
            if c.lower() in raw_norm.lower():
                return c

    # 再兜底：全文包含
    text_l = (text or "").lower()
    for c in candidates:
        if c.lower() in text_l:
            return c

    return "General"


async def _run_many(
    *,
    ids: List[int],
    build_prompt_fn: Callable[[int], str],
    client: OpenAIChatClient,
    parse_fn: Callable[[str], Tuple[str, str]],
    save_path: Path,
    log: Callable[[str], None],
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    # 为了避免重复写入旧文件：调用方应确保先删/新建输出目录
    if save_path.exists():
        save_path.unlink()

    async def run_one(sample_id: int) -> None:
        prompt = build_prompt_fn(sample_id)
        resp = await client.chat(prompt=prompt)
        label, conf = parse_fn(resp)
        obj = {"id": sample_id, "label": label, "confidence": conf, "raw": resp}
        results[sample_id] = obj
        append_jsonl(save_path, obj)

    tasks: List[asyncio.Task[None]] = []
    for i, sample_id in enumerate(ids, start=1):
        tasks.append(asyncio.create_task(run_one(sample_id)))
        if i % 50 == 0:
            log(f"Scheduled {i}/{len(ids)} tasks for {save_path.name}")

    if tasks:
        await asyncio.gather(*tasks)
    return results


def _load_existing_results(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    out: Dict[int, Dict[str, Any]] = {}
    for obj in rows:
        if obj.get("id") is None:
            continue
        try:
            sample_id = int(obj["id"])
        except Exception:
            continue
        out[sample_id] = obj
    return out


async def run_pipeline_infer(
    *,
    repo_root: Path,
    cfg: Dict[str, Any],
    samples: List[Dict[str, Any]],
    output_split_dir: Path,
    api_key: str,
    previous_result_path: Optional[Path],
    smoke_n: int,
    log: Callable[[str], None],
) -> Tuple[InferencePaths, Dict[str, Any]]:
    """
    执行三阶段推理并落盘中间产物/最终预测。

    约定：
    - samples 里的每条样本需要有 `index`（int）；缺失时按行号生成。
    - 输出写入 output_split_dir 下（例如 outputs/.../dev/）。
    """
    output_split_dir.mkdir(parents=True, exist_ok=True)

    if smoke_n and smoke_n > 0:
        samples = samples[: int(smoke_n)]
        log(f"SMOKE 模式：仅处理前 N={len(samples)} 条样本。")

    # ===== 基本配置 =====
    defaults = cfg.get("defaults", {}) or {}
    gate_cfg = cfg.get("gate", {}) or {}
    step2_cfg = cfg.get("step2_correction", {}) or {}
    step3_cfg = cfg.get("step3_spectrum", {}) or {}

    non_reply_labels = list(gate_cfg.get("non_reply_labels") or [])
    gate_prompt_path = (repo_root / str(gate_cfg.get("prompt_path"))).resolve()
    gate_template = read_text(gate_prompt_path)

    chat_cfg = ChatConfig(
        base_url=str(defaults.get("base_url", "")).strip(),
        model=str(defaults.get("model", "")).strip(),
        temperature=float(defaults.get("temperature", 0.0)),
        max_tokens=int(defaults.get("max_tokens", 1800)),
        timeout_s=float(defaults.get("timeout_s", 90.0)),
        max_retries=int(defaults.get("max_retries", 5)),
        concurrency=int(defaults.get("concurrency", 8)),
    )

    # ===== 样本索引 =====
    ordered_ids: List[int] = []
    sample_by_id: Dict[int, Dict[str, Any]] = {}
    for i, s in enumerate(samples):
        raw_id = s.get("index", i)
        try:
            sample_id = int(raw_id)
        except Exception:
            sample_id = i
        ordered_ids.append(sample_id)
        sample_by_id[sample_id] = s

    paths = InferencePaths(
        gate_results_jsonl=output_split_dir / "gate_results.jsonl",
        step2_results_jsonl=output_split_dir / "step2_results.jsonl",
        step3_results_jsonl=output_split_dir / "step3_results.jsonl",
        predictions_jsonl=output_split_dir / "predictions.jsonl",
    )

    # ===== previous_result_path 复用（可选）=====
    prev_dir = None
    if previous_result_path:
        cand = (previous_result_path / output_split_dir.name)
        prev_dir = cand if cand.exists() else previous_result_path
        log(f"previous_result_path 已启用：{prev_dir}")

    prev_gate = _load_existing_results(prev_dir / "gate_results.jsonl") if prev_dir else {}
    prev_step2 = _load_existing_results(prev_dir / "step2_results.jsonl") if prev_dir else {}
    prev_step3 = _load_existing_results(prev_dir / "step3_results.jsonl") if prev_dir else {}

    # ===== Step1: Gate =====
    gate_results: Dict[int, Dict[str, Any]] = {}

    def _gate_prompt(sample_id: int) -> str:
        return build_gate_prompt(gate_template=gate_template, sample=sample_by_id[sample_id])

    def _gate_parse(resp: str) -> Tuple[str, str]:
        label = _parse_gate_label(resp, non_reply_labels=non_reply_labels)
        conf = _parse_confidence(resp)
        return label, conf

    gate_missing = [sid for sid in ordered_ids if sid not in prev_gate]
    if prev_gate:
        log(f"复用 Gate 结果：{len(prev_gate)} 条；需要重跑：{len(gate_missing)} 条。")

    async with OpenAIChatClient(api_key=api_key, cfg=chat_cfg) as client:
        if gate_missing:
            gate_new = await _run_many(
                ids=gate_missing,
                build_prompt_fn=_gate_prompt,
                client=client,
                parse_fn=_gate_parse,
                save_path=paths.gate_results_jsonl,
                log=log,
            )
        else:
            gate_new = {}

        # 合并 + 落盘（确保输出目录有完整 gate_results.jsonl）
        gate_results = {**prev_gate, **gate_new}
        write_jsonl(paths.gate_results_jsonl, [gate_results[i] for i in sorted(gate_results.keys())])

        # ===== 路由 =====
        step2_ids: List[int] = []
        step3_ids: List[int] = []
        direct_ids: List[int] = []
        for sid in ordered_ids:
            g = gate_results.get(sid, {})
            g_label = str(g.get("label", "OTHER"))
            g_conf = str(g.get("confidence", "Unknown"))
            if g_label in non_reply_labels and g_conf == "High":
                direct_ids.append(sid)
            elif g_label in non_reply_labels:
                step2_ids.append(sid)
            else:
                step3_ids.append(sid)

        log(f"Gate 路由统计：direct={len(direct_ids)} | step2={len(step2_ids)} | step3={len(step3_ids)}")

        # ===== Step2: 9 类纠错复核 =====
        step2_results: Dict[int, Dict[str, Any]] = {}
        step2_templates_dir = (repo_root / str(step2_cfg.get("templates_dir"))).resolve()
        step2_template_path = step2_templates_dir / "stage2_dynamic_template.txt"
        step2_defs_path = (repo_root / str(step2_cfg.get("definitions_path"))).resolve()
        step2_fewshots_dir = (repo_root / str(step2_cfg.get("fewshots_dir"))).resolve()

        def _step2_prompt(sample_id: int) -> str:
            g = gate_results[sample_id]
            return build_stage_prompt(
                sample=sample_by_id[sample_id],
                gate_label=str(g.get("label", "")),
                gate_confidence=str(g.get("confidence", "Unknown")),
                template_path=step2_template_path,
                definitions_path=step2_defs_path,
                templates_dir=step2_templates_dir,
                fewshots_dir=step2_fewshots_dir,
                include_prior_analysis=bool(step2_cfg.get("include_prior_analysis", True)),
                num_anchors=int(step2_cfg.get("num_anchors", 2)),
                num_boundaries=int(step2_cfg.get("num_boundaries", 3)),
            )

        def _step2_parse(resp: str) -> Tuple[str, str]:
            label = _parse_task2_label(resp, restrict_to_6class=False)
            conf = _parse_confidence(resp)
            return label, conf

        step2_missing = [sid for sid in step2_ids if sid not in prev_step2]
        if prev_step2:
            log(f"复用 Step2 结果：{len(prev_step2)} 条；需要重跑：{len(step2_missing)} 条。")

        if step2_missing:
            step2_new = await _run_many(
                ids=step2_missing,
                build_prompt_fn=_step2_prompt,
                client=client,
                parse_fn=_step2_parse,
                save_path=paths.step2_results_jsonl,
                log=log,
            )
        else:
            step2_new = {}

        step2_results = {**prev_step2, **step2_new}
        write_jsonl(paths.step2_results_jsonl, [step2_results[i] for i in sorted(step2_results.keys())])

        # ===== Step3: 6 类 spectrum 细分 =====
        step3_results: Dict[int, Dict[str, Any]] = {}
        step3_templates_dir = (repo_root / str(step3_cfg.get("templates_dir"))).resolve()
        step3_template_path = step3_templates_dir / "stage2_dynamic_template.txt"
        step3_defs_path = (repo_root / str(step3_cfg.get("definitions_path"))).resolve()
        step3_fewshots_dir = (repo_root / str(step3_cfg.get("fewshots_dir"))).resolve()

        def _step3_prompt(sample_id: int) -> str:
            g = gate_results[sample_id]
            return build_stage_prompt(
                sample=sample_by_id[sample_id],
                gate_label="OTHER",
                gate_confidence=str(g.get("confidence", "Unknown")),
                template_path=step3_template_path,
                definitions_path=step3_defs_path,
                templates_dir=step3_templates_dir,
                fewshots_dir=step3_fewshots_dir,
                include_prior_analysis=bool(step3_cfg.get("include_prior_analysis", False)),
                num_anchors=int(step3_cfg.get("num_anchors", 6)),
                num_boundaries=int(step3_cfg.get("num_boundaries", 10)),
            )

        def _step3_parse(resp: str) -> Tuple[str, str]:
            label = _parse_task2_label(resp, restrict_to_6class=True)
            conf = _parse_confidence(resp)
            return label, conf

        step3_missing = [sid for sid in step3_ids if sid not in prev_step3]
        if prev_step3:
            log(f"复用 Step3 结果：{len(prev_step3)} 条；需要重跑：{len(step3_missing)} 条。")

        if step3_missing:
            step3_new = await _run_many(
                ids=step3_missing,
                build_prompt_fn=_step3_prompt,
                client=client,
                parse_fn=_step3_parse,
                save_path=paths.step3_results_jsonl,
                log=log,
            )
        else:
            step3_new = {}

        step3_results = {**prev_step3, **step3_new}
        write_jsonl(paths.step3_results_jsonl, [step3_results[i] for i in sorted(step3_results.keys())])

        # ===== Merge: predictions.jsonl =====
        preds_rows: List[Dict[str, Any]] = []
        for sid in ordered_ids:
            g = gate_results.get(sid, {})
            g_label = str(g.get("label", "OTHER"))
            g_conf = str(g.get("confidence", "Unknown"))

            if sid in direct_ids:
                task2_label = g_label
                source = "gate_direct"
            elif sid in step2_ids:
                task2_label = str(step2_results.get(sid, {}).get("label", "General"))
                source = "step2_correction"
            else:
                task2_label = str(step3_results.get(sid, {}).get("label", "General"))
                source = "step3_spectrum"

            task1_label = task2_to_task1(task2_label, mapping=cfg.get("mapping"))
            preds_rows.append(
                {
                    "id": sid,
                    "task1_prediction": task1_label,
                    "task2_prediction": task2_label,
                    "gate_label": g_label,
                    "gate_confidence": g_conf,
                    "source": source,
                }
            )

        write_jsonl(paths.predictions_jsonl, preds_rows)
        log(f"Wrote predictions: {paths.predictions_jsonl}")

        stats = {
            "num_samples": len(ordered_ids),
            "num_direct": len(direct_ids),
            "num_step2": len(step2_ids),
            "num_step3": len(step3_ids),
            "api_call_count": int(client.api_call_count),
            "gate_prompt_path": str(gate_prompt_path.relative_to(repo_root)) if gate_prompt_path.exists() else str(gate_prompt_path),
        }
        return paths, stats
