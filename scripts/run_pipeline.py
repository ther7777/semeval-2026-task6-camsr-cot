"""
用途：运行 CAMSR-CoT 三阶段提示词推理流水线（dev / evaltest）。

输入：
- --config：YAML 配置（见 configs/camsr_cot_dev.yaml）
- --split：dev / evaltest / both
- --data_path：JSONL 数据路径（优先显式指定）
- （可选）--dev_path / --evaltest_path：当 --split=both 时分别指定两个文件
- API key：默认从环境变量 CAMSR_COT_API_KEY 读取；也可用 --api_key_file 指定

输出（outputs/<timestamp>_<run_id>/）：
- run.log：运行日志
- pipeline_params.json：参数归档（包含 api_call_count）
- experiment.md：实验记录（指标允许漂移，记录实际值即可）
- dev/ 或 evaltest/：
  - gate_results.jsonl / step2_results.jsonl / step3_results.jsonl（中间产物）
  - predictions.jsonl（最终预测）
  - dev 下还会生成：eval_task1.txt / eval_task2.txt / eval_results.jsonl

运行示例（Windows / Linux 单行）：
  python scripts/run_pipeline.py --config configs/camsr_cot_dev.yaml --split dev --data_path your/path/to/dev.jsonl
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from camsr_cot.io import now_stamp_local, now_utc_iso, sha256_file, write_json, write_text  # noqa: E402
from camsr_cot.pipeline import run_pipeline_infer  # noqa: E402


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _load_api_key(*, env_name: str, api_key_file: str) -> str:
    if api_key_file:
        path = _resolve_path(api_key_file)
        if not path.exists():
            raise FileNotFoundError(f"API key file not found: {path}")
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"Empty API key file: {path}")
        return key

    key = os.environ.get(env_name, "").strip()
    if not key:
        raise ValueError(
            f"Missing API key. Please set env '{env_name}' or pass --api_key_file your/path/to/key.txt"
        )
    return key


def _make_logger(run_log_path: Path):
    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} | {msg}"
        print(line)
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(run_log_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    return log


def _run_eval(
    *,
    task: str,
    gold_path: Path,
    pred_jsonl_path: Path,
    out_txt_path: Path,
    per_class: bool,
    no_record: bool,
) -> Tuple[float, str]:
    cmd = [
        sys.executable,
        str((_REPO_ROOT / "scripts" / "eval_competition.py").resolve()),
        "--task",
        task,
        "--gold",
        str(gold_path),
        "--pred_jsonl",
        str(pred_jsonl_path),
    ]
    if per_class:
        cmd.append("--per_class")
    if no_record:
        cmd.append("--no_record")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip() + ("\n" if proc.stdout else "")
    err = (proc.stderr or "").strip()

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text(out, encoding="utf-8", newline="\n")

    if proc.returncode != 0:
        raise RuntimeError(f"Eval failed (task={task}). stderr:\n{err}")

    macro_f1 = None
    for line in out.splitlines():
        if line.startswith("macro_f1\t"):
            try:
                macro_f1 = float(line.split("\t", 1)[1])
            except Exception:
                macro_f1 = None
            break
    if macro_f1 is None:
        raise RuntimeError(f"Cannot parse macro_f1 from eval output (task={task}).")
    return float(macro_f1), out


def _write_experiment_md(
    *,
    output_dir: Path,
    config_path: Path,
    data_paths: Dict[str, Path],
    previous_result_path: Optional[Path],
    eval_scores: Dict[str, Dict[str, float]],
    notes: str,
) -> None:
    lines = []
    lines.append(f"# experiment.md — {output_dir.name}")
    lines.append("")
    lines.append(f"- created_at_utc: `{now_utc_iso()}`")
    lines.append(f"- config: `{config_path.as_posix()}`")
    for split, dp in data_paths.items():
        lines.append(f"- data_path_{split}: `{dp.as_posix()}`")
    if previous_result_path:
        lines.append(f"- previous_result_path: `{previous_result_path.as_posix()}`")
    lines.append("")
    lines.append("## 目标")
    lines.append("- 复现 CAMSR-CoT 主方法三阶段推理流程（Step1 Gate + Step2 9类纠错 + Step3 6类细分）。")
    lines.append("- 指标允许漂移：以流程可跑通与产物齐全为主。")
    lines.append("")
    lines.append("## 结果")
    if "dev" in eval_scores:
        t1 = eval_scores["dev"].get("task1_macro_f1")
        t2 = eval_scores["dev"].get("task2_macro_f1")
        if t1 is not None:
            lines.append(f"- dev task1 macro_f1: `{t1:.6f}`")
        if t2 is not None:
            lines.append(f"- dev task2 macro_f1: `{t2:.6f}`")
    else:
        lines.append("- 本次未运行 dev 评测。")
    lines.append("")
    if notes:
        lines.append("## 备注")
        lines.append(notes.strip())
        lines.append("")

    write_text(output_dir / "experiment.md", "\n".join(lines).strip() + "\n")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML 配置路径")
    p.add_argument("--split", choices=["dev", "evaltest", "both"], default="dev", help="运行 dev / evaltest / both")
    p.add_argument("--data_path", default="", help="数据 JSONL 路径（split=dev 或 evaltest 时使用）")
    p.add_argument("--dev_path", default="", help="split=both 时 dev JSONL 路径")
    p.add_argument("--evaltest_path", default="", help="split=both 时 evaltest JSONL 路径")
    p.add_argument("--gold_path", default="", help="可选：dev 评测用 gold JSONL（默认复用 dev_path/data_path）")

    p.add_argument("--output_root", default="outputs", help="输出根目录（默认 outputs/）")
    p.add_argument("--run_id", default="", help="可选：输出目录后缀（默认用 config.pipeline.name）")
    p.add_argument("--previous_result_path", default="", help="可选：显式复用旧 run 的中间产物目录")
    p.add_argument("--smoke_n", type=int, default=0, help="冒烟模式：仅跑前 N 条样本（0 表示全量）")

    # API key & LLM overrides
    p.add_argument("--api_key_env", default="CAMSR_COT_API_KEY", help="读取 API key 的环境变量名")
    p.add_argument("--api_key_file", default="", help="从文件读取 API key（优先级高于 env）")
    p.add_argument("--base_url", default="", help="覆盖 defaults.base_url")
    p.add_argument("--model", default="", help="覆盖 defaults.model")
    p.add_argument("--concurrency", type=int, default=0, help="覆盖 defaults.concurrency（>0 生效）")
    p.add_argument("--max_retries", type=int, default=0, help="覆盖 defaults.max_retries（>0 生效）")
    p.add_argument("--temperature", type=float, default=-1.0, help="覆盖 defaults.temperature（>=0 生效）")
    p.add_argument("--max_tokens", type=int, default=0, help="覆盖 defaults.max_tokens（>0 生效）")
    p.add_argument("--timeout_s", type=float, default=0.0, help="覆盖 defaults.timeout_s（>0 生效）")

    return p.parse_args(argv)


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)
    defaults = dict(cfg.get("defaults", {}) or {})

    if args.base_url:
        defaults["base_url"] = args.base_url
    if args.model:
        defaults["model"] = args.model
    if args.concurrency and args.concurrency > 0:
        defaults["concurrency"] = int(args.concurrency)
    if args.max_retries and args.max_retries > 0:
        defaults["max_retries"] = int(args.max_retries)
    if args.temperature is not None and float(args.temperature) >= 0:
        defaults["temperature"] = float(args.temperature)
    if args.max_tokens and args.max_tokens > 0:
        defaults["max_tokens"] = int(args.max_tokens)
    if args.timeout_s and float(args.timeout_s) > 0:
        defaults["timeout_s"] = float(args.timeout_s)

    cfg["defaults"] = defaults
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    config_path = _resolve_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = _apply_overrides(cfg, args)

    run_id = args.run_id or str((cfg.get("pipeline", {}) or {}).get("name") or "camsr_cot_run")
    timestamp = now_stamp_local()
    output_root = _resolve_path(args.output_root)
    output_dir = (output_root / f"{timestamp}_{run_id}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_log_path = output_dir / "run.log"
    log = _make_logger(run_log_path)

    previous_result_path = _resolve_path(args.previous_result_path) if args.previous_result_path else None
    api_key = _load_api_key(env_name=str(args.api_key_env), api_key_file=str(args.api_key_file))

    # 解析 split -> data_path
    data_paths: Dict[str, Path] = {}
    if args.split == "both":
        if not args.dev_path or not args.evaltest_path:
            raise ValueError("split=both 时必须同时提供 --dev_path 与 --evaltest_path")
        data_paths["dev"] = _resolve_path(args.dev_path)
        data_paths["evaltest"] = _resolve_path(args.evaltest_path)
    else:
        if not args.data_path:
            raise ValueError("请提供 --data_path your/path/to/data.jsonl")
        data_paths[str(args.split)] = _resolve_path(args.data_path)

    for split, dp in data_paths.items():
        if not dp.exists():
            raise FileNotFoundError(f"Data file not found ({split}): {dp}")

    gold_path = _resolve_path(args.gold_path) if args.gold_path else data_paths.get("dev")

    # pipeline_params.json（先写一版，结束后补齐 api_call_count 与 eval 指标）
    params = {
        "created_at_utc": now_utc_iso(),
        "cwd": os.getcwd(),
        "command": " ".join([sys.executable, str(Path(__file__).as_posix())] + sys.argv[1:]),
        "config_path": str(config_path),
        "config": cfg,
        "args": vars(args),
        "paths": {
            "output_dir": str(output_dir),
            "data_paths": {k: str(v) for k, v in data_paths.items()},
            "gold_path": str(gold_path) if gold_path else "",
            "previous_result_path": str(previous_result_path) if previous_result_path else "",
        },
        "run_stats": {"api_call_count": 0},
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(output_dir / "pipeline_params.json", params)

    # 记录数据签名（便于复现排查；不上传数据本身）
    for split, dp in data_paths.items():
        try:
            params.setdefault("data_signature", {})[split] = {
                "path": str(dp),
                "sha256": sha256_file(dp),
                "bytes": dp.stat().st_size,
            }
        except Exception:
            continue
    write_json(output_dir / "pipeline_params.json", params)

    total_api_calls = 0
    split_stats: Dict[str, Dict[str, Any]] = {}
    eval_scores: Dict[str, Dict[str, float]] = {}

    for split, dp in data_paths.items():
        log(f"=== Running split={split} | data={dp} ===")
        split_dir = output_dir / split
        samples = []
        with open(dp, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    import json as _json

                    samples.append(_json.loads(line))

        paths, stats = asyncio.run(
            run_pipeline_infer(
                repo_root=_REPO_ROOT,
                cfg=cfg,
                samples=samples,
                output_split_dir=split_dir,
                api_key=api_key,
                previous_result_path=previous_result_path,
                smoke_n=int(args.smoke_n or 0),
                log=log,
            )
        )
        split_stats[split] = stats
        total_api_calls += int(stats.get("api_call_count", 0))

        # dev 评测
        if split == "dev" and gold_path:
            t1, _ = _run_eval(
                task="task1",
                gold_path=gold_path,
                pred_jsonl_path=paths.predictions_jsonl,
                out_txt_path=split_dir / "eval_task1.txt",
                per_class=True,
                no_record=False,
            )
            t2, _ = _run_eval(
                task="task2",
                gold_path=gold_path,
                pred_jsonl_path=paths.predictions_jsonl,
                out_txt_path=split_dir / "eval_task2.txt",
                per_class=True,
                no_record=False,
            )
            eval_scores["dev"] = {"task1_macro_f1": float(t1), "task2_macro_f1": float(t2)}

    # experiment.md
    _write_experiment_md(
        output_dir=output_dir,
        config_path=config_path,
        data_paths=data_paths,
        previous_result_path=previous_result_path,
        eval_scores=eval_scores,
        notes="",
    )

    # 补齐 pipeline_params.json
    params["run_stats"]["api_call_count"] = int(total_api_calls)
    params["splits"] = split_stats
    params["eval_scores"] = eval_scores
    write_json(output_dir / "pipeline_params.json", params)

    log(f"Total API Calls: {total_api_calls}")
    print(f"Total API Calls: {total_api_calls}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

