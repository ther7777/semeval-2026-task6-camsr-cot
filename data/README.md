# Data (JSONL)

本仓库提供三份 JSONL 文件，便于复现实验与直接跑通 pipeline：

- `data/train.jsonl`：训练集（可选；主要用于复现实验/自行训练，跑推理不需要）
- `data/dev.jsonl`：开发集（含 dev gold；可本地评测）
- `data/eval_test.jsonl`：评测集（无 gold；用于跑推理/打包提交）

## 字段契约（最小要求）

每行一个 JSON object，至少包含：
- `index`：样本 id（int）
- `interview_question`：采访问题段（str）
- `interview_answer`：回答段（str）
- `question`：sub-question（str，本方法判定口径以此为准）

dev 评测相关字段（用于 Task2 multi-reference 口径）：
- `clarity_label`（Task1 gold）
- `annotator1` / `annotator2` / `annotator3`（Task2 多参考 gold）

## 快速检查

```bash
python scripts/check_data_layout.py --data_path data/dev.jsonl
```
