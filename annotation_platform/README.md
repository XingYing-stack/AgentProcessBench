# AgentProcessBench Annotation Platform

一个可部署的本地/内网数据标注平台（前后端一体），用于对 `output/annotation_file/` 下的 4 种 JSONL（`gaia_dev.jsonl`, `hotpotqa.jsonl`, `tau2.jsonl`, `bfcl.jsonl`）进行逐步标注：

- 以 “assistant 每一步” 为单位打标签（正/负/不确定）
- 单独标注最终结果（正/负/不确定）
- 标注结果落盘到 SQLite + 追加式 JSONL 导出文件，便于后续训练/统计

标注规范手册见：`annotation_platform/ANNOTATION_GUIDE.md`

## 1) 启动

在仓库根目录运行：

```bash
python annotation_platform/server.py --host 0.0.0.0 --port 8000
```

浏览器打开：

- `http://localhost:8000/`

## 1.1) 前端快捷操作

- 每条 assistant 消息右侧有 `+ / 0 / -` 按钮，点击即可标注该步（会高亮）
- 顶部 “最终结果标注” 也用按钮选择 `+1/0/-1`（会高亮）
- 首次打开会弹出 “开始标注” 页面输入用户名（也可在右上角 Annotator 修改）
- 键盘（对当前聚焦的 assistant 步）：
  - `j/k` 或 `↑/↓`：在 assistant 步之间切换
  - `1/0/-`：给当前步打分 `+1/0/-1`

## 2) 数据位置

默认读取：

- `output/annotation_file/*.jsonl`

默认落盘：

- SQLite：`annotation_platform/data/annotations.sqlite3`
- 导出 JSONL：`annotation_platform/data/exports/<dataset>__<username>.jsonl`（每行都带 `username` 字段）

可通过参数覆盖：

```bash
python annotation_platform/server.py --annotation_dir output/annotation_file --data_dir annotation_platform/data
```

## 3) 标注规范（建议）

- `+1`：该步 assistant 判断/行动/输出正确
- `-1`：该步 assistant 判断/行动/输出错误 (建议给-1要比较谨慎，当且仅当这一步得出了错误结论、显著影响了后续过程时标为-1)
- `0`：该步偏探索 / 没有显著影响主线任务 / 没有显著的好处或者坏处 (考虑到 LLM Agent RL过程中给-1会显著影响训练，如果是一步偏探索的步骤，获取了一些补充信息但与主线不强相关，我们将其标为0)

## 4) 导出

平台会在每次保存标注时，向对应导出文件追加一行 JSON（append-only）。

## 5) LLM 自动标注（离线脚本）

在仓库根目录运行（读取环境变量 `OPENAI_BASE_URL` / `OPENAI_API_KEY`，也可用参数覆盖）：

- 不带 `ground_truth`/`reward_info`（推荐用于正式评测的过程打分）：
  - `python -m annotation_platform.llm_annotate output/annotation_file/gaia_dev.jsonl --model <model> --mode blind`
- 带 `ground_truth`/`reward_info`（仅建议用于做参考标注/对齐数据）：
  - `python -m annotation_platform.llm_annotate output/annotation_file/gaia_dev.jsonl --model <model> --mode reference`

默认输出到：`annotation_platform/data/exports/<dataset>__<annotator>.jsonl`，并在每行多写入一个 `explanations` 字段（便于人工复核）。
