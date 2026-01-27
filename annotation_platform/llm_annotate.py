from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from utils.openai_client import openai_chat_completions
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.openai_client import openai_chat_completions

from tqdm import tqdm


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_record_id(dataset_name: str, obj: dict[str, Any]) -> str:
    data_source = str(obj.get("data_source") or dataset_name)
    query_index = obj.get("query_index")
    sample_index = obj.get("sample_index")
    if query_index is not None and sample_index is not None:
        return f"{data_source}:{query_index}:{sample_index}"
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{data_source}:{digest}"


def _assistant_message_indices(messages: Any) -> list[int]:
    if not isinstance(messages, list):
        return []
    indices: list[int] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            indices.append(i)
    return indices


def _env_first(names: list[str]) -> str | None:
    for name in names:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return None


JUDGE_RUBRIC = """
You are a strict but fair trajectory annotator for tool-use agents.

You will be given one complete trajectory consisting of system, user, assistant,
and tool messages, together with the tool definitions.

Your task is to label EACH assistant message (each assistant message constitutes
one Step) using the following scheme:

+1: Correct and effective.
    The step is factually correct given the information available at that time
    and clearly moves the task closer to successful completion by:
    (i) correctly invoking a tool or interpreting tool outputs, or
    (ii) introducing valid constraints, decisions, or information that
         reduces the remaining uncertainty of the task.

 0: Neutral or exploratory.
    The step is reasonable but has limited or unclear impact on task progress.
    This includes exploratory reasoning, redundant restatements, partial planning,
    or cases where the correctness is debatable given the available evidence.
    Tool calls that fail due to external reasons (e.g., timeout, 404), when the
    attempt itself is reasonable, are typically labeled 0.

-1: Incorrect or harmful.
    The step contains factual errors, misinterprets tool outputs, violates
    constraints, repeats failed actions without a meaningful change in strategy,
    fabricates tool results or evidence, or otherwise pushes the trajectory away
    from successful completion.

Important rules:

- Only assistant messages are labeled. User and tool messages serve only as evidence.
- Avoid hindsight bias: judge each step strictly based on the information available
  up to that point in the trajectory.
- Any step labeled -1 triggers a cumulative penalty: all subsequent assistant steps
  in the same workflow should also be labeled -1, unless one of the following holds:
    (i) the assistant explicitly acknowledges and corrects the earlier mistake, or
    (ii) the assistant produces a subsequent step that no longer depends on the
         incorrect assumption and effectively resumes progress toward the task.
- Repeating the same failed action without a meaningful change in parameters or
  strategy typically transitions from 0 to -1.
- If an incorrect statement does not affect any subsequent reasoning or actions
  and is not relied upon later, it may be labeled 0; otherwise, it should be labeled -1.
- Violating any policy or requirement specified in the system prompt is always -1.

After labeling all assistant steps, also assign a label to:

FINAL_RESULT:
+1: The overall task is successfully completed.
 0: The outcome is partial, ambiguous, or incomplete.
-1: The task fails due to incorrect reasoning, tool misuse, or unresolved errors.

Return STRICT JSON ONLY.
Do not include explanations, markdown, or any additional text.
"""


REFERENCE_MODE_JUDGE_RUBRIC = """You are a strict but fair trajectory annotator for tool-use agents.

You will be given one full trajectory (system/user/assistant/tool messages), plus tool definitions.
Your job is to label EACH assistant message (each assistant message is one Step) as:
- +1: correct/reasonable and clearly moves the task closer to completion
-  0: neutral / exploratory / low-impact / insufficient evidence / debatable
- -1: wrong/misleading/redundant/violates constraints OR pushes trajectory away from success

Important:
- Only assistant messages are labeled. User/tool messages are evidence.
- Any -1 score triggers a cumulative penalty: all following steps in that workflow remain -1 unless the mistake is fixed or an independent task begins.
- If a tool call fails for external reasons (timeout/404/etc) and the attempt was reasonable: usually 0.
- Repeating the same failed action without changing strategy/params trends from 0 to -1.
- Making up tool results, citing non-existent evidence, or misreading tool outputs is -1.
- Avoid hindsight bias: judge each step by information available up to that point
- Violating the policy requirements in the system prompt is -1.

Reference mode only:
- You may use ground_truth/reward_info to verify the FINAL_RESULT and to spot incorrect steps.

You must also label FINAL_RESULT as +1/0/-1 for the overall outcome.
Return STRICT JSON ONLY (no Markdown, no extra text)."""

@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    api_key: str | None
    model: str
    temperature: float
    max_tokens: int
    timeout_s: int
    mode: str  # "blind" | "reference"


def _truncate_text(text: Any, *, max_chars: int) -> Any:
    if not isinstance(text, str):
        return text
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    removed = len(text) - (head + tail)
    return text[:head] + f"\n…[TRUNCATED {removed} CHARS]…\n" + text[-tail:]


def _build_judge_input(
    *,
    item: dict[str, Any],
    dataset: str,
    assistant_indices: list[int],
    cfg: JudgeConfig,
) -> list[dict[str, Any]]:

    payload: dict[str, Any] = {
        "question": item.get("question"),
        "task_description": item.get("task_description"),
        "tools": item.get("tools"),
        "messages": item.get("messages"),
        "assistant_message_indices": assistant_indices,
        "notes": {
            "step_definition": "Each Step == one message with role=='assistant'. Use the given indices.",
            "output_requirements": "Return JSON with step_labels, final_label, explanations.",
        },
    }

    if cfg.mode == "reference":
        payload["ground_truth"] = item.get("ground_truth")
        payload["reward_info"] = item.get("reward_info")
        used_judge_rubric = REFERENCE_MODE_JUDGE_RUBRIC
    else:
        used_judge_rubric = JUDGE_RUBRIC

    user_instructions = """Label every index in assistant_message_indices.

Output JSON schema:
{
  "step_labels": {"<assistant_index>": -1|0|1, ...},
  "final_label": -1|0|1,
  "explanations": {
    "steps": {"<assistant_index>": "short reason for humans", ...},
    "final": "short reason for humans"
  }
}

Rules:
- step_labels MUST contain ALL assistant indices (as strings).
- explanations.steps MUST contain ALL assistant indices (as strings).
- Keep each explanation concise (<= 2 sentences).
"""

    return [
        {"role": "system", "content": used_judge_rubric},
        {"role": "user", "content": user_instructions + "\n\nTRAJECTORY_JSON:\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("LLM output is not JSON")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("LLM output JSON is not an object")
    return obj


def _coerce_int_label(val: Any) -> int:
    if isinstance(val, bool):
        raise ValueError("label must be -1/0/1, got bool")
    if isinstance(val, int):
        out = val
    elif isinstance(val, str) and val.strip() in {"-1", "0", "1"}:
        out = int(val.strip())
    else:
        raise ValueError(f"label must be -1/0/1, got {val!r}")
    if out not in (-1, 0, 1):
        raise ValueError(f"label must be -1/0/1, got {out}")
    return out


def _normalize_judge_output(
    raw: dict[str, Any],
    *,
    assistant_indices: list[int],
) -> tuple[dict[str, int], int, dict[str, Any]]:
    if "step_labels" not in raw or "final_label" not in raw:
        raise ValueError("missing step_labels/final_label in judge output")
    step_labels_raw = raw.get("step_labels")
    if not isinstance(step_labels_raw, dict):
        raise ValueError("step_labels must be an object")

    expected_keys = [str(i) for i in assistant_indices]
    step_labels: dict[str, int] = {}
    for k in expected_keys:
        if k not in step_labels_raw:
            raise ValueError(f"missing step_labels[{k}]")
        step_labels[k] = _coerce_int_label(step_labels_raw[k])

    final_label = _coerce_int_label(raw.get("final_label"))

    explanations = raw.get("explanations") or {}
    if not isinstance(explanations, dict):
        raise ValueError("explanations must be an object")
    steps_expl = explanations.get("steps") or {}
    if not isinstance(steps_expl, dict):
        raise ValueError("explanations.steps must be an object")
    for k in expected_keys:
        if k not in steps_expl:
            raise ValueError(f"missing explanations.steps[{k}]")
        if not isinstance(steps_expl[k], str):
            steps_expl[k] = str(steps_expl[k])
    final_expl = explanations.get("final")
    if not isinstance(final_expl, str):
        final_expl = "" if final_expl is None else str(final_expl)
    explanations_out = {"steps": steps_expl, "final": final_expl}

    return step_labels, final_label, explanations_out


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            yield i, obj


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_existing_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    record_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            rid = obj.get("record_id")
            if isinstance(rid, str) and rid:
                record_ids.add(rid)
    return record_ids


def _count_remaining_jsonl_items(
    path: Path,
    *,
    start: int,
    end: int,
    dataset: str,
    existing_record_ids: set[str],
) -> int:
    if not existing_record_ids:
        return _count_selected_jsonl_items(path, start=start, end=end)
    cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if end >= 0 and i >= end:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            rid = _stable_record_id(dataset, obj)
            if rid in existing_record_ids:
                continue
            cnt += 1
    return cnt


def _count_selected_jsonl_items(path: Path, *, start: int, end: int) -> int:
    cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if end >= 0 and i >= end:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                cnt += 1
    return cnt


def annotate_file(
    *,
    input_path: Path,
    output_path: Path,
    raw_output_path: Path | None,
    dataset: str,
    annotator: str,
    username: str,
    cfg: JudgeConfig,
    start: int,
    end: int,
    concurrency: int,
    dry_run: bool,
) -> None:
    if input_path.suffix == ".json":
        item = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(item, dict):
            raise ValueError(f"expected JSON object in {input_path}")
        items: Iterable[tuple[int, dict[str, Any]]] = [(0, item)]
    else:
        items = _iter_jsonl(input_path)

    existing_record_ids = _load_existing_record_ids(output_path) if not dry_run else set()

    def _annotate_one(
        index_in_dataset: int, item: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
        assistant_indices = _assistant_message_indices(item.get("messages"))
        if not assistant_indices:
            return None

        messages = _build_judge_input(
            item=item,
            dataset=dataset,
            assistant_indices=assistant_indices,
            cfg=cfg,
        )

        outs: list[dict[str, Any]] = []
        content = ""
        last_error: str | None = None
        step_labels: dict[str, Any]
        final_label: Any
        explanations: dict[str, Any]

        if cfg.mode == 'reference':
            max_attempts = 5
        else:
            max_attempts = 1
        print(f'In {cfg.mode} mode, using max_attempts={max_attempts}')
        for attempt in range(1, max_attempts + 1):
            try:
                outs = openai_chat_completions(
                    base_url=cfg.base_url,
                    model=cfg.model,
                    messages=messages,
                    n=1,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    timeout_s=cfg.timeout_s,
                    api_key=cfg.api_key,
                )
                content = (outs[0].get("content") or "").strip() if outs else ""
                raw = _extract_json_object(content)
                step_labels, final_label, explanations = _normalize_judge_output(
                    raw, assistant_indices=assistant_indices
                )
                last_error = None
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_attempts:
                    time.sleep(0.2 * attempt)
                    continue
                step_labels = {str(i): None for i in assistant_indices}
                final_label = 0
                explanations = {
                    "steps": {str(i): None for i in assistant_indices},
                    "final": None,
                }

        record_id = _stable_record_id(dataset, item)
        out_record: dict[str, Any] = {
            "dataset": dataset,
            "record_id": record_id,
            "annotator": annotator,
            "username": username,
            "index_in_dataset": index_in_dataset,
            "data_source": item.get("data_source"),
            "query_index": item.get("query_index"),
            "sample_index": item.get("sample_index"),
            "step_labels": step_labels,
            "final_label": final_label,
            "final_label_touched": True,
            "status": "done",
            "comment": "" if last_error is None else f"llm_annotate_failed: {last_error}",
            "updated_at": _utc_now_iso(),
            "explanations": explanations,
        }

        raw_record: dict[str, Any] | None = None
        if raw_output_path is not None:
            raw_record = {
                "dataset": dataset,
                "record_id": record_id,
                "mode": cfg.mode,
                "judge_model": cfg.model,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "timeout_s": cfg.timeout_s,
                "index_in_dataset": index_in_dataset,
                "data_source": item.get("data_source"),
                "query_index": item.get("query_index"),
                "sample_index": item.get("sample_index"),
                "judge_messages": messages,
                "judge_outs": outs,
                "judge_content": content,
                "error": last_error,
                "updated_at": _utc_now_iso(),
            }

        return out_record, raw_record

    def _iter_selected() -> Iterable[tuple[int, dict[str, Any]]]:
        for index_in_dataset, item in items:
            if index_in_dataset < start:
                continue
            if end >= 0 and index_in_dataset >= end:
                break
            if existing_record_ids:
                rid = _stable_record_id(dataset, item)
                if rid in existing_record_ids:
                    continue
            yield index_in_dataset, item

    total: int | None = None
    if not dry_run and tqdm is not None:
        if input_path.suffix == ".json":
            if 0 < start or (end >= 0 and 0 >= end):
                total = 0
            else:
                rid = _stable_record_id(dataset, item)
                total = 0 if rid in existing_record_ids else 1
        else:
            total = _count_remaining_jsonl_items(
                input_path,
                start=start,
                end=end,
                dataset=dataset,
                existing_record_ids=existing_record_ids,
            )

    if dry_run:
        for index_in_dataset, item in _iter_selected():
            assistant_indices = _assistant_message_indices(item.get("messages"))
            if not assistant_indices:
                continue
            messages = _build_judge_input(
                item=item,
                dataset=dataset,
                assistant_indices=assistant_indices,
                cfg=cfg,
            )
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            return
        return

    if concurrency <= 1:
        selected_iter: Iterable[tuple[int, dict[str, Any]]] = _iter_selected()
        if tqdm is not None:
            selected_iter = tqdm(
                selected_iter,
                total=total,
                desc="annotate",
                unit="item",
            )
        for index_in_dataset, item in selected_iter:
            res = _annotate_one(index_in_dataset, item)
            if res is not None:
                record, raw_record = res
                _append_jsonl(output_path, record)
                if raw_output_path is not None and raw_record is not None:
                    _append_jsonl(raw_output_path, raw_record)
        return

    max_workers = max(1, int(concurrency))
    max_in_flight = max_workers * 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=total, desc="annotate", unit="item")
        pending: set[
            concurrent.futures.Future[tuple[dict[str, Any], dict[str, Any] | None] | None]
        ] = set()

        for index_in_dataset, item in _iter_selected():
            pending.add(executor.submit(_annotate_one, index_in_dataset, item))
            if len(pending) >= max_in_flight:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    res = fut.result()
                    if pbar is not None:
                        pbar.update(1)
                    if res is not None:
                        record, raw_record = res
                        _append_jsonl(output_path, record)
                        if raw_output_path is not None and raw_record is not None:
                            _append_jsonl(raw_output_path, raw_record)

        while pending:
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                res = fut.result()
                if pbar is not None:
                    pbar.update(1)
                if res is not None:
                    record, raw_record = res
                    _append_jsonl(output_path, record)
                    if raw_output_path is not None and raw_record is not None:
                        _append_jsonl(raw_output_path, raw_record)
        if pbar is not None:
            pbar.close()
    return


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge trajectories into annotation_platform exports JSONL.")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to a trajectory JSONL (e.g. output/annotation_file/gaia_dev.jsonl) or a single JSON object file.",
    )
    parser.add_argument("--dataset", type=str, default="", help="Dataset name (default: stem of input file).")
    parser.add_argument(
        "--mode",
        type=str,
        default="blind",
        choices=["blind", "reference"],
        help="blind: do NOT provide ground_truth/reward_info to judge; reference: include them.",
    )
    parser.add_argument("--model", type=str, required=True, help="OpenAI-compatible model name.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=30000)
    parser.add_argument("--timeout_s", type=int, default=300)
    parser.add_argument("--base_url", type=str, default="", help="OpenAI-compatible base URL (or read from env).")
    parser.add_argument("--api_key", type=str, default="", help="API key (or read from env).")
    parser.add_argument("--output_path", type=str, default="", help="Output JSONL path (default: annotation_platform/data/llm_annotations/<dataset>__<annotator>.jsonl).")
    parser.add_argument(
        "--raw_output_dir",
        type=str,
        default="./data/llm_annotations_raw",
        help='Also write raw judge prompt/response JSONL to this dir (filename: f"{dataset}__{mode}_{annotator}.jsonl"). Use empty string to disable.',
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in input (inclusive).")
    parser.add_argument("--end", type=int, default=-1, help="End index in input (exclusive); -1 means all.")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent LLM requests (default: 8). Use 1 to disable.")
    parser.add_argument("--dry_run", action="store_true", help="Print the judge prompt messages for the first item and exit.")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    dataset = args.dataset.strip() or input_path.stem
    annotator = args.model.strip()
    username = args.model.strip()

    base_url = args.base_url.strip() or _env_first(["OPENAI_BASE_URL", "LLM_BASE_URL", "base_url", "BASE_URL"])
    if not base_url:
       raise Exception("Missing base_url: pass --base_url or set BASE_URL.")

    api_key = args.api_key.strip() or _env_first(["OPENAI_API_KEY", "api_key", "API_KEY"])
    if not api_key:
        raise Exception("Missing api_key: pass --api_key or set API_KEY.")

    output_path = Path(args.output_path) if args.output_path else Path("data/llm_annotations") / f"{dataset}__{args.mode}_{annotator}.jsonl"
    raw_output_dir = args.raw_output_dir.strip()
    raw_output_path = None if not raw_output_dir else Path(raw_output_dir) / output_path.name

    cfg = JudgeConfig(
        base_url=base_url,
        api_key=api_key,
        model=args.model,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        timeout_s=int(args.timeout_s),
        mode=args.mode,
    )
    annotate_file(
        input_path=input_path,
        output_path=output_path,
        raw_output_path=raw_output_path,
        dataset=dataset,
        annotator=annotator,
        username=username,
        cfg=cfg,
        start=int(args.start),
        end=int(args.end),
        concurrency=int(args.concurrency),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
