from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from annotation_platform.dataset import DatasetIndex, discover_datasets
    from annotation_platform.storage import AnnotationStore
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from annotation_platform.dataset import DatasetIndex, discover_datasets
    from annotation_platform.storage import AnnotationStore


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


class AppState:
    def __init__(self, dataset_dir: Path, data_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.data_dir = data_dir
        self.static_dir = Path(__file__).resolve().parent / "static"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        exports_dir = self.data_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        self.store = AnnotationStore(self.data_dir / "annotations.sqlite3", exports_dir=exports_dir)
        self.datasets: dict[str, DatasetIndex] = discover_datasets(dataset_dir)
        self.lock = threading.Lock()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "AgentProcessBenchAnnotation/0.1"

    @property
    def state(self) -> AppState:
        return self.server.state  # type: ignore[attr-defined]

    def _serve_static(self, rel_path: str) -> None:
        if rel_path == "" or rel_path == "/":
            rel_path = "index.html"
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]

        safe_path = (self.state.static_dir / rel_path).resolve()
        if not str(safe_path).startswith(str(self.state.static_dir.resolve())):
            self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not safe_path.exists() or not safe_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        ctype, _ = mimetypes.guess_type(str(safe_path))
        ctype = ctype or "application/octet-stream"
        content = safe_path.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path.startswith("/static/"):
            rel = "index.html" if path == "/" else path[len("/static/") :]
            self._serve_static(rel)
            return

        if path == "/api/datasets":
            datasets = []
            for ds in sorted(self.state.datasets.values(), key=lambda d: d.name):
                datasets.append(
                    {
                        "name": ds.name,
                        "path": str(ds.path),
                        "size": ds.size,
                    }
                )
            _json_response(self, HTTPStatus.OK, {"datasets": datasets})
            return

        if path == "/api/progress":
            dataset = (qs.get("dataset", [""])[0] or "").strip()
            annotator = (qs.get("annotator", [""])[0] or "").strip()
            if dataset == "" or annotator == "":
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dataset and annotator required"})
                return
            ds = self.state.datasets.get(dataset)
            if ds is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
                return
            progress = self.state.store.get_progress(dataset=dataset, annotator=annotator)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "dataset": dataset,
                    "annotator": annotator,
                    "done": progress.done,
                    "skipped": progress.skipped,
                    "total": ds.size,
                },
            )
            return

        if path == "/api/item":
            dataset = (qs.get("dataset", [""])[0] or "").strip()
            annotator = (qs.get("annotator", [""])[0] or "").strip()
            index_str = (qs.get("index", [""])[0] or "").strip()
            if dataset == "" or index_str == "":
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dataset and index required"})
                return
            ds = self.state.datasets.get(dataset)
            if ds is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
                return
            try:
                index = int(index_str)
            except ValueError:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "index must be int"})
                return

            if index < 0 or index >= ds.size:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "index out of range"})
                return
            item = ds.read_item(index)
            record_id = ds.record_ids[index]

            existing = None
            if annotator != "":
                existing = self.state.store.get_annotation(dataset=dataset, record_id=record_id, annotator=annotator)

            payload = ds.to_payload(item=item, dataset=dataset, index=index, record_id=record_id, existing=existing)
            _json_response(self, HTTPStatus.OK, payload)
            return

        if path == "/api/next":
            dataset = (qs.get("dataset", [""])[0] or "").strip()
            annotator = (qs.get("annotator", [""])[0] or "").strip()
            if dataset == "" or annotator == "":
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dataset and annotator required"})
                return
            ds = self.state.datasets.get(dataset)
            if ds is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
                return

            cursor = self.state.store.get_cursor(dataset=dataset, annotator=annotator)
            next_index = self.state.store.find_next_unannotated(
                dataset=dataset, annotator=annotator, record_ids=ds.record_ids, start_index=cursor
            )
            if next_index is None:
                _json_response(self, HTTPStatus.OK, {"done": True, "dataset": dataset})
                return

            item = ds.read_item(next_index)
            record_id = ds.record_ids[next_index]
            existing = self.state.store.get_annotation(dataset=dataset, record_id=record_id, annotator=annotator)
            payload = ds.to_payload(
                item=item, dataset=dataset, index=next_index, record_id=record_id, existing=existing
            )
            _json_response(self, HTTPStatus.OK, payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path != "/api/annotation":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        body = _read_json_body(self)
        dataset = (body.get("dataset") or "").strip()
        record_id = (body.get("record_id") or "").strip()
        annotator = (body.get("annotator") or "").strip()
        index_in_dataset = body.get("index_in_dataset")
        step_labels = body.get("step_labels") or {}
        final_label = body.get("final_label")
        final_label_touched = body.get("final_label_touched")
        status = (body.get("status") or "in_progress").strip()
        comment = body.get("comment")
        data_source = body.get("data_source")
        query_index = body.get("query_index")
        sample_index = body.get("sample_index")
        username = body.get("username")

        if dataset == "" or record_id == "" or annotator == "":
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "dataset, record_id, annotator required"})
            return
        if dataset not in self.state.datasets:
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
            return
        if not isinstance(step_labels, dict):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "step_labels must be dict"})
            return
        if final_label is None:
            final_label = 0
        if not isinstance(final_label, int) or final_label not in (-1, 0, 1):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "final_label must be -1/0/1"})
            return
        if isinstance(final_label_touched, bool):
            final_label_touched_bool = final_label_touched
        elif isinstance(final_label_touched, int) and final_label_touched in (0, 1):
            final_label_touched_bool = bool(final_label_touched)
        elif final_label_touched is None:
            final_label_touched_bool = False
        else:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "final_label_touched must be bool"})
            return
        if status not in ("in_progress", "done", "skipped"):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "status must be in_progress/done/skipped"})
            return
        if index_in_dataset is None:
            index_in_dataset = -1
        if not isinstance(index_in_dataset, int):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "index_in_dataset must be int"})
            return
        if data_source is not None and not isinstance(data_source, str):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "data_source must be str"})
            return
        if query_index is not None and not isinstance(query_index, int):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "query_index must be int"})
            return
        if sample_index is not None and not isinstance(sample_index, int):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "sample_index must be int"})
            return
        if username is not None and not isinstance(username, str):
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "username must be str"})
            return
        if username is None:
            username = annotator

        if status in ("done", "in_progress"):
            if not final_label_touched_bool:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"final_label_touched required for status={status}"},
                )
                return

            ds = self.state.datasets.get(dataset)
            if ds is None:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown dataset: {dataset}"})
                return
            if index_in_dataset < 0 or index_in_dataset >= ds.size:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "index_in_dataset out of range"})
                return
            if ds.record_ids[index_in_dataset] != record_id:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "record_id does not match index_in_dataset"})
                return
            item = ds.read_item(index_in_dataset)
            messages = item.get("messages") or []
            if not isinstance(messages, list):
                messages = []
            assistant_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]
            missing_steps: list[int] = []
            invalid_steps: list[int] = []
            for idx in assistant_indices:
                v = step_labels.get(str(idx))
                if v not in (-1, 0, 1):
                    if v is None:
                        missing_steps.append(idx)
                    else:
                        invalid_steps.append(idx)
            if missing_steps:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "missing step_labels for assistant messages", "missing": missing_steps},
                )
                return
            if invalid_steps:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid step_labels for assistant messages", "invalid": invalid_steps},
                )
                return

        with self.state.lock:
            self.state.store.upsert_annotation(
                dataset=dataset,
                record_id=record_id,
                annotator=annotator,
                username=username,
                index_in_dataset=index_in_dataset,
                data_source=data_source,
                query_index=query_index,
                sample_index=sample_index,
                step_labels=step_labels,
                final_label=final_label,
                final_label_touched=final_label_touched_bool,
                status=status,
                comment=comment,
            )
            if status in ("done", "skipped") and index_in_dataset >= 0:
                self.state.store.set_cursor(dataset=dataset, annotator=annotator, cursor_index=index_in_dataset + 1)

        _json_response(self, HTTPStatus.OK, {"ok": True})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentProcessBench annotation platform (stdlib server).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--annotation_dir",
        default=str(Path("output/annotation_file_diverse_queries")),
        help="Directory containing *.jsonl to annotate.",
    )
    parser.add_argument(
        "--data_dir",
        default=str(Path("annotation_platform/data")),
        help="Directory for sqlite + exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.annotation_dir).resolve()
    data_dir = Path(args.data_dir).resolve()

    if not dataset_dir.exists():
        raise SystemExit(f"annotation_dir not found: {dataset_dir}")

    state = AppState(dataset_dir=dataset_dir, data_dir=data_dir)
    if len(state.datasets) == 0:
        raise SystemExit(f"no datasets found under: {dataset_dir} (expected *.jsonl)")

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    server.state = state  # type: ignore[attr-defined]

    print(f"[annotation] serving on http://{args.host}:{args.port}")
    print(f"[annotation] datasets: {', '.join(sorted(state.datasets))}")
    print(f"[annotation] sqlite: {state.store.db_path}")
    print(f"[annotation] exports: {state.store.exports_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
