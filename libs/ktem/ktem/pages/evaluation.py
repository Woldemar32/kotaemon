from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
from ktem.app import BasePage
from ktem.db.models import Conversation, engine
from kotaemon.indices.rankings.llm_trulens import (
    get_llm_relevance_scorer_errors_count,
)
from sqlmodel import Session, select


METRICS = [
    "source_hit",
    "source_recall",
    "context_reference_coverage",
    "answer_similarity_lexical",
    "latency_ms_avg",
]

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "has", "have", "you", "your", "not", "but", "die", "der", "das", "und",
    "ein", "eine", "ist", "im", "in", "zu", "auf", "mit", "für", "von",
}


class EvaluationPage(BasePage):
    """In-app KU D3B evaluation tab.

    This page intentionally calls ``self._app.chat_page.run_ui_chat_query`` instead
    of a subprocess or separate RAG implementation. Therefore evaluation uses the
    same live chatbot pipeline as the normal Chat UI: same selectors, settings,
    retrieval, evidence preparation, answer LLM call, citations and info panel.
    """

    def __init__(self, app):
        super().__init__(app)
        self.kotaemon_root = Path(__file__).resolve().parents[4]
        self.eval_root = self.kotaemon_root / "ku_d3b_eval"
        self.dataset_dir = self.eval_root / "datasets"
        self.results_dir = self.eval_root / "results"
        self.debug_dir = self.eval_root / "debug"
        self.default_dataset = self.dataset_dir / "ku_d3b_eval_dataset.jsonl"
        self.default_result_dir = self.results_dir / "_last_ui_eval"
        self._ensure_eval_layout()
        self.on_building_ui()

    def _ensure_eval_layout(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / ".gitkeep").touch(exist_ok=True)
        (self.debug_dir / ".gitkeep").touch(exist_ok=True)

        legacy_dataset = self.kotaemon_root.parent / "ku_d3b_eval_pack" / "ku_d3b_eval_dataset.jsonl"
        if not self.default_dataset.exists() and legacy_dataset.exists():
            shutil.copyfile(legacy_dataset, self.default_dataset)

        readme = self.eval_root / "README.md"
        if not readme.exists():
            readme.write_text(
                "# KU D3B Evaluation\n\n"
                "Run evaluation from the chatbot UI Evaluation tab.\n\n"
                "- `datasets/ku_d3b_eval_dataset.jsonl` - default dataset.\n"
                "- `results/_last_ui_eval/` - last UI evaluation output.\n"
                "- `debug/` - reserved for manual debug artifacts.\n",
                encoding="utf-8",
            )

    def on_building_ui(self):
        gr.Markdown("### Evaluation")
        gr.Markdown(
            "Runs the dataset through the same live chatbot pipeline used by the "
            "Chat tab. Results are saved inside `ku_d3b_eval/results/_last_ui_eval/`."
        )

        default_total = self._safe_count_jsonl(self.default_dataset)
        default_limit = min(5, default_total) if default_total else 5
        self.dataset_path = gr.Textbox(
            label="Dataset path (.jsonl)",
            value=str(self.default_dataset),
            placeholder="ku_d3b_eval/datasets/ku_d3b_eval_dataset.jsonl",
        )
        self.question_limit = gr.Slider(
            label="Questions to evaluate",
            minimum=1,
            maximum=max(default_total, default_limit, 1),
            value=max(default_limit, 1),
            step=1,
        )
        self.dataset_info = gr.Markdown(
            self._format_dataset_info(default_total, int(max(default_limit, 1)))
        )
        self.disable_llm_relevance_scorer = gr.Checkbox(
            label="Disable LLM relevance scoring during evaluation",
            value=True,
            info=(
                "Recommended for bulk evaluation: keeps original retrieval order and "
                "skips the extra LLM calls used only for UI relevance scores."
            ),
        )

        with gr.Row():
            self.run_button = gr.Button("Run Evaluation", variant="primary")
            self.export_button = gr.Button("Export Last Results")

        self.status = gr.Markdown("Idle")
        self.metrics_table = gr.Textbox(
            label="Metrics",
            lines=8,
            max_lines=16,
            interactive=False,
        )
        self.output_text = gr.Textbox(
            label="Evaluation log",
            lines=14,
            max_lines=30,
            interactive=False,
        )
        self.last_result_dir = gr.State(value=str(self.default_result_dir))

    def on_register_events(self):
        self.dataset_path.change(
            self.update_dataset_info,
            inputs=[self.dataset_path, self.question_limit],
            outputs=[self.question_limit, self.dataset_info],
            show_progress="minimal",
        )

        chat_page = self._app.chat_page
        self.run_button.click(
            self.run_evaluation,
            inputs=[
                self.dataset_path,
                self.question_limit,
                self._app.settings_state,
                chat_page._reasoning_type,
                chat_page.model_type,
                chat_page.use_mindmap,
                chat_page.citation,
                chat_page.language,
                chat_page.state_chat,
                chat_page._command_state,
                chat_page.chat_control.conversation_id,
                self._app.user_id,
                self.disable_llm_relevance_scorer,
            ]
            + chat_page._indices_input,
            outputs=[
                self.status,
                self.metrics_table,
                self.output_text,
                self.last_result_dir,
            ],
            show_progress="minimal",
        )
        self.export_button.click(
            self.export_results,
            inputs=[self.last_result_dir],
            outputs=[self.status, self.output_text],
            show_progress="minimal",
        )

    def _resolve_dataset(self, dataset_path: str | None) -> Path:
        if dataset_path and dataset_path.strip():
            path = Path(dataset_path.strip()).expanduser()
            if not path.is_absolute():
                candidate = self.kotaemon_root / path
                path = candidate if candidate.exists() else (self.dataset_dir / path)
        else:
            path = self.default_dataset

        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Dataset must be a .jsonl file: {path}")
        return path

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        return rows

    @staticmethod
    def _count_jsonl(path: Path) -> int:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _safe_count_jsonl(self, path: Path) -> int:
        try:
            if path.exists() and path.suffix.lower() == ".jsonl":
                return self._count_jsonl(path)
        except Exception:
            return 0
        return 0

    @staticmethod
    def _format_dataset_info(total: int, limit: int) -> str:
        if total <= 0:
            return "Dataset status: no dataset loaded yet."
        selected = min(max(int(limit), 1), total)
        return f"Dataset status: {total} questions available. This run will evaluate {selected} question(s)."

    def update_dataset_info(self, dataset_path: str | None, question_limit: int | float | None):
        try:
            dataset = self._resolve_dataset(dataset_path)
            total = self._count_jsonl(dataset)
        except Exception as exc:
            return gr.update(), f"Dataset status: {exc}"
        if total < 1:
            return gr.update(maximum=1, value=1), f"Dataset status: empty dataset: {dataset}"
        selected = min(max(int(question_limit or min(5, total)), 1), total)
        return gr.update(maximum=max(total, 1), value=selected), self._format_dataset_info(total, selected)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"[^\wäöüßа-яё]+", " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _meaningful_tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in cls._normalize_text(text).split()
            if len(token) > 2 and token not in STOPWORDS
        }

    @classmethod
    def _token_f1(cls, a: str, b: str) -> float:
        ta, tb = cls._meaningful_tokens(a), cls._meaningful_tokens(b)
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        if inter == 0:
            return 0.0
        precision = inter / len(ta)
        recall = inter / len(tb)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _contains_source(candidate: str, expected_file: str) -> bool:
        candidate_norm = str(candidate or "").replace("\\", "/").lower()
        expected_norm = str(expected_file or "").replace("\\", "/").lower()
        candidate_name = Path(candidate_norm).name
        expected_name = Path(expected_norm).name
        return (
            bool(candidate_norm and expected_norm)
            and (
                expected_norm == candidate_norm
                or expected_norm in candidate_norm
                or expected_name == candidate_name
                or expected_name in candidate_norm
            )
        )

    def _source_metrics(self, sample: dict[str, Any], pred: dict[str, Any]) -> tuple[float, float]:
        expected = sample.get("expected_source_files", []) or []
        if not expected:
            return 1.0, 1.0
        got = [str(x) for x in pred.get("retrieved_sources", []) or []]
        hits = sum(1 for exp in expected if any(self._contains_source(g, exp) for g in got))
        return (1.0 if hits > 0 else 0.0), hits / max(len(expected), 1)

    def _context_reference_coverage(self, sample: dict[str, Any], pred: dict[str, Any]) -> float:
        refs = sample.get("reference_contexts", []) or []
        if not refs:
            return 1.0
        retrieved_text = "\n".join(str(x) for x in pred.get("retrieved_contexts", []) or [])
        retrieved_tokens = self._meaningful_tokens(retrieved_text)
        if not retrieved_tokens:
            return 0.0
        scores = []
        for ref in refs:
            ref_tokens = self._meaningful_tokens(str(ref))
            if not ref_tokens:
                continue
            scores.append(len(ref_tokens & retrieved_tokens) / len(ref_tokens))
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def _evaluate_predictions(
        self, dataset: list[dict[str, Any]], predictions: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        pred_by_id = {p.get("id"): p for p in predictions}
        rows = []
        for sample in dataset:
            pred = pred_by_id.get(sample["id"], {})
            sh, sr = self._source_metrics(sample, pred)
            coverage = self._context_reference_coverage(sample, pred)
            ans = self._token_f1(str(pred.get("response", "")), str(sample.get("reference", "")))
            rows.append(
                {
                    "id": sample["id"],
                    "question_type": sample.get("question_type"),
                    "language": sample.get("language"),
                    "status": pred.get("status", "ok"),
                    "error": pred.get("error", ""),
                    "source_hit": sh,
                    "source_recall": sr,
                    "context_reference_coverage": coverage,
                    "answer_similarity_lexical": round(ans, 4),
                    "latency_ms": pred.get("latency_ms"),
                    "response": pred.get("response", ""),
                }
            )

        def avg(values: Iterable[Any]) -> float | None:
            vals = [float(v) for v in values if isinstance(v, (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else None

        summary = {
            "n_dataset": len(dataset),
            "n_predictions": len(predictions),
            "n_successful_predictions": sum(1 for p in predictions if p.get("status", "ok") != "error"),
            "n_failed_predictions": sum(1 for p in predictions if p.get("status") == "error"),
            "overall": {
                "source_hit": avg(r["source_hit"] for r in rows),
                "source_recall": avg(r["source_recall"] for r in rows),
                "context_reference_coverage": avg(r["context_reference_coverage"] for r in rows),
                "answer_similarity_lexical": avg(r["answer_similarity_lexical"] for r in rows),
                "latency_ms_avg": avg(r["latency_ms"] for r in rows),
            },
        }
        return summary, rows

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _format_metrics(summary: dict[str, Any]) -> str:
        overall = summary.get("overall", {}) or {}
        headers = ["metric", "value"]
        table = [" | ".join(headers), "--- | ---"]
        for metric in METRICS:
            table.append(f"{metric} | {overall.get(metric, '')}")
        table.append(f"successful | {summary.get('n_successful_predictions')}")
        table.append(f"failed | {summary.get('n_failed_predictions')}")
        return "\n".join(table)

    @staticmethod
    def _tail(lines: list[str], limit: int = 120) -> str:
        return "\n".join(lines[-limit:])

    def run_evaluation(
        self,
        dataset_path: str | None,
        question_limit: int | float | None,
        settings: dict,
        reasoning_type: str,
        llm_type: str,
        use_mind_map: bool | str,
        use_citation: str,
        language: str,
        chat_state: dict,
        command_state: str | None,
        conversation_id: str | None,
        user_id: int | None,
        disable_llm_relevance_scorer: bool | str | None,
        *selecteds,
    ):
        try:
            dataset_file = self._resolve_dataset(dataset_path)
            full_dataset = self._read_jsonl(dataset_file)
            if not full_dataset:
                raise ValueError(f"Dataset is empty: {dataset_file}")
            limit = min(max(int(question_limit or len(full_dataset)), 1), len(full_dataset))
            dataset = full_dataset[:limit]
        except Exception as exc:
            yield f"❌ {exc}", "", str(exc), str(self.default_result_dir)
            return

        if self.default_result_dir.exists():
            shutil.rmtree(self.default_result_dir)
        debug_prompt_dir = self.default_result_dir / "debug_prompts"
        debug_prompt_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(disable_llm_relevance_scorer, str):
            disable_llm_relevance_scorer = disable_llm_relevance_scorer.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        else:
            disable_llm_relevance_scorer = bool(disable_llm_relevance_scorer)
        settings = deepcopy(settings or {})
        if disable_llm_relevance_scorer:
            for key in list(settings.keys()):
                if key.endswith(".use_llm_reranking"):
                    settings[key] = False

        predictions: list[dict[str, Any]] = []
        log_lines = [
            "Live UI evaluation started.",
            f"Dataset: {dataset_file}",
            f"Samples selected: {len(dataset)} of {len(full_dataset)}",
            f"Results directory: {self.default_result_dir}",
            "Evaluation uses ChatPage.run_ui_chat_query (same live Chat UI pipeline).",
            (
                "LLM relevance scoring for the Information panel: disabled."
                if disable_llm_relevance_scorer
                else "LLM relevance scoring for the Information panel: enabled."
            ),
        ]
        yield f"⏳ Running... 0/{len(dataset)}", "", self._tail(log_lines), str(self.default_result_dir)

        chat_page = self._app.chat_page
        selecteds = self._resolve_evaluation_selecteds(
            selecteds, conversation_id, chat_page, log_lines
        )
        success_count = 0
        failed_count = 0
        scorer_parse_failures_count = 0
        scorer_enabled_seen = False
        scorer_errors_before = get_llm_relevance_scorer_errors_count()
        for idx, sample in enumerate(dataset, start=1):
            sample_id = str(sample.get("id", f"sample_{idx}"))
            question = str(sample.get("user_input", ""))
            started = time.perf_counter()
            try:
                result = chat_page.run_ui_chat_query(
                    question,
                    [],
                    f"eval-{sample_id}",
                    settings,
                    reasoning_type,
                    llm_type,
                    use_mind_map,
                    use_citation,
                    language,
                    deepcopy(chat_state or {"app": {"regen": False}}),
                    command_state,
                    user_id if user_id is not None else -1,
                    *selecteds,
                    debug=True,
                    sample_id=sample_id,
                    debug_mode="evaluation_ui",
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                debug_payload = result.get("debug") or {}
                debug_payload["latency_ms"] = latency_ms
                debug_payload["id"] = sample_id
                debug_payload["mode"] = "evaluation_ui"
                debug_payload["reference"] = sample.get("reference", "")
                debug_payload["reference_contexts"] = sample.get("reference_contexts", []) or []
                debug_payload["expected_source_files"] = sample.get("expected_source_files", []) or []
                debug_payload["llm_relevance_scorer_enabled"] = bool(
                    debug_payload.get("llm_relevance_scorer_enabled", False)
                )
                debug_payload["llm_relevance_scorer_failed"] = bool(
                    debug_payload.get("llm_relevance_scorer_failed", False)
                )
                debug_payload["llm_relevance_scorer_errors_count"] = int(
                    debug_payload.get("llm_relevance_scorer_errors_count", 0) or 0
                )
                scorer_enabled_seen = (
                    scorer_enabled_seen
                    or debug_payload["llm_relevance_scorer_enabled"]
                )
                scorer_parse_failures_count += debug_payload[
                    "llm_relevance_scorer_errors_count"
                ]

                retrieved_items = debug_payload.get("retrieved_contexts_available", []) or []
                pred = {
                    "id": sample_id,
                    "user_input": question,
                    "response": result.get("response", ""),
                    "retrieved_contexts": [item.get("full_text", "") for item in retrieved_items],
                    "retrieved_context_ids": [item.get("context_id", "") for item in retrieved_items],
                    "retrieved_sources": [item.get("source", "") for item in retrieved_items],
                    "latency_ms": latency_ms,
                    "status": "ok",
                }
                if not pred["retrieved_contexts"]:
                    warning = (
                        "No retrieved contexts were captured. This means evaluation "
                        "is not receiving retrieval output correctly."
                    )
                    debug_payload.setdefault("warnings", []).append(warning)
                    log_lines.append(f"WARNING {sample_id}: {warning}")
                if not pred["retrieved_sources"]:
                    warning = "Evaluation UI has empty retrieved_sources."
                    debug_payload.setdefault("warnings", []).append(warning)
                    log_lines.append(f"WARNING {sample_id}: {warning}")
                success_count += 1
                log_lines.append(f"[{idx}/{len(dataset)}] {sample_id} OK latency={latency_ms}ms")
                if debug_payload["llm_relevance_scorer_errors_count"]:
                    log_lines.append(
                        f"WARNING {sample_id}: LLM relevance scorer parse failures="
                        f"{debug_payload['llm_relevance_scorer_errors_count']}"
                    )
                for warning in debug_payload.get("warnings", []) or []:
                    log_lines.append(f"WARNING {sample_id}: {warning}")
            except Exception as exc:
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                pred = {
                    "id": sample_id,
                    "user_input": question,
                    "response": "",
                    "retrieved_contexts": [],
                    "retrieved_context_ids": [],
                    "retrieved_sources": [],
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                debug_payload = {
                    "id": sample_id,
                    "mode": "evaluation_ui",
                    "question": question,
                    "reference": sample.get("reference", ""),
                    "reference_contexts": sample.get("reference_contexts", []) or [],
                    "expected_source_files": sample.get("expected_source_files", []) or [],
                    "error": pred["error"],
                    "latency_ms": latency_ms,
                }
                failed_count += 1
                log_lines.append(f"[{idx}/{len(dataset)}] {sample_id} ERROR {pred['error']}")

            predictions.append(pred)
            with (debug_prompt_dir / f"{sample_id}.json").open("w", encoding="utf-8") as f:
                json.dump(debug_payload, f, ensure_ascii=False, indent=2)
            self._write_jsonl(self.default_result_dir / "predictions.jsonl", predictions)
            yield (
                f"⏳ Running... {idx}/{len(dataset)} | successful={success_count} failed={failed_count}",
                "",
                self._tail(log_lines),
                str(self.default_result_dir),
            )

        summary, per_sample = self._evaluate_predictions(dataset, predictions)
        scorer_parse_failures_count = max(
            scorer_parse_failures_count,
            get_llm_relevance_scorer_errors_count() - scorer_errors_before,
        )
        summary["llm_relevance_scorer_enabled"] = scorer_enabled_seen
        summary["llm_relevance_scorer_errors_count"] = scorer_parse_failures_count
        with (self.default_result_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self._write_csv(self.default_result_dir / "per_sample_metrics.csv", per_sample)
        metrics_table = self._format_metrics(summary)
        log_lines.append("Evaluation completed.")
        log_lines.append(
            f"LLM relevance scorer parse failures during evaluation: {scorer_parse_failures_count}"
        )
        log_lines.append(f"Summary written: {self.default_result_dir / 'summary.json'}")
        yield (
            f"✅ Completed {len(dataset)}/{len(dataset)} | successful={success_count} failed={failed_count}",
            metrics_table,
            self._tail(log_lines),
            str(self.default_result_dir),
        )

    def export_results(self, last_result_dir: str | None):
        if not last_result_dir:
            return "❌ No evaluation run to export yet.", "Run evaluation first."
        src = Path(last_result_dir)
        if not src.exists():
            return f"❌ Result directory does not exist: {src}", ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = self.results_dir / f"eval_{timestamp}"
        shutil.copytree(src, dst)
        return f"✅ Exported: {dst}", f"Exported results to: {dst}"

    def _resolve_evaluation_selecteds(
        self,
        selecteds: tuple[Any, ...],
        conversation_id: str | None,
        chat_page: Any,
        log_lines: list[str],
    ) -> tuple[Any, ...]:
        """Use the same file/index state as Chat when Evaluation inputs are empty.

        Gradio passes the current file selector components into Evaluation. If the
        Evaluation tab is opened after a page refresh, those components can still be
        in their default disabled state even though the active conversation has a
        saved Chat-tab selection. In that case, recover the saved state from the
        conversation record instead of running a no-context evaluation.
        """
        try:
            described = chat_page._describe_live_selected_state(selecteds)
            if described.get("selected_file_ids"):
                return selecteds
        except Exception:
            pass

        if not conversation_id:
            log_lines.append(
                "WARNING evaluation: no selected files in UI state and no active conversation_id to recover from."
            )
            fallback = self._select_all_indexed_files_for_evaluation(selecteds, log_lines)
            return fallback or selecteds

        try:
            with Session(engine) as session:
                row = session.exec(
                    select(Conversation).where(Conversation.id == conversation_id)
                ).one_or_none()
            saved_selected = ((row.data_source or {}).get("selected", {}) if row else {})
        except Exception as exc:
            log_lines.append(
                f"WARNING evaluation: could not read conversation selected state: {exc}"
            )
            fallback = self._select_all_indexed_files_for_evaluation(selecteds, log_lines)
            return fallback or selecteds

        if not saved_selected:
            log_lines.append(
                "WARNING evaluation: active conversation has no saved selected index/file state."
            )
            fallback = self._select_all_indexed_files_for_evaluation(selecteds, log_lines)
            return fallback or selecteds

        resolved = list(selecteds)
        for index in self._app.index_manager.indices:
            saved = saved_selected.get(str(index.id))
            if saved is None:
                continue
            if isinstance(index.selector, int):
                while len(resolved) <= index.selector:
                    resolved.append(None)
                resolved[index.selector] = saved
            elif isinstance(index.selector, tuple):
                saved_list = saved if isinstance(saved, list) else [saved]
                for offset, selector_idx in enumerate(index.selector):
                    while len(resolved) <= selector_idx:
                        resolved.append(None)
                    resolved[selector_idx] = (
                        saved_list[offset] if offset < len(saved_list) else None
                    )

        recovered = tuple(resolved)
        try:
            described = chat_page._describe_live_selected_state(recovered)
            if not described.get("selected_file_ids"):
                log_lines.append(
                    "WARNING evaluation: recovered conversation state still selected no files."
                )
                fallback = self._select_all_indexed_files_for_evaluation(
                    recovered, log_lines
                )
                return fallback or recovered
        except Exception:
            pass

        log_lines.append(
            "Evaluation recovered selected index/file state from the active Chat conversation."
        )
        return recovered

    def _select_all_indexed_files_for_evaluation(
        self,
        selecteds: tuple[Any, ...],
        log_lines: list[str],
    ) -> tuple[Any, ...] | None:
        """Last-resort evaluation fallback: select indexed files directly.

        This is still the normal live retriever/QA pipeline. It only supplies the
        same file-selection input the Chat tab would have supplied if "Search All"
        were active. This prevents Evaluation from silently running with
        ["disabled", [], -1] and therefore no retrievers.
        """
        resolved = list(selecteds)
        changed = False

        for index in self._app.index_manager.indices:
            source_cls = getattr(index, "_resources", {}).get("Source")
            if source_cls is None or index.selector is None:
                continue

            try:
                with Session(engine) as session:
                    rows = session.exec(select(source_cls)).all()
            except Exception as exc:
                log_lines.append(
                    f"WARNING evaluation: could not inspect sources for index {getattr(index, 'id', '?')}: {exc}"
                )
                continue

            file_ids = [str(row.id) for row in rows if getattr(row, "id", None)]
            if not file_ids:
                continue

            # Prefer an existing owner id from the Source table for private indexes.
            owner = next(
                (
                    str(getattr(row, "user"))
                    for row in rows
                    if getattr(row, "user", None) not in (None, "", -1)
                ),
                "default",
            )
            selector_value = ["select", file_ids, owner]

            if isinstance(index.selector, int):
                while len(resolved) <= index.selector:
                    resolved.append(None)
                resolved[index.selector] = selector_value
                changed = True
            elif isinstance(index.selector, tuple):
                for offset, selector_idx in enumerate(index.selector):
                    while len(resolved) <= selector_idx:
                        resolved.append(None)
                    resolved[selector_idx] = selector_value[offset] if offset < 3 else None
                changed = True

            log_lines.append(
                f"Evaluation auto-selected {len(file_ids)} indexed file(s) for index {getattr(index, 'name', index.id)}."
            )

        return tuple(resolved) if changed else None
