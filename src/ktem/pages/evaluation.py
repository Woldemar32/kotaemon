from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from ktem.app import BasePage
from ktem.evaluation import find_default_dataset_path, load_eval_dataset, run_evaluation


CARD_STYLE = """
<style>
.eval-hero {
  border-radius: 18px;
  padding: 22px 26px;
  margin-bottom: 14px;
  background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(16,185,129,.12));
  border: 1px solid rgba(148,163,184,.25);
}
.eval-hero h1 { margin: 0 0 8px; font-size: 28px; }
.eval-hero p { margin: 0; color: var(--body-text-color-subdued); font-size: 15px; }
.eval-summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin: 10px 0 16px;
}
.eval-card {
  border-radius: 14px;
  border: 1px solid rgba(148,163,184,.28);
  padding: 14px 16px;
  background: rgba(255,255,255,.04);
}
.eval-card .label { color: var(--body-text-color-subdued); font-size: 12px; }
.eval-card .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
.eval-ragas-hint {
  color: var(--body-text-color-subdued);
  font-size: 13px;
  margin: 0 0 10px;
}
.eval-ragas-panel {
  max-height: min(70vh, 560px);
  overflow: auto;
  border: 1px solid rgba(148,163,184,.28);
  border-radius: 12px;
  background: rgba(255,255,255,.02);
}
.eval-ragas-table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.eval-ragas-table th,
.eval-ragas-table td {
  padding: 8px 12px;
  border-bottom: 1px solid rgba(148,163,184,.18);
}
.eval-ragas-table thead th {
  position: sticky;
  top: 0;
  z-index: 3;
  background: var(--block-background-fill, #1e1e1e);
  font-weight: 600;
  white-space: nowrap;
  text-align: center;
}
.eval-ragas-table th.row-header,
.eval-ragas-table td.row-header {
  position: sticky;
  left: 0;
  z-index: 2;
  background: var(--block-background-fill, #1e1e1e);
  text-align: left;
  max-width: 200px;
  min-width: 120px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.eval-ragas-table thead th.row-header { z-index: 4; }
.eval-ragas-table td.score {
  text-align: center;
  font-variant-numeric: tabular-nums;
  min-width: 76px;
  white-space: nowrap;
}
.eval-ragas-table td.score-good { color: #10b981; font-weight: 600; }
.eval-ragas-table td.score-mid { color: #f59e0b; font-weight: 600; }
.eval-ragas-table td.score-low { color: #ef4444; font-weight: 600; }
.eval-ragas-table td.score-na {
  color: var(--body-text-color-subdued);
  font-size: 12px;
}
.eval-ragas-empty {
  padding: 24px;
  text-align: center;
  color: var(--body-text-color-subdued);
}
.eval-detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
  gap: 10px;
  margin: 8px 0 4px;
}
.eval-detail-metric {
  border-radius: 10px;
  border: 1px solid rgba(148,163,184,.25);
  padding: 10px 12px;
  background: rgba(255,255,255,.03);
}
.eval-detail-metric .name {
  font-size: 11px;
  color: var(--body-text-color-subdued);
  line-height: 1.3;
}
.eval-detail-metric .val {
  font-size: 22px;
  font-weight: 700;
  margin-top: 4px;
  font-variant-numeric: tabular-nums;
}
.eval-detail-metric .val.score-good { color: #10b981; }
.eval-detail-metric .val.score-mid { color: #f59e0b; }
.eval-detail-metric .val.score-low { color: #ef4444; }
.eval-detail-metric .val.score-na {
  color: var(--body-text-color-subdued);
  font-size: 16px;
  font-weight: 500;
}
.eval-text-block {
  margin-top: 8px;
  border-radius: 10px;
  border: 1px solid rgba(148,163,184,.2);
  padding: 0;
  overflow: hidden;
}
.eval-text-block summary {
  cursor: pointer;
  font-weight: 600;
  font-size: 13px;
  padding: 10px 12px;
  list-style: none;
}
.eval-text-block summary::-webkit-details-marker { display: none; }
.eval-text-block .body {
  padding: 0 12px 12px;
  max-height: 200px;
  overflow: auto;
  font-size: 13px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--body-text-color-subdued);
}
</style>
"""

_RAGAS_META_COLUMNS = {"id", "source_file"}
_RAGAS_TEXT_COLUMNS = {
    "user_input",
    "question",
    "response",
    "answer",
    "retrieved_contexts",
    "contexts",
    "reference",
    "ground_truth",
    "ground_truths",
}
_RAGAS_METRIC_ORDER = [
    "ragas_local_score",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "factual_correctness",
    "factual_correctness(mode=f1)",
    "llm_context_precision_with_reference",
    "context_precision",
    "context_recall",
    "context_entity_recall",
    "semantic_similarity",
    "non_llm_string_similarity",
    "answer_keyword_recall",
    "context_keyword_recall",
]


class EvaluationPage(BasePage):
    """UI for evaluating the Kotaemon RAG chatbot on a curated dataset."""

    def __init__(self, app):
        super().__init__(app)
        self.default_dataset_path = str(find_default_dataset_path() or "rag_eval_dataset.json")
        self.on_building_ui()

    def on_building_ui(self):
        gr.HTML(
            CARD_STYLE
            + """
            <div class="eval-hero">
              <h1>RAG Evaluation</h1>
              <p>
                Runs questions from rag_eval_dataset through the current Kotaemon RAG,
                collects retrieved contexts, and scores quality with local Kotaemon
                evaluator models.
              </p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                self.dataset_path = gr.Textbox(
                    label="Dataset path",
                    value=self.default_dataset_path,
                    placeholder="rag_eval_dataset.json",
                    info="JSON/JSONL with question, ground_truth/reference, and source_file fields.",
                )
            with gr.Column(scale=1, min_width=180):
                self.reload_btn = gr.Button("Reload dataset", variant="secondary")

        self.dataset_status = gr.Markdown()
        with gr.Row():
            self.question_limit = gr.Slider(
                minimum=1,
                maximum=1,
                value=1,
                step=1,
                label="Number of questions to ask the chatbot",
                info="Run a quick smoke test or the full dataset.",
            )
            self.enable_ragas = gr.Checkbox(
                label="Compute RAGAS metrics",
                value=True,
                info="Uses local LLM/embeddings from Kotaemon settings, without an OpenAI API key.",
            )

        self.dataset_preview = gr.DataFrame(
            label="Dataset preview",
            interactive=False,
            wrap=True,
        )

        with gr.Row():
            self.run_btn = gr.Button("Run evaluation", variant="primary", size="lg")
            self.clear_btn = gr.Button("Clear results", variant="secondary")

        self.run_status = gr.Markdown()
        self.summary_html = gr.HTML()

        with gr.Tabs():
            with gr.Tab("RAGAS scores"):
                gr.Markdown(
                    "Numeric metrics only in the scrollable matrix. Long text fields "
                    "(question, answer, contexts) are in **sample detail** below."
                )
                self.ragas_scores_html = gr.HTML(
                    label="Scores matrix",
                    elem_classes=["eval-ragas-matrix"],
                )
                self.ragas_row_select = gr.Dropdown(
                    label="Sample detail",
                    choices=[],
                    value=None,
                    interactive=True,
                    info="Inspect one row: metric cards + expandable text fields.",
                )
                self.ragas_row_detail = gr.HTML()
                self.ragas_state = gr.State([])
            with gr.Tab("Answers & contexts"):
                self.samples_table = gr.DataFrame(
                    label="Generated answers, retrieved chunks and errors",
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Warnings"):
                self.warnings_box = gr.Textbox(
                    label="Warnings / errors",
                    lines=8,
                    interactive=False,
                )

    @staticmethod
    def _empty_outputs(status: str):
        return (
            status,
            gr.update(maximum=1, value=1),
            pd.DataFrame(),
        )

    @staticmethod
    def _empty_ragas_outputs():
        return (
            EvaluationPage._ragas_scores_table_html(pd.DataFrame()),
            gr.update(choices=[], value=None),
            "",
            [],
        )

    @staticmethod
    def _escape(value: Any) -> str:
        return html.escape(str(value), quote=True)

    @staticmethod
    def _format_long_value(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, (list, tuple)):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return "\n\n---\n\n".join(parts)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value).strip()

    @staticmethod
    def _looks_like_metric_column(column: str) -> bool:
        normalized = column.lower().replace(" ", "_")
        metric_tokens = (
            "faithfulness",
            "relevancy",
            "relevance",
            "recall",
            "precision",
            "correctness",
            "similarity",
            "score",
        )
        return any(token in normalized for token in metric_tokens)

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        """Extract a finite numeric score from scalar or RAGAS result-like values.

        Different RAGAS versions/providers may return native numbers, numpy scalars,
        single-value containers, dict/object wrappers, or numeric strings. The old
        Gradio DataFrame displayed many of those values implicitly; the custom HTML
        table must do the coercion explicitly so computed scores do not render as em
        dashes.
        """

        if value is None or isinstance(value, bool):
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if hasattr(value, "item") and not isinstance(value, str):
            try:
                return EvaluationPage._coerce_score(value.item())
            except Exception:
                pass

        if isinstance(value, (int, float)):
            number = float(value)
            return number if math.isfinite(number) else None

        if isinstance(value, dict):
            preferred_keys = (
                "score",
                "value",
                "f1",
                "f1_score",
                "mean",
                "faithfulness",
                "answer_relevancy",
                "context_recall",
                "llm_context_precision_with_reference",
                "factual_correctness",
            )
            for key in preferred_keys:
                if key in value:
                    score = EvaluationPage._coerce_score(value[key])
                    if score is not None:
                        return score
            for nested in value.values():
                score = EvaluationPage._coerce_score(nested)
                if score is not None:
                    return score
            return None

        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return EvaluationPage._coerce_score(value[0])
            return None

        for attr in ("score", "value", "f1", "f1_score"):
            if hasattr(value, attr):
                try:
                    score = EvaluationPage._coerce_score(getattr(value, attr))
                except Exception:
                    score = None
                if score is not None:
                    return score

        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "—", "-"}:
            return None

        match = re.search(r"[-+]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        try:
            number = float(match.group(0).replace(",", "."))
        except ValueError:
            return None
        if "%" in text[max(0, match.end() - 1) : match.end() + 2]:
            number /= 100.0
        return number if math.isfinite(number) else None

    @classmethod
    def _is_text_column(cls, df: pd.DataFrame, column: str) -> bool:
        if column in _RAGAS_META_COLUMNS:
            return False
        if column in _RAGAS_TEXT_COLUMNS:
            return True
        series = df[column].dropna()
        if series.empty:
            return False
        if cls._looks_like_metric_column(column):
            return False
        if any(cls._coerce_score(value) is not None for value in series.head(20)):
            return False
        for value in series.head(5):
            if isinstance(value, (list, dict)):
                return True
            text = cls._format_long_value(value)
            if len(text) > 72:
                return True
        return False

    @classmethod
    def _metric_columns(cls, df: pd.DataFrame) -> list[str]:
        metrics = []
        for column in df.columns:
            if column in _RAGAS_META_COLUMNS or cls._is_text_column(df, column):
                continue
            series = df[column].dropna()
            has_score = any(cls._coerce_score(value) is not None for value in series.head(50))
            if has_score or cls._looks_like_metric_column(column):
                metrics.append(column)
        ordered = [name for name in _RAGAS_METRIC_ORDER if name in metrics]
        ordered += sorted(column for column in metrics if column not in ordered)
        return ordered

    @classmethod
    def _text_columns(cls, df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in df.columns
            if column not in _RAGAS_META_COLUMNS and cls._is_text_column(df, column)
        ]

    @classmethod
    def _score_class(cls, value: Any) -> str:
        number = cls._coerce_score(value)
        if number is None:
            return "score-na"
        if number >= 0.7:
            return "score-good"
        if number >= 0.4:
            return "score-mid"
        return "score-low"

    @classmethod
    def _format_score(cls, value: Any) -> str:
        number = cls._coerce_score(value)
        if number is None:
            return "—"
        return f"{number:.3f}"

    @classmethod
    def _metric_label(cls, column: str) -> str:
        return column.replace("_", " ").title()

    @classmethod
    def _row_label(cls, row: dict[str, Any]) -> str:
        row_id = str(row.get("id", "")).strip() or "?"
        source = str(row.get("source_file", "")).strip()
        if source:
            return f"{row_id} · {source}"
        return row_id

    @classmethod
    def _parse_row_label(cls, label: str | None) -> str:
        if not label:
            return ""
        if " · " in label:
            return label.split(" · ", 1)[0].strip()
        return str(label).strip()

    @classmethod
    def _normalize_ragas_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        normalized = df.copy()
        for column in cls._metric_columns(normalized):
            numeric_values = normalized[column].map(cls._coerce_score)
            if numeric_values.notna().any():
                normalized[column] = numeric_values.round(4)
        return normalized.astype(object).where(pd.notna(normalized), None)

    @classmethod
    def _ragas_scores_table_html(cls, df: pd.DataFrame) -> str:
        if df.empty:
            return (
                '<div class="eval-ragas-panel">'
                '<div class="eval-ragas-empty">No RAGAS scores yet. Run evaluation.</div>'
                "</div>"
            )

        metrics = cls._metric_columns(df)
        if not metrics:
            return (
                '<div class="eval-ragas-panel">'
                '<div class="eval-ragas-empty">No numeric metrics found in RAGAS output.</div>'
                "</div>"
            )

        header = ['<th class="row-header">Sample</th>'] + [
            f"<th>{cls._escape(cls._metric_label(metric))}</th>" for metric in metrics
        ]
        body_rows: list[str] = []
        for _, row in df.iterrows():
            sample_label = cls._escape(cls._row_label(row.to_dict()))
            cells = [f'<td class="row-header" title="{sample_label}">{sample_label}</td>']
            for metric in metrics:
                raw_value = row.get(metric)
                score_text = cls._escape(cls._format_score(raw_value))
                css_class = cls._score_class(raw_value)
                cells.append(f'<td class="score {css_class}">{score_text}</td>')
            body_rows.append(f"<tr>{''.join(cells)}</tr>")

        table = (
            '<table class="eval-ragas-table">'
            f"<thead><tr>{''.join(header)}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
        )
        return f'<div class="eval-ragas-panel">{table}</div>'

    @classmethod
    def _ragas_row_detail_html(cls, row: dict[str, Any] | None) -> str:
        if not row:
            return (
                '<p style="color: var(--body-text-color-subdued); margin: 8px 0;">'
                "Select a sample to see metric cards and text fields."
                "</p>"
            )

        metric_cards: list[str] = []
        for column in cls._metric_columns(pd.DataFrame([row])):
            raw_value = row.get(column)
            score_text = cls._format_score(raw_value)
            css = cls._score_class(raw_value)
            metric_cards.append(
                f'<div class="eval-detail-metric">'
                f'<div class="name">{cls._escape(cls._metric_label(column))}</div>'
                f'<div class="val {css}">{cls._escape(score_text)}</div>'
                "</div>"
            )

        metrics_html = (
            f'<div class="eval-detail-grid">{"".join(metric_cards)}</div>'
            if metric_cards
            else ""
        )

        text_blocks: list[str] = []
        for column in cls._text_columns(pd.DataFrame([row])):
            content = cls._format_long_value(row.get(column))
            if not content:
                continue
            text_blocks.append(
                "<details class='eval-text-block'>"
                f"<summary>{cls._escape(cls._metric_label(column))}</summary>"
                f'<div class="body">{cls._escape(content)}</div>'
                "</details>"
            )

        meta_parts = []
        if row.get("source_file"):
            meta_parts.append(
                f"<strong>Source:</strong> {cls._escape(row.get('source_file'))}"
            )
        meta_html = (
            f'<p style="margin: 0 0 8px; font-size: 13px;">{" · ".join(meta_parts)}</p>'
            if meta_parts
            else ""
        )

        return meta_html + metrics_html + "".join(text_blocks)

    @classmethod
    def _build_ragas_ui(cls, df: pd.DataFrame):
        if df.empty:
            return cls._empty_ragas_outputs()

        normalized = cls._normalize_ragas_df(df)
        records = normalized.to_dict(orient="records")
        choices = [cls._row_label(record) for record in records]
        first_label = choices[0]
        first_row = records[0]

        return (
            cls._ragas_scores_table_html(normalized),
            gr.update(choices=choices, value=first_label),
            cls._ragas_row_detail_html(first_row),
            records,
        )

    @classmethod
    def show_ragas_row_detail(cls, selection: str | None, records: list[dict[str, Any]]):
        if not records:
            return cls._ragas_row_detail_html(None)
        row_id = cls._parse_row_label(selection)
        for record in records:
            if str(record.get("id", "")) == row_id:
                return cls._ragas_row_detail_html(record)
        return cls._ragas_row_detail_html(records[0])

    def load_dataset_preview(self, dataset_path: str):
        try:
            resolved = Path(dataset_path).expanduser().resolve()
            samples = load_eval_dataset(resolved)
            preview = pd.DataFrame(samples).head(10)
            count = len(samples)
            return (
                f"✅ Loaded **{count}** questions from `{resolved}`.",
                gr.update(maximum=max(count, 1), value=min(5, max(count, 1))),
                preview,
            )
        except Exception as exc:
            return self._empty_outputs(f"❌ Dataset load failed: `{exc}`")

    @staticmethod
    def _summary_cards(summary: dict[str, Any]) -> str:
        if not summary:
            return ""

        preferred = [
            "samples_total",
            "samples_ok",
            "samples_failed",
            "avg_latency_sec",
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "factual_correctness",
            "answer_correctness",
            "ragas_local_score",
            "semantic_similarity",
            "non_llm_string_similarity",
            "answer_keyword_recall",
            "context_keyword_recall",
        ]
        keys = [key for key in preferred if key in summary]
        keys += [key for key in summary if key not in keys]

        def label(key: str) -> str:
            return key.replace("_", " ").title()

        cards = "".join(
            f'<div class="eval-card"><div class="label">{label(key)}</div>'
            f'<div class="value">{summary[key]}</div></div>'
            for key in keys
        )
        return f'<div class="eval-summary-grid">{cards}</div>'

    @staticmethod
    def _display_samples(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        columns = [
            "id",
            "status",
            "source_file",
            "indexed_source",
            "question",
            "reference",
            "answer",
            "context_count",
            "top_score",
            "top_context_preview",
            "latency_sec",
            "error",
        ]
        visible_columns = [col for col in columns if col in df.columns]
        visible_columns = [
            col
            for col in visible_columns
            if col != "error" or df[col].fillna("").astype(str).str.len().sum() > 0
        ]
        preview = df[visible_columns].copy()
        for col in ("question", "reference", "answer", "top_context_preview"):
            if col in preview.columns:
                preview[col] = (
                    preview[col]
                    .fillna("")
                    .astype(str)
                    .apply(lambda text: text if len(text) <= 220 else text[:217] + "…")
                )
        return preview.where(pd.notna(preview), "")

    def run_evaluation_ui(
        self,
        dataset_path: str,
        question_limit: float,
        enable_ragas: bool,
        settings: dict,
        user_id: Any,
        progress=gr.Progress(track_tqdm=False),
    ):
        try:
            total = max(1, int(question_limit))

            def on_progress(done: int, all_items: int, desc: str):
                progress((done, all_items), desc=desc)

            progress((0, total), desc="Starting evaluation")
            result = run_evaluation(
                app=self._app,
                settings=settings,
                user_id=user_id,
                dataset_path=dataset_path,
                question_limit=total,
                run_ragas_metrics=enable_ragas,
                progress=on_progress,
            )
            progress((total, total), desc="Done")

            status_icon = "✅" if result.summary.get("samples_failed", 0) == 0 else "⚠️"
            warnings_text = "\n".join(result.warnings)
            if enable_ragas and result.ragas_scores.empty:
                warnings_text = (warnings_text + "\n" if warnings_text else "") + (
                    "RAGAS table is empty. Check that ragas is installed and a local "
                    "LangChain-compatible LLM/embedding model is configured in Kotaemon."
                )

            ragas_html, ragas_dropdown, ragas_detail, ragas_records = self._build_ragas_ui(
                result.ragas_scores
            )

            return (
                f"{status_icon} Evaluation finished: "
                f"{result.summary.get('samples_ok', 0)}/{result.summary.get('samples_total', 0)} samples OK.",
                self._summary_cards(result.summary),
                ragas_html,
                ragas_dropdown,
                ragas_detail,
                ragas_records,
                self._display_samples(result.samples),
                warnings_text,
            )
        except Exception as exc:
            empty_ragas = self._empty_ragas_outputs()
            return (
                f"❌ Evaluation failed: `{exc}`",
                "",
                *empty_ragas,
                pd.DataFrame(),
                str(exc),
            )

    @staticmethod
    def clear_results():
        empty_ragas = EvaluationPage._empty_ragas_outputs()
        return ("", "", *empty_ragas, pd.DataFrame(), "")

    def on_register_events(self):
        self.reload_btn.click(
            self.load_dataset_preview,
            inputs=[self.dataset_path],
            outputs=[self.dataset_status, self.question_limit, self.dataset_preview],
            show_progress="hidden",
        )
        self.ragas_row_select.change(
            self.show_ragas_row_detail,
            inputs=[self.ragas_row_select, self.ragas_state],
            outputs=[self.ragas_row_detail],
            show_progress="hidden",
        )
        self.run_btn.click(
            self.run_evaluation_ui,
            inputs=[
                self.dataset_path,
                self.question_limit,
                self.enable_ragas,
                self._app.settings_state,
                self._app.user_id,
            ],
            outputs=[
                self.run_status,
                self.summary_html,
                self.ragas_scores_html,
                self.ragas_row_select,
                self.ragas_row_detail,
                self.ragas_state,
                self.samples_table,
                self.warnings_box,
            ],
        )
        self.clear_btn.click(
            self.clear_results,
            inputs=[],
            outputs=[
                self.run_status,
                self.summary_html,
                self.ragas_scores_html,
                self.ragas_row_select,
                self.ragas_row_detail,
                self.ragas_state,
                self.samples_table,
                self.warnings_box,
            ],
            show_progress="hidden",
        )

    def _on_app_created(self):
        self._app.app.load(
            self.load_dataset_preview,
            inputs=[self.dataset_path],
            outputs=[self.dataset_status, self.question_limit, self.dataset_preview],
            show_progress="hidden",
        )
