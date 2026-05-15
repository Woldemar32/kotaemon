import logging
import re
import threading
from textwrap import dedent
from typing import Generator

from decouple import config
from ktem.embeddings.manager import embedding_models_manager as embeddings
from ktem.llms.manager import llms
from ktem.reasoning.prompt_optimization import (
    DecomposeQuestionPipeline,
    RewriteQuestionPipeline,
)
from ktem.utils.render import Render
from ktem.utils.visualize_cited import CreateCitationVizPipeline
from plotly.io import to_json

from kotaemon.base import (
    AIMessage,
    BaseComponent,
    Document,
    HumanMessage,
    Node,
    RetrievedDocument,
    SystemMessage,
)
from kotaemon.indices.qa.citation_qa import (
    CONTEXT_RELEVANT_WARNING_SCORE,
    DEFAULT_QA_TEXT_PROMPT,
    AnswerWithContextPipeline,
)
from kotaemon.indices.qa.citation_qa_inline import AnswerWithInlineCitation
from kotaemon.indices.qa.format_context import PrepareEvidencePipeline
from kotaemon.indices.qa.utils import replace_think_tag_with_details
from kotaemon.indices.rankings.llm_trulens import (
    get_llm_relevance_scorer_errors_count,
)
from kotaemon.llms import ChatLLM

from ..utils import SUPPORTED_LANGUAGE_MAP
from .base import BaseReasoning

logger = logging.getLogger(__name__)


NAVIGATION_BOILERPLATE = {
    "Study at the KU",
    "Study offer",
    "Intranet",
    "Library",
    "KU.Campus",
    "ILIAS",
    "Campus map",
}


def approx_token_count(text: str) -> int:
    """Cheap, backend-independent token estimate for budgeting/debugging."""
    return max(1, int(len(str(text or "")) / 4))


def clean_context_for_generation(text: str) -> str:
    """Clean noisy extracted context only for answer generation.

    This deliberately does not modify indexed/stored chunks.
    """
    cleaned = str(text or "")
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", cleaned)  # markdown images
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)  # markdown links
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\[Translate to English:\]|\[Translate to Englisch:\]", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"Translate to English|Translate to Englisch", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"arrow right icon", " ", cleaned, flags=re.I)
    for phrase in NAVIGATION_BOILERPLATE:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _doc_generation_text(doc: RetrievedDocument) -> str:
    metadata = getattr(doc, "metadata", None) or {}
    if "window" in metadata:
        return str(metadata["window"])
    if metadata.get("type") == "table":
        return str(metadata.get("table_origin", doc.text))
    if metadata.get("type") == "chatbot":
        return str(metadata.get("window", doc.text))
    if metadata.get("type") == "image":
        return str(metadata.get("image_origin", doc.text))
    return str(doc.text)


def _truncate_around_relevant_evidence(text: str, question: str, budget_tokens: int) -> str:
    max_chars = max(256, int(budget_tokens * 4))
    if len(text) <= max_chars:
        return text

    lowered = text.lower()
    markers = [
        "the standard length of the program is six semesters",
        "standard length of the program 6 semester",
        "six semesters",
        "6 semester",
        "180 ects",
        "standard length",
    ]
    pos = -1
    for marker in markers:
        pos = lowered.find(marker)
        if pos >= 0:
            break

    if pos < 0:
        # Fall back to question-token overlap.
        q_tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 3]
        positions = [lowered.find(t) for t in q_tokens if lowered.find(t) >= 0]
        pos = min(positions) if positions else 0

    start = max(0, pos - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end].strip()


def build_context_for_generation(
    retrieved_docs: list[RetrievedDocument],
    question: str,
    model_context_window: int,
    answer_max_tokens: int,
    generation_top_k: int,
    clean_context: bool = True,
) -> tuple[list[RetrievedDocument], dict]:
    """Select, clean and budget contexts before the final answer LLM call."""
    reserved_prompt_tokens = max(900, approx_token_count(question) + answer_max_tokens + 500)
    available_context_tokens = max(256, model_context_window - reserved_prompt_tokens)
    selected: list[RetrievedDocument] = []
    cleaned_contexts = []
    used_tokens = 0

    for rank, doc in enumerate(retrieved_docs[: max(1, generation_top_k)], start=1):
        raw_text = _doc_generation_text(doc)
        text = clean_context_for_generation(raw_text) if clean_context else raw_text
        remaining = available_context_tokens - used_tokens
        if remaining <= 0:
            break
        tokens = approx_token_count(text)
        if tokens > remaining:
            text = _truncate_around_relevant_evidence(text, question, remaining)
            tokens = approx_token_count(text)
        if not text:
            continue

        new_doc = RetrievedDocument(
            text=text,
            metadata=dict(getattr(doc, "metadata", {}) or {}),
            doc_id=getattr(doc, "doc_id", ""),
            score=getattr(doc, "score", 0.0),
            retrieval_metadata=dict(getattr(doc, "retrieval_metadata", {}) or {}),
        )
        selected.append(new_doc)
        used_tokens += tokens
        cleaned_contexts.append(
            {
                "rank": rank,
                "source": new_doc.metadata.get("file_name")
                or new_doc.metadata.get("file_path")
                or new_doc.metadata.get("source")
                or "",
                "context_id": str(getattr(new_doc, "doc_id", "")),
                "estimated_tokens": tokens,
                "text_preview": text[:500],
                "full_text": text,
            }
        )

    estimated_context_tokens = sum(item["estimated_tokens"] for item in cleaned_contexts)
    estimated_final_prompt_tokens = reserved_prompt_tokens + estimated_context_tokens
    debug = {
        "model_context_window": model_context_window,
        "answer_max_tokens": answer_max_tokens,
        "reserved_prompt_tokens": reserved_prompt_tokens,
        "available_context_tokens": available_context_tokens,
        "estimated_context_tokens": estimated_context_tokens,
        "estimated_final_prompt_tokens": estimated_final_prompt_tokens,
        "prompt_exceeds_context_window": estimated_final_prompt_tokens > model_context_window,
        "num_retrieved_contexts": len(retrieved_docs),
        "num_contexts_sent_to_llm": len(selected),
        "num_contexts_before_filtering": len(retrieved_docs),
        "num_contexts_after_filtering": len(selected),
        "cleaned_contexts_sent_to_llm": cleaned_contexts,
        "context_cleaning_enabled": clean_context,
        "generation_top_k": generation_top_k,
    }
    return selected, debug


class AddQueryContextPipeline(BaseComponent):

    n_last_interactions: int = 5
    llm: ChatLLM = Node(default_callback=lambda _: llms.get_default())

    def run(self, question: str, history: list) -> Document:
        messages = [
            SystemMessage(
                content="Below is a history of the conversation so far, and a new "
                "question asked by the user that needs to be answered by searching "
                "in a knowledge base.\nYou have access to a Search index "
                "with 100's of documents.\nGenerate a search query based on the "
                "conversation and the new question.\nDo not include cited source "
                "filenames and document names e.g info.txt or doc.pdf in the search "
                "query terms.\nDo not include any text inside [] or <<>> in the "
                "search query terms.\nDo not include any special characters like "
                "'+'.\nIf the question is not in English, rewrite the query in "
                "the language used in the question.\n If the question contains enough "
                "information, return just the number 1\n If it's unnecessary to do "
                "the searching, return just the number 0."
            ),
            HumanMessage(content="How did crypto do last year?"),
            AIMessage(
                content="Summarize Cryptocurrency Market Dynamics from last year"
            ),
            HumanMessage(content="What are my health plans?"),
            AIMessage(content="Show available health plans"),
        ]
        for human, ai in history[-self.n_last_interactions :]:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))

        messages.append(HumanMessage(content=f"Generate search query for: {question}"))

        resp = self.llm(messages).text
        if resp == "0":
            return Document(content="")

        if resp == "1":
            return Document(content=question)

        return Document(content=resp)


class FullQAPipeline(BaseReasoning):
    """Question answering pipeline. Handle from question to answer"""

    class Config:
        allow_extra = True

    # configuration parameters
    trigger_context: int = 150
    use_rewrite: bool = False
    model_context_window: int = 8192
    answer_max_tokens: int = 700
    generation_temperature: float = 0.0
    generation_top_k: int = 3
    enable_dynamic_context_budget: bool = True
    enable_context_cleaning: bool = True
    enable_extract_answer_retry: bool = True

    retrievers: list[BaseComponent]

    evidence_pipeline: PrepareEvidencePipeline = PrepareEvidencePipeline.withx()
    answering_pipeline: AnswerWithContextPipeline
    rewrite_pipeline: RewriteQuestionPipeline | None = None
    create_citation_viz_pipeline: CreateCitationVizPipeline = Node(
        default_callback=lambda _: CreateCitationVizPipeline(
            embedding=embeddings.get_default()
        )
    )
    add_query_context: AddQueryContextPipeline = AddQueryContextPipeline.withx()

    def retrieve(
        self, message: str, history: list
    ) -> tuple[list[RetrievedDocument], list[Document]]:
        """Retrieve the documents based on the message"""
        # if len(message) < self.trigger_context:
        #     # prefer adding context for short user questions, avoid adding context for
        #     # long questions, as they are likely to contain enough information
        #     # plus, avoid the situation where the original message is already too long
        #     # for the model to handle
        #     query = self.add_query_context(message, history).content
        # else:
        #     query = message
        # print(f"Rewritten query: {query}")
        query = None
        if not query:
            # TODO: previously return [], [] because we think this message as something
            # like "Hello", "I need help"...
            query = message

        docs, doc_ids = [], []
        plot_docs = []

        for idx, retriever in enumerate(self.retrievers):
            retriever_node = self._prepare_child(retriever, f"retriever_{idx}")
            retriever_docs = retriever_node(text=query)

            retriever_docs_text = []
            retriever_docs_plot = []

            for doc in retriever_docs:
                if doc.metadata.get("type", "") == "plot":
                    retriever_docs_plot.append(doc)
                else:
                    retriever_docs_text.append(doc)

            for doc in retriever_docs_text:
                if doc.doc_id not in doc_ids:
                    docs.append(doc)
                    doc_ids.append(doc.doc_id)

            plot_docs.extend(retriever_docs_plot)

        info = [
            Document(
                channel="info",
                content=Render.collapsible_with_header(doc, open_collapsible=True),
            )
            for doc in docs
        ] + [
            Document(
                channel="plot",
                content=doc.metadata.get("data", ""),
            )
            for doc in plot_docs
        ]

        return docs, info

    def prepare_mindmap(self, answer) -> Document | None:
        mindmap = answer.metadata["mindmap"]
        if mindmap:
            mindmap_text = mindmap.text
            mindmap_svg = dedent(
                """
                <div class="markmap">
                <script type="text/template">
                ---
                markmap:
                    colorFreezeLevel: 2
                    activeNode:
                        placement: center
                    initialExpandLevel: 4
                    maxWidth: 200
                ---
                {}
                </script>
                </div>
                """
            ).format(mindmap_text)

            mindmap_content = Document(
                channel="info",
                content=Render.collapsible(
                    header="""
                    <i>Mindmap</i>
                    <a href="#" id='mindmap-toggle'>
                        [Expand]</a>
                    <a href="#" id='mindmap-export'>
                        [Export]</a>""",
                    content=mindmap_svg,
                    open=True,
                ),
            )
        else:
            mindmap_content = None

        return mindmap_content

    def prepare_citation_viz(self, answer, question, docs) -> Document | None:
        doc_texts = [doc.text for doc in docs]
        citation_plot = None
        plot_content = None

        if answer.metadata["citation_viz"] and len(docs) > 1:
            try:
                citation_plot = self.create_citation_viz_pipeline(doc_texts, question)
            except Exception as e:
                print("Failed to create citation plot:", e)

            if citation_plot:
                plot = to_json(citation_plot)
                plot_content = Document(channel="plot", content=plot)

        return plot_content

    def show_citations_and_addons(self, answer, docs, question):
        # show the evidence
        with_citation, without_citation = self.answering_pipeline.prepare_citations(
            answer, docs
        )
        mindmap_output = self.prepare_mindmap(answer)
        citation_plot_output = self.prepare_citation_viz(answer, question, docs)

        if not with_citation and not without_citation:
            yield Document(channel="info", content="<h5><b>No evidence found.</b></h5>")
        else:
            # clear the Info panel
            max_llm_rerank_score = max(
                doc.metadata.get("llm_trulens_score", 0.0) for doc in docs
            )
            has_llm_score = any("llm_trulens_score" in doc.metadata for doc in docs)
            # clear previous info
            yield Document(channel="info", content=None)

            # yield mindmap output
            if mindmap_output:
                yield mindmap_output

            # yield citation plot output
            if citation_plot_output:
                yield citation_plot_output

            # yield warning message
            if has_llm_score and max_llm_rerank_score < CONTEXT_RELEVANT_WARNING_SCORE:
                yield Document(
                    channel="info",
                    content=(
                        "<h5>WARNING! Context relevance score is low. "
                        "Double check the model answer for correctness.</h5>"
                    ),
                )

            # show QA score
            qa_score = (
                round(answer.metadata["qa_score"], 2)
                if answer.metadata.get("qa_score")
                else None
            )
            if qa_score:
                yield Document(
                    channel="info",
                    content=f"<h5>Answer confidence: {qa_score}</h5>",
                )

            yield from with_citation
            if without_citation:
                yield from without_citation

    async def ainvoke(  # type: ignore
        self, message: str, conv_id: str, history: list, **kwargs  # type: ignore
    ) -> Document:  # type: ignore
        raise NotImplementedError

    def stream(  # type: ignore
        self, message: str, conv_id: str, history: list, **kwargs  # type: ignore
    ) -> Generator[Document, None, Document]:
        if self.use_rewrite and self.rewrite_pipeline:
            print("Chosen rewrite pipeline", self.rewrite_pipeline)
            message = self.rewrite_pipeline(question=message).text
            print("Rewrite result", message)

        print(f"Retrievers {self.retrievers}")
        # should populate the context
        docs, infos = self.retrieve(message, history)
        print(f"Got {len(docs)} retrieved documents")
        yield from infos

        self._llm_relevance_scorer_enabled = bool(
            self.retrievers
            and getattr(self.retrievers[0], "llm_scorer", None)
            and not config(
                "KOTAEMON_DISABLE_LLM_RELEVANCE_SCORER",
                default=False,
                cast=bool,
            )
        )
        self._llm_relevance_scorer_failed = False
        self._llm_relevance_scorer_errors_count = 0

        generation_docs = docs
        generation_debug = {
            "model_context_window": self.model_context_window,
            "answer_max_tokens": self.answer_max_tokens,
            "num_retrieved_contexts": len(docs),
            "num_contexts_sent_to_llm": len(docs),
            "llm_relevance_scorer_enabled": self._llm_relevance_scorer_enabled,
            "llm_relevance_scorer_failed": self._llm_relevance_scorer_failed,
            "llm_relevance_scorer_errors_count": self._llm_relevance_scorer_errors_count,
        }
        if self.enable_dynamic_context_budget:
            generation_docs, generation_debug = build_context_for_generation(
                docs,
                message,
                self.model_context_window,
                self.answer_max_tokens,
                self.generation_top_k,
                clean_context=self.enable_context_cleaning,
            )
        generation_debug.update(
            {
                "llm_relevance_scorer_enabled": self._llm_relevance_scorer_enabled,
                "llm_relevance_scorer_failed": self._llm_relevance_scorer_failed,
                "llm_relevance_scorer_errors_count": self._llm_relevance_scorer_errors_count,
            }
        )
        self._generation_debug = generation_debug

        evidence_mode, evidence, images = self.evidence_pipeline(generation_docs).content
        retry_context = ""
        if generation_docs:
            retry_doc = generation_docs[0]
            retry_context = clean_context_for_generation(_doc_generation_text(retry_doc))

        def generate_relevant_scores():
            nonlocal docs
            before_errors = get_llm_relevance_scorer_errors_count()
            try:
                docs = self.retrievers[0].generate_relevant_scores(message, docs)
            except Exception as exc:
                self._llm_relevance_scorer_failed = True
                logger.warning(
                    "generate_relevant_scores failed; keeping original retrieved docs: %s",
                    exc,
                )
            finally:
                retriever = self.retrievers[0] if self.retrievers else None
                self._llm_relevance_scorer_enabled = bool(
                    getattr(
                        retriever,
                        "llm_relevance_scorer_enabled",
                        self._llm_relevance_scorer_enabled,
                    )
                )
                self._llm_relevance_scorer_failed = bool(
                    self._llm_relevance_scorer_failed
                    or getattr(retriever, "llm_relevance_scorer_failed", False)
                )
                retriever_errors = getattr(
                    retriever, "llm_relevance_scorer_errors_count", None
                )
                if retriever_errors is None:
                    retriever_errors = max(
                        0,
                        get_llm_relevance_scorer_errors_count() - before_errors,
                    )
                self._llm_relevance_scorer_errors_count = int(retriever_errors or 0)
                self._generation_debug.update(
                    {
                        "llm_relevance_scorer_enabled": self._llm_relevance_scorer_enabled,
                        "llm_relevance_scorer_failed": self._llm_relevance_scorer_failed,
                        "llm_relevance_scorer_errors_count": self._llm_relevance_scorer_errors_count,
                    }
                )

        # generate relevant score using
        if evidence and self.retrievers:
            scoring_thread = threading.Thread(target=generate_relevant_scores)
            scoring_thread.start()
        else:
            scoring_thread = None

        answer = yield from self.answering_pipeline.stream(
            question=message,
            history=history,
            evidence=evidence,
            evidence_mode=evidence_mode,
            images=images,
            conv_id=conv_id,
            retry_context=retry_context,
            **kwargs,
        )
        self._answer_generation_debug = answer.metadata.get("generation_debug", {})

        # check <think> tag from reasoning models
        processed_answer = replace_think_tag_with_details(answer.text)
        if processed_answer != answer.text:
            # clear the chat message and render again
            yield Document(channel="chat", content=None)
            yield Document(channel="chat", content=processed_answer)

        # show the evidence
        if scoring_thread:
            scoring_thread.join()

        yield from self.show_citations_and_addons(answer, docs, message)

        return answer

    @classmethod
    def prepare_pipeline_instance(cls, settings, retrievers):
        return cls(
            retrievers=retrievers,
            rewrite_pipeline=None,
        )

    @classmethod
    def get_pipeline(cls, settings, states, retrievers):
        """Get the reasoning pipeline

        Args:
            settings: the settings for the pipeline
            retrievers: the retrievers to use
        """
        max_context_length_setting = settings.get("reasoning.max_context_length", 32000)

        pipeline = cls.prepare_pipeline_instance(settings, retrievers)

        prefix = f"reasoning.options.{cls.get_info()['id']}"
        llm_name = settings.get(f"{prefix}.llm", None)
        llm = llms.get(llm_name, llms.get_default())

        # prepare evidence pipeline configuration
        evidence_pipeline = pipeline.evidence_pipeline
        evidence_pipeline.max_context_length = max_context_length_setting

        # answering pipeline configuration
        use_inline_citation = settings[f"{prefix}.highlight_citation"] == "inline"

        if use_inline_citation:
            answer_pipeline = pipeline.answering_pipeline = AnswerWithInlineCitation()
        else:
            answer_pipeline = pipeline.answering_pipeline = AnswerWithContextPipeline()

        answer_pipeline.llm = llm
        answer_pipeline.citation_pipeline.llm = llm
        answer_pipeline.n_last_interactions = settings[f"{prefix}.n_last_interactions"]
        answer_pipeline.enable_citation = (
            settings[f"{prefix}.highlight_citation"] != "off"
        )
        answer_pipeline.enable_mindmap = settings[f"{prefix}.create_mindmap"]
        answer_pipeline.enable_citation_viz = settings[f"{prefix}.create_citation_viz"]
        answer_pipeline.use_multimodal = settings[f"{prefix}.use_multimodal"]
        answer_pipeline.system_prompt = settings[f"{prefix}.system_prompt"]
        answer_pipeline.qa_template = settings[f"{prefix}.qa_prompt"]
        answer_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(
            settings["reasoning.lang"], "English"
        )
        pipeline.model_context_window = int(settings.get(f"{prefix}.model_context_window", 8192))
        pipeline.answer_max_tokens = int(settings.get(f"{prefix}.answer_max_tokens", 700))
        pipeline.generation_temperature = float(settings.get(f"{prefix}.generation_temperature", 0.0))
        pipeline.generation_top_k = int(settings.get(f"{prefix}.generation_top_k", 3))
        pipeline.enable_dynamic_context_budget = bool(
            settings.get(f"{prefix}.enable_dynamic_context_budget", True)
        )
        pipeline.enable_context_cleaning = bool(
            settings.get(f"{prefix}.enable_context_cleaning", True)
        )
        pipeline.enable_extract_answer_retry = bool(
            settings.get(f"{prefix}.enable_extract_answer_retry", True)
        )

        answer_pipeline.model_context_window = pipeline.model_context_window
        answer_pipeline.answer_max_tokens = pipeline.answer_max_tokens
        answer_pipeline.generation_temperature = pipeline.generation_temperature
        answer_pipeline.enable_extract_answer_retry = pipeline.enable_extract_answer_retry

        for attr, value in (
            ("temperature", pipeline.generation_temperature),
            ("max_tokens", pipeline.answer_max_tokens),
            ("n_ctx", pipeline.model_context_window),
            ("num_ctx", pipeline.model_context_window),
        ):
            if hasattr(llm, attr):
                try:
                    setattr(llm, attr, value)
                except Exception:
                    logger.warning("Could not set LLM attribute %s=%s", attr, value)

        pipeline.add_query_context.llm = llm
        pipeline.add_query_context.n_last_interactions = settings[
            f"{prefix}.n_last_interactions"
        ]

        pipeline.trigger_context = settings[f"{prefix}.trigger_context"]
        pipeline.use_rewrite = states.get("app", {}).get("regen", False)
        if pipeline.rewrite_pipeline:
            pipeline.rewrite_pipeline.llm = llm
            pipeline.rewrite_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(
                settings["reasoning.lang"], "English"
            )
        return pipeline

    @classmethod
    def get_user_settings(cls) -> dict:
        from ktem.llms.manager import llms

        llm = ""
        choices = [("(default)", "")]
        try:
            choices += [(_, _) for _ in llms.options().keys()]
        except Exception as e:
            logger.exception(f"Failed to get LLM options: {e}")

        return {
            "llm": {
                "name": "Language model",
                "value": llm,
                "component": "dropdown",
                "choices": choices,
                "special_type": "llm",
                "info": (
                    "The language model to use for generating the answer. If None, "
                    "the application default language model will be used."
                ),
            },
            "highlight_citation": {
                "name": "Citation style",
                "value": (
                    "highlight"
                    if not config("USE_LOW_LLM_REQUESTS", default=False, cast=bool)
                    else "off"
                ),
                "component": "radio",
                "choices": [
                    ("citation: highlight", "highlight"),
                    ("citation: inline", "inline"),
                    ("no citation", "off"),
                ],
            },
            "create_mindmap": {
                "name": "Create Mindmap",
                "value": False,
                "component": "checkbox",
            },
            "create_citation_viz": {
                "name": "Create Embeddings Visualization",
                "value": False,
                "component": "checkbox",
            },
            "use_multimodal": {
                "name": "Use Multimodal Input",
                "value": False,
                "component": "checkbox",
            },
            "system_prompt": {
                "name": "System Prompt",
                "value": ("This is a question answering system."),
            },
            "qa_prompt": {
                "name": "QA Prompt (contains {context}, {question}, {lang})",
                "value": DEFAULT_QA_TEXT_PROMPT,
            },
            "n_last_interactions": {
                "name": "Number of interactions to include",
                "value": 5,
                "component": "number",
                "info": "The maximum number of chat interactions to include in the LLM",
            },
            "trigger_context": {
                "name": "Maximum message length for context rewriting",
                "value": 150,
                "component": "number",
                "info": (
                    "The maximum length of the message to trigger context addition. "
                    "Exceeding this length, the message will be used as is."
                ),
            },
            "model_context_window": {
                "name": "Model context window",
                "value": 8192,
                "component": "number",
                "info": "Maximum usable model context window for answer generation. Do not set above backend/model capability (Ollama: num_ctx; llama.cpp: ctx-size/n_ctx; vLLM: max_model_len).",
            },
            "answer_max_tokens": {
                "name": "Answer max tokens",
                "value": 700,
                "component": "number",
            },
            "generation_temperature": {
                "name": "Generation temperature",
                "value": 0.0,
                "component": "number",
            },
            "generation_top_k": {
                "name": "Answer generation top-k contexts",
                "value": 3,
                "component": "number",
            },
            "enable_dynamic_context_budget": {
                "name": "Enable dynamic context budget",
                "value": True,
                "component": "checkbox",
            },
            "enable_context_cleaning": {
                "name": "Enable answer context cleaning",
                "value": True,
                "component": "checkbox",
            },
            "enable_extract_answer_retry": {
                "name": "Enable extractive answer retry",
                "value": True,
                "component": "checkbox",
            },
        }

    @classmethod
    def get_info(cls) -> dict:
        return {
            "id": "simple",
            "name": "Simple QA",
            "description": (
                "Simple RAG-based question answering pipeline. This pipeline can "
                "perform both keyword search and similarity search to retrieve the "
                "context. After that it includes that context to generate the answer."
            ),
        }


class FullDecomposeQAPipeline(FullQAPipeline):
    def answer_sub_questions(
        self, messages: list, conv_id: str, history: list, **kwargs
    ):
        output_str = ""
        for idx, message in enumerate(messages):
            yield Document(
                channel="chat",
                content=f"<br><b>Sub-question {idx + 1}</b>"
                f"<br>{message}<br><b>Answer</b><br>",
            )
            # should populate the context
            docs, infos = self.retrieve(message, history)
            print(f"Got {len(docs)} retrieved documents")

            yield from infos

            evidence_mode, evidence, images = self.evidence_pipeline(docs).content
            answer = yield from self.answering_pipeline.stream(
                question=message,
                history=history,
                evidence=evidence,
                evidence_mode=evidence_mode,
                images=images,
                conv_id=conv_id,
                **kwargs,
            )

            output_str += (
                f"Sub-question {idx + 1}-th: '{message}'\nAnswer: '{answer.text}'\n\n"
            )

        return output_str

    def stream(  # type: ignore
        self, message: str, conv_id: str, history: list, **kwargs  # type: ignore
    ) -> Generator[Document, None, Document]:
        sub_question_answer_output = ""
        if self.rewrite_pipeline:
            print("Chosen rewrite pipeline", self.rewrite_pipeline)
            result = self.rewrite_pipeline(question=message)
            print("Rewrite result", result)
            if isinstance(result, Document):
                message = result.text
            elif (
                isinstance(result, list)
                and len(result) > 0
                and isinstance(result[0], Document)
            ):
                yield Document(
                    channel="chat",
                    content="<h4>Sub questions and their answers</h4>",
                )
                sub_question_answer_output = yield from self.answer_sub_questions(
                    [r.text for r in result], conv_id, history, **kwargs
                )

        yield Document(
            channel="chat",
            content=f"<h4>Main question</h4>{message}<br><b>Answer</b><br>",
        )

        # should populate the context
        docs, infos = self.retrieve(message, history)
        print(f"Got {len(docs)} retrieved documents")
        yield from infos

        evidence_mode, evidence, images = self.evidence_pipeline(docs).content
        answer = yield from self.answering_pipeline.stream(
            question=message,
            history=history,
            evidence=evidence + "\n" + sub_question_answer_output,
            evidence_mode=evidence_mode,
            images=images,
            conv_id=conv_id,
            **kwargs,
        )

        # show the evidence
        with_citation, without_citation = self.answering_pipeline.prepare_citations(
            answer, docs
        )
        if not with_citation and not without_citation:
            yield Document(channel="info", content="<h5><b>No evidence found.</b></h5>")
        else:
            yield Document(channel="info", content=None)
            yield from with_citation
            yield from without_citation

        return answer

    @classmethod
    def get_user_settings(cls) -> dict:
        user_settings = super().get_user_settings()
        user_settings["decompose_prompt"] = {
            "name": "Decompose Prompt",
            "value": DecomposeQuestionPipeline.DECOMPOSE_SYSTEM_PROMPT_TEMPLATE,
        }
        return user_settings

    @classmethod
    def prepare_pipeline_instance(cls, settings, retrievers):
        prefix = f"reasoning.options.{cls.get_info()['id']}"
        pipeline = cls(
            retrievers=retrievers,
            rewrite_pipeline=DecomposeQuestionPipeline(
                prompt_template=settings.get(f"{prefix}.decompose_prompt")
            ),
        )
        return pipeline

    @classmethod
    def get_info(cls) -> dict:
        return {
            "id": "complex",
            "name": "Complex QA",
            "description": (
                "Use multi-step reasoning to decompose a complex question into "
                "multiple sub-questions. This pipeline can "
                "perform both keyword search and similarity search to retrieve the "
                "context. After that it includes that context to generate the answer."
            ),
        }
