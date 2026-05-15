import asyncio
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from decouple import config
from ktem.app import BasePage
from ktem.components import reasonings
from ktem.db.models import Conversation, engine
from ktem.index.file.ui import File
from ktem.reasoning.prompt_optimization.mindmap import MINDMAP_HTML_EXPORT_TEMPLATE
from ktem.reasoning.prompt_optimization.suggest_conversation_name import (
    SuggestConvNamePipeline,
)
from ktem.reasoning.prompt_optimization.suggest_followup_chat import (
    SuggestFollowupQuesPipeline,
)
from plotly.io import from_json
from sqlmodel import Session, select
from theflow.settings import settings as flowsettings
from theflow.utils.modules import import_dotted_string

from kotaemon.base import Document
from kotaemon.indices.ingests.files import KH_DEFAULT_FILE_EXTRACTORS
from kotaemon.indices.qa.utils import strip_think_tag
from sqlalchemy import select as sa_select

from ...utils import SUPPORTED_LANGUAGE_MAP, get_file_names_regex, get_urls
from ...utils.commands import WEB_SEARCH_COMMAND
from ...utils.hf_papers import get_recommended_papers
from ...utils.rate_limit import check_rate_limit
from .chat_panel import ChatPanel
from .chat_suggestion import ChatSuggestion
from .common import STATE
from .control import ConversationControl
from .demo_hint import HintPage
from .paper_list import PaperListPage
from .report import ReportIssue

KH_DEMO_MODE = getattr(flowsettings, "KH_DEMO_MODE", False)
KH_SSO_ENABLED = getattr(flowsettings, "KH_SSO_ENABLED", False)
KH_WEB_SEARCH_BACKEND = getattr(flowsettings, "KH_WEB_SEARCH_BACKEND", None)
WebSearch = None
if KH_WEB_SEARCH_BACKEND:
    try:
        WebSearch = import_dotted_string(KH_WEB_SEARCH_BACKEND, safe=False)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {KH_WEB_SEARCH_BACKEND}: {e}")

REASONING_LIMITS = 2 if KH_DEMO_MODE else 10
DEFAULT_SETTING = "(default)"
INFO_PANEL_SCALES = {True: 8, False: 4}
DEFAULT_QUESTION = (
    "What is the summary of this document?"
    if not KH_DEMO_MODE
    else "What is the summary of this paper?"
)

chat_input_focus_js = """
function() {
    let chatInput = document.querySelector("#chat-input textarea");
    chatInput.focus();
}
"""

quick_urls_submit_js = """
function() {
    let urlInput = document.querySelector("#quick-url-demo textarea");
    console.log("URL input:", urlInput);
    urlInput.dispatchEvent(new KeyboardEvent('keypress', {'key': 'Enter'}));
}
"""

recommended_papers_js = """
function() {
    // Get all links and attach click event
    var links = document.querySelectorAll("#related-papers a");

    function submitPaper(event) {
        event.preventDefault();
        var target = event.currentTarget;
        var url = target.getAttribute("href");
        console.log("URL:", url);

        let newChatButton = document.querySelector("#new-conv-button");
        newChatButton.click();

        setTimeout(() => {
            let urlInput = document.querySelector("#quick-url-demo textarea");
            // Fill the URL input
            urlInput.value = url;
            urlInput.dispatchEvent(new Event("input", { bubbles: true }));
            urlInput.dispatchEvent(new KeyboardEvent('keypress', {'key': 'Enter'}));
            }, 500
        );
    }

    for (var i = 0; i < links.length; i++) {
        links[i].onclick = submitPaper;
    }
}
"""

clear_bot_message_selection_js = """
function() {
    var bot_messages = document.querySelectorAll(
        "div#main-chat-bot div.message-row.bot-row"
    );
    bot_messages.forEach(message => {
        message.classList.remove("text_selection");
    });
}
"""

pdfview_js = """
function() {
    setTimeout(fullTextSearch(), 100);

    // Get all links and attach click event
    var links = document.getElementsByClassName("pdf-link");
    for (var i = 0; i < links.length; i++) {
        links[i].onclick = openModal;
    }

    // Get all citation links and attach click event
    var links = document.querySelectorAll("a.citation");
    for (var i = 0; i < links.length; i++) {
        links[i].onclick = scrollToCitation;
    }

    var markmap_div = document.querySelector("div.markmap");
    var mindmap_el_script = document.querySelector('div.markmap script');

    if (mindmap_el_script) {
        markmap_div_html = markmap_div.outerHTML;
    }

    // render the mindmap if the script tag is present
    if (mindmap_el_script) {
        markmap.autoLoader.renderAll();
    }

    setTimeout(() => {
        var mindmap_el = document.querySelector('svg.markmap');

        var text_nodes = document.querySelectorAll("svg.markmap div");
        for (var i = 0; i < text_nodes.length; i++) {
            text_nodes[i].onclick = fillChatInput;
        }

        if (mindmap_el) {
            function on_svg_export(event) {
                html = "{html_template}";
                html = html.replace("{markmap_div}", markmap_div_html);
                spawnDocument(html, {window: "width=1000,height=1000"});
            }

            var link = document.getElementById("mindmap-toggle");
            if (link) {
                link.onclick = function(event) {
                    event.preventDefault(); // Prevent the default link behavior
                    var div = document.querySelector("div.markmap");
                    if (div) {
                        var currentHeight = div.style.height;
                        if (currentHeight === '400px' || (currentHeight === '')) {
                            div.style.height = '650px';
                        } else {
                            div.style.height = '400px'
                        }
                    }
                };
            }

            if (markmap_div_html) {
                var link = document.getElementById("mindmap-export");
                if (link) {
                    link.addEventListener('click', on_svg_export);
                }
            }
        }
    }, 250);

    return [links.length]
}
""".replace(
    "{html_template}",
    MINDMAP_HTML_EXPORT_TEMPLATE.replace("\n", "").replace('"', '\\"'),
)

fetch_api_key_js = """
function(_, __) {
    api_key = getStorage('google_api_key', '');
    console.log('session API key:', api_key);
    return [api_key, _];
}
"""


class ChatPage(BasePage):
    def __init__(self, app):
        self._app = app
        self._indices_input = []

        self.on_building_ui()

        self._preview_links = gr.State(value=None)
        self._reasoning_type = gr.State(value=None)
        self._conversation_renamed = gr.State(value=False)
        self._use_suggestion = gr.State(
            value=getattr(flowsettings, "KH_FEATURE_CHAT_SUGGESTION", False)
        )
        self._info_panel_expanded = gr.State(value=True)
        self._command_state = gr.State(value=None)
        self._user_api_key = gr.Text(value="", visible=False)

    def on_building_ui(self):
        with gr.Row():
            self.state_chat = gr.State(STATE)
            self.state_retrieval_history = gr.State([])
            self.state_plot_history = gr.State([])
            self.state_plot_panel = gr.State(None)
            self.first_selector_choices = gr.State(None)

            with gr.Column(scale=1, elem_id="conv-settings-panel") as self.conv_column:
                self.chat_control = ConversationControl(self._app)

                for index_id, index in enumerate(self._app.index_manager.indices):
                    index.selector = None
                    index_ui = index.get_selector_component_ui()
                    if not index_ui:
                        # the index doesn't have a selector UI component
                        continue

                    index_ui.unrender()  # need to rerender later within Accordion
                    is_first_index = index_id == 0
                    index_name = index.name

                    if KH_DEMO_MODE and is_first_index:
                        index_name = "Select from Paper Collection"

                    with gr.Accordion(
                        label=index_name,
                        open=is_first_index,
                        elem_id=f"index-{index_id}",
                    ):
                        index_ui.render()
                        gr_index = index_ui.as_gradio_component()

                        # get the file selector choices for the first index
                        if index_id == 0:
                            self.first_selector_choices = index_ui.selector_choices
                            self.first_indexing_url_fn = None

                        if gr_index:
                            if isinstance(gr_index, list):
                                index.selector = tuple(
                                    range(
                                        len(self._indices_input),
                                        len(self._indices_input) + len(gr_index),
                                    )
                                )
                                index.default_selector = index_ui.default()
                                self._indices_input.extend(gr_index)
                            else:
                                index.selector = len(self._indices_input)
                                index.default_selector = index_ui.default()
                                self._indices_input.append(gr_index)
                        setattr(self, f"_index_{index.id}", index_ui)

                self.chat_suggestion = ChatSuggestion(self._app)

                if len(self._app.index_manager.indices) > 0:
                    quick_upload_label = (
                        "Quick Upload" if not KH_DEMO_MODE else "Or input new paper URL"
                    )

                    with gr.Accordion(label=quick_upload_label) as _:
                        self.quick_file_upload_status = gr.Markdown()
                        if not KH_DEMO_MODE:
                            self.quick_file_upload = File(
                                file_types=list(KH_DEFAULT_FILE_EXTRACTORS.keys()),
                                file_count="multiple",
                                container=True,
                                show_label=False,
                                elem_id="quick-file",
                            )
                        self.quick_urls = gr.Textbox(
                            placeholder=(
                                "Or paste URLs"
                                if not KH_DEMO_MODE
                                else "Paste Arxiv URLs\n(https://arxiv.org/abs/xxx)"
                            ),
                            lines=1,
                            container=False,
                            show_label=False,
                            elem_id=(
                                "quick-url" if not KH_DEMO_MODE else "quick-url-demo"
                            ),
                        )

                if not KH_DEMO_MODE:
                    self.report_issue = ReportIssue(self._app)
                else:
                    with gr.Accordion(label="Related papers", open=False):
                        self.related_papers = gr.Markdown(elem_id="related-papers")

                    self.hint_page = HintPage(self._app)

            with gr.Column(scale=6, elem_id="chat-area"):
                if KH_DEMO_MODE:
                    self.paper_list = PaperListPage(self._app)

                self.chat_panel = ChatPanel(self._app)

                with gr.Accordion(
                    label="Chat settings",
                    elem_id="chat-settings-expand",
                    open=False,
                    visible=not KH_DEMO_MODE,
                ) as self.chat_settings:
                    with gr.Row(elem_id="quick-setting-labels"):
                        gr.HTML("Reasoning method")
                        gr.HTML(
                            "Model", visible=not KH_DEMO_MODE and not KH_SSO_ENABLED
                        )
                        gr.HTML("Language")

                    with gr.Row():
                        reasoning_setting = (
                            self._app.default_settings.reasoning.settings["use"]
                        )
                        model_setting = self._app.default_settings.reasoning.options[
                            "simple"
                        ].settings["llm"]
                        language_setting = (
                            self._app.default_settings.reasoning.settings["lang"]
                        )
                        citation_setting = self._app.default_settings.reasoning.options[
                            "simple"
                        ].settings["highlight_citation"]

                        self.reasoning_type = gr.Dropdown(
                            choices=reasoning_setting.choices[:REASONING_LIMITS],
                            value=reasoning_setting.value,
                            container=False,
                            show_label=False,
                        )
                        self.model_type = gr.Dropdown(
                            choices=model_setting.choices,
                            value=model_setting.value,
                            container=False,
                            show_label=False,
                            visible=not KH_DEMO_MODE and not KH_SSO_ENABLED,
                        )
                        self.language = gr.Dropdown(
                            choices=language_setting.choices,
                            value=language_setting.value,
                            container=False,
                            show_label=False,
                        )

                        self.citation = gr.Dropdown(
                            choices=citation_setting.choices,
                            value=citation_setting.value,
                            container=False,
                            show_label=False,
                            interactive=True,
                            elem_id="citation-dropdown",
                        )

                        if not config("USE_LOW_LLM_REQUESTS", default=False, cast=bool):
                            self.use_mindmap = gr.State(value=True)
                            self.use_mindmap_check = gr.Checkbox(
                                label="Mindmap (on)",
                                container=False,
                                elem_id="use-mindmap-checkbox",
                                value=True,
                            )
                        else:
                            self.use_mindmap = gr.State(value=False)
                            self.use_mindmap_check = gr.Checkbox(
                                label="Mindmap (off)",
                                container=False,
                                elem_id="use-mindmap-checkbox",
                                value=False,
                            )

            with gr.Column(
                scale=INFO_PANEL_SCALES[False], elem_id="chat-info-panel"
            ) as self.info_column:
                with gr.Accordion(
                    label="Information panel", open=True, elem_id="info-expand"
                ):
                    self.modal = gr.HTML("<div id='pdf-modal'></div>")
                    self.plot_panel = gr.Plot(visible=False)
                    self.info_panel = gr.HTML(elem_id="html-info-panel")

        self.followup_questions = self.chat_suggestion.examples
        self.followup_questions_ui = self.chat_suggestion.accordion

    def _json_to_plot(self, json_dict: dict | None):
        if json_dict:
            plot = from_json(json_dict)
            plot = gr.update(visible=True, value=plot)
        else:
            plot = gr.update(visible=False)
        return plot

    def on_register_events(self):
        # first index paper recommendation
        if KH_DEMO_MODE and len(self._indices_input) > 0:
            self._indices_input[1].change(
                self.get_recommendations,
                inputs=[self.first_selector_choices, self._indices_input[1]],
                outputs=[self.related_papers],
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                js=recommended_papers_js,
            )

        chat_event = (
            gr.on(
                triggers=[
                    self.chat_panel.text_input.submit,
                ],
                fn=self.submit_msg,
                inputs=[
                    self.chat_panel.text_input,
                    self.chat_panel.chatbot,
                    self._app.user_id,
                    self._app.settings_state,
                    self.chat_control.conversation_id,
                    self.chat_control.conversation_rn,
                    self.first_selector_choices,
                ],
                outputs=[
                    self.chat_panel.text_input,
                    self.chat_panel.chatbot,
                    self.chat_control.conversation_id,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                    # file selector from the first index
                    self._indices_input[0],
                    self._indices_input[1],
                    self._command_state,
                ],
                concurrency_limit=20,
                show_progress="hidden",
            )
            .success(
                fn=self.chat_fn,
                inputs=[
                    self.chat_control.conversation_id,
                    self.chat_panel.chatbot,
                    self._app.settings_state,
                    self._reasoning_type,
                    self.model_type,
                    self.use_mindmap,
                    self.citation,
                    self.language,
                    self.state_chat,
                    self._command_state,
                    self._app.user_id,
                ]
                + self._indices_input,
                outputs=[
                    self.chat_panel.chatbot,
                    self.info_panel,
                    self.plot_panel,
                    self.state_plot_panel,
                    self.state_chat,
                ],
                concurrency_limit=20,
                show_progress="minimal",
            )
            .then(
                fn=lambda: True,
                inputs=None,
                outputs=[self._preview_links],
                js=pdfview_js,
            )
            .success(
                fn=self.check_and_suggest_name_conv,
                inputs=self.chat_panel.chatbot,
                outputs=[
                    self.chat_control.conversation_rn,
                    self._conversation_renamed,
                ],
            )
            .success(
                self.chat_control.rename_conv,
                inputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation_rn,
                    self._conversation_renamed,
                    self._app.user_id,
                ],
                outputs=[
                    self.chat_control.conversation,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                ],
                show_progress="hidden",
            )
        )

        onSuggestChatEvent = {
            "fn": self.suggest_chat_conv,
            "inputs": [
                self._app.settings_state,
                self.language,
                self.chat_panel.chatbot,
                self._use_suggestion,
            ],
            "outputs": [
                self.followup_questions_ui,
                self.followup_questions,
            ],
            "show_progress": "hidden",
        }
        # chat suggestion toggle
        chat_event = chat_event.success(**onSuggestChatEvent)

        # final data persist
        if not KH_DEMO_MODE:
            chat_event = chat_event.then(
                fn=self.persist_data_source,
                inputs=[
                    self.chat_control.conversation_id,
                    self._app.user_id,
                    self.info_panel,
                    self.state_plot_panel,
                    self.state_retrieval_history,
                    self.state_plot_history,
                    self.chat_panel.chatbot,
                    self.state_chat,
                ]
                + self._indices_input,
                outputs=[
                    self.state_retrieval_history,
                    self.state_plot_history,
                ],
                concurrency_limit=20,
            )

        self.chat_control.btn_info_expand.click(
            fn=lambda is_expanded: (
                gr.update(scale=INFO_PANEL_SCALES[is_expanded]),
                not is_expanded,
            ),
            inputs=self._info_panel_expanded,
            outputs=[self.info_column, self._info_panel_expanded],
        )
        self.chat_control.btn_chat_expand.click(
            fn=None, inputs=None, js="function() {toggleChatColumn();}"
        )

        if KH_DEMO_MODE:
            self.chat_control.btn_demo_logout.click(
                fn=None,
                js=self.chat_control.logout_js,
            )
            self.chat_control.btn_new.click(
                fn=lambda: self.chat_control.select_conv("", None),
                outputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                    self.chat_panel.chatbot,
                    self.followup_questions,
                    self.info_panel,
                    self.state_plot_panel,
                    self.state_retrieval_history,
                    self.state_plot_history,
                    self.chat_control.cb_is_public,
                    self.state_chat,
                ]
                + self._indices_input,
            ).then(
                lambda: (gr.update(visible=False), gr.update(visible=True)),
                outputs=[self.paper_list.accordion, self.chat_settings],
            ).then(
                fn=None,
                inputs=None,
                js=chat_input_focus_js,
            )

        if not KH_DEMO_MODE:
            self.chat_control.btn_new.click(
                self.chat_control.new_conv,
                inputs=self._app.user_id,
                outputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation,
                ],
                show_progress="hidden",
            ).then(
                self.chat_control.select_conv,
                inputs=[self.chat_control.conversation, self._app.user_id],
                outputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                    self.chat_panel.chatbot,
                    self.followup_questions,
                    self.info_panel,
                    self.state_plot_panel,
                    self.state_retrieval_history,
                    self.state_plot_history,
                    self.chat_control.cb_is_public,
                    self.state_chat,
                ]
                + self._indices_input,
                show_progress="hidden",
            ).then(
                fn=self._json_to_plot,
                inputs=self.state_plot_panel,
                outputs=self.plot_panel,
            ).then(
                fn=None,
                inputs=None,
                js=chat_input_focus_js,
            )

            self.chat_control.btn_del.click(
                lambda id: self.toggle_delete(id),
                inputs=[self.chat_control.conversation_id],
                outputs=[
                    self.chat_control._new_delete,
                    self.chat_control._delete_confirm,
                ],
            )
            self.chat_control.btn_del_conf.click(
                self.chat_control.delete_conv,
                inputs=[self.chat_control.conversation_id, self._app.user_id],
                outputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation,
                ],
                show_progress="hidden",
            ).then(
                self.chat_control.select_conv,
                inputs=[self.chat_control.conversation, self._app.user_id],
                outputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                    self.chat_panel.chatbot,
                    self.followup_questions,
                    self.info_panel,
                    self.state_plot_panel,
                    self.state_retrieval_history,
                    self.state_plot_history,
                    self.chat_control.cb_is_public,
                    self.state_chat,
                ]
                + self._indices_input,
                show_progress="hidden",
            ).then(
                fn=self._json_to_plot,
                inputs=self.state_plot_panel,
                outputs=self.plot_panel,
            ).then(
                lambda: self.toggle_delete(""),
                outputs=[
                    self.chat_control._new_delete,
                    self.chat_control._delete_confirm,
                ],
            )
            self.chat_control.btn_del_cnl.click(
                lambda: self.toggle_delete(""),
                outputs=[
                    self.chat_control._new_delete,
                    self.chat_control._delete_confirm,
                ],
            )
            self.chat_control.btn_conversation_rn.click(
                lambda: gr.update(visible=True),
                outputs=[
                    self.chat_control.conversation_rn,
                ],
            )
            self.chat_control.conversation_rn.submit(
                self.chat_control.rename_conv,
                inputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation_rn,
                    gr.State(value=True),
                    self._app.user_id,
                ],
                outputs=[
                    self.chat_control.conversation,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                ],
                show_progress="hidden",
            )

        onConvSelect = (
            self.chat_control.conversation.select(
                self.chat_control.select_conv,
                inputs=[self.chat_control.conversation, self._app.user_id],
                outputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                    self.chat_panel.chatbot,
                    self.followup_questions,
                    self.info_panel,
                    self.state_plot_panel,
                    self.state_retrieval_history,
                    self.state_plot_history,
                    self.chat_control.cb_is_public,
                    self.state_chat,
                ]
                + self._indices_input,
                show_progress="hidden",
            )
            .then(
                fn=self._json_to_plot,
                inputs=self.state_plot_panel,
                outputs=self.plot_panel,
            )
            .then(
                lambda: self.toggle_delete(""),
                outputs=[
                    self.chat_control._new_delete,
                    self.chat_control._delete_confirm,
                ],
            )
        )

        if KH_DEMO_MODE:
            onConvSelect = onConvSelect.then(
                lambda: (gr.update(visible=False), gr.update(visible=True)),
                outputs=[self.paper_list.accordion, self.chat_settings],
            )

        onConvSelect = (
            onConvSelect.then(
                fn=lambda: True,
                js=clear_bot_message_selection_js,
            )
            .then(
                fn=lambda: True,
                inputs=None,
                outputs=[self._preview_links],
                js=pdfview_js,
            )
            .then(fn=None, inputs=None, outputs=None, js=chat_input_focus_js)
        )

        if not KH_DEMO_MODE:
            # evidence display on message selection
            self.chat_panel.chatbot.select(
                self.message_selected,
                inputs=[
                    self.state_retrieval_history,
                    self.state_plot_history,
                ],
                outputs=[
                    self.info_panel,
                    self.state_plot_panel,
                ],
            ).then(
                fn=self._json_to_plot,
                inputs=self.state_plot_panel,
                outputs=self.plot_panel,
            ).then(
                fn=lambda: True,
                inputs=None,
                outputs=[self._preview_links],
                js=pdfview_js,
            )

        self.chat_control.cb_is_public.change(
            self.on_set_public_conversation,
            inputs=[self.chat_control.cb_is_public, self.chat_control.conversation],
            outputs=None,
            show_progress="hidden",
        )

        if not KH_DEMO_MODE:
            # user feedback events
            self.chat_panel.chatbot.like(
                fn=self.is_liked,
                inputs=[self.chat_control.conversation_id],
                outputs=None,
            )
            self.report_issue.report_btn.click(
                self.report_issue.report,
                inputs=[
                    self.report_issue.correctness,
                    self.report_issue.issues,
                    self.report_issue.more_detail,
                    self.chat_control.conversation_id,
                    self.chat_panel.chatbot,
                    self._app.settings_state,
                    self._app.user_id,
                    self.info_panel,
                    self.state_chat,
                ]
                + self._indices_input,
                outputs=None,
            )

        self.reasoning_type.change(
            self.reasoning_changed,
            inputs=[self.reasoning_type],
            outputs=[self._reasoning_type],
        )
        self.use_mindmap_check.change(
            lambda x: (x, gr.update(label="Mindmap " + ("(on)" if x else "(off)"))),
            inputs=[self.use_mindmap_check],
            outputs=[self.use_mindmap, self.use_mindmap_check],
            show_progress="hidden",
        )

        def toggle_chat_suggestion(current_state):
            return current_state, gr.update(visible=current_state)

        def raise_error_on_state(state):
            if not state:
                raise ValueError("Chat suggestion disabled")

        self.chat_control.cb_suggest_chat.change(
            fn=toggle_chat_suggestion,
            inputs=[self.chat_control.cb_suggest_chat],
            outputs=[self._use_suggestion, self.followup_questions_ui],
            show_progress="hidden",
        ).then(
            fn=raise_error_on_state,
            inputs=[self._use_suggestion],
            show_progress="hidden",
        ).success(
            **onSuggestChatEvent
        )
        self.chat_control.conversation_id.change(
            lambda: gr.update(visible=False),
            outputs=self.plot_panel,
        )

        self.followup_questions.select(
            self.chat_suggestion.select_example,
            outputs=[self.chat_panel.text_input],
            show_progress="hidden",
        ).then(
            fn=None,
            inputs=None,
            outputs=None,
            js=chat_input_focus_js,
        )

        if KH_DEMO_MODE:
            self.paper_list.examples.select(
                self.paper_list.select_example,
                inputs=[self.paper_list.papers_state],
                outputs=[self.quick_urls],
                show_progress="hidden",
            ).then(
                lambda: (gr.update(visible=False), gr.update(visible=True)),
                outputs=[self.paper_list.accordion, self.chat_settings],
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                js=quick_urls_submit_js,
            )

    def submit_msg(
        self,
        chat_input,
        chat_history,
        user_id,
        settings,
        conv_id,
        conv_name,
        first_selector_choices,
        request: gr.Request,
    ):
        """Submit a message to the chatbot"""
        if KH_DEMO_MODE:
            sso_user_id = check_rate_limit("chat", request)
            print("User ID:", sso_user_id)

        if not chat_input:
            raise ValueError("Input is empty")

        chat_input_text = chat_input.get("text", "")
        file_ids = []
        used_command = None

        first_selector_choices_map = {
            item[0]: item[1] for item in first_selector_choices
        }

        # get all file names with pattern @"filename" in input_str
        file_names, chat_input_text = get_file_names_regex(chat_input_text)

        # check if web search command is in file_names
        if WEB_SEARCH_COMMAND in file_names:
            used_command = WEB_SEARCH_COMMAND

        # get all urls in input_str
        urls, chat_input_text = get_urls(chat_input_text)

        if urls and self.first_indexing_url_fn:
            print("Detected URLs", urls)
            file_ids = self.first_indexing_url_fn(
                "\n".join(urls),
                True,
                settings,
                user_id,
                request=None,
            )
        elif file_names:
            for file_name in file_names:
                file_id = first_selector_choices_map.get(file_name)
                if file_id:
                    file_ids.append(file_id)

        # add new file ids to the first selector choices
        first_selector_choices.extend(zip(urls, file_ids))

        # if file_ids is not empty and chat_input_text is empty
        # set the input to summary
        if not chat_input_text and file_ids:
            chat_input_text = DEFAULT_QUESTION

        # if start of conversation and no query is specified
        if not chat_input_text and not chat_history:
            chat_input_text = DEFAULT_QUESTION

        if file_ids:
            selector_output = [
                "select",
                gr.update(value=file_ids, choices=first_selector_choices),
            ]
        else:
            selector_output = [gr.update(), gr.update()]

        # check if regen mode is active
        if chat_input_text:
            chat_history = chat_history + [(chat_input_text, None)]
        else:
            if not chat_history:
                raise gr.Error("Empty chat")

        if not conv_id:
            if not KH_DEMO_MODE:
                id_, update = self.chat_control.new_conv(user_id)
                with Session(engine) as session:
                    statement = select(Conversation).where(Conversation.id == id_)
                    name = session.exec(statement).one().name
                    new_conv_id = id_
                    conv_update = update
                    new_conv_name = name
            else:
                new_conv_id, new_conv_name, conv_update = None, None, gr.update()
        else:
            new_conv_id = conv_id
            conv_update = gr.update()
            new_conv_name = conv_name

        return (
            [
                {},
                chat_history,
                new_conv_id,
                conv_update,
                new_conv_name,
            ]
            + selector_output
            + [used_command]
        )

    def get_recommendations(self, first_selector_choices, file_ids):
        first_selector_choices_map = {
            item[1]: item[0] for item in first_selector_choices
        }
        file_names = [first_selector_choices_map[file_id] for file_id in file_ids]
        if not file_names:
            return ""

        first_file_name = file_names[0].split(".")[0].replace("_", " ")
        return get_recommended_papers(first_file_name)

    def toggle_delete(self, conv_id):
        if conv_id:
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)

    def on_set_public_conversation(self, is_public, convo_id):
        if not convo_id:
            gr.Warning("No conversation selected")
            return

        with Session(engine) as session:
            statement = select(Conversation).where(Conversation.id == convo_id)

            result = session.exec(statement).one()
            name = result.name

            if result.is_public != is_public:
                # Only trigger updating when user
                # select different value from the current
                result.is_public = is_public
                session.add(result)
                session.commit()

                gr.Info(
                    f"Conversation: {name} is {'public' if is_public else 'private'}."
                )

    def on_subscribe_public_events(self):
        if self._app.f_user_management:
            self._app.subscribe_event(
                name="onSignIn",
                definition={
                    "fn": self.chat_control.reload_conv,
                    "inputs": [self._app.user_id],
                    "outputs": [self.chat_control.conversation],
                    "show_progress": "hidden",
                },
            )

            self._app.subscribe_event(
                name="onSignOut",
                definition={
                    "fn": lambda: self.chat_control.select_conv("", None),
                    "outputs": [
                        self.chat_control.conversation_id,
                        self.chat_control.conversation,
                        self.chat_control.conversation_rn,
                        self.chat_panel.chatbot,
                        self.followup_questions,
                        self.info_panel,
                        self.state_plot_panel,
                        self.state_retrieval_history,
                        self.state_plot_history,
                        self.chat_control.cb_is_public,
                        self.state_chat,
                    ]
                    + self._indices_input,
                    "show_progress": "hidden",
                },
            )

    def _on_app_created(self):
        if KH_DEMO_MODE:
            self._app.app.load(
                fn=lambda x: x,
                inputs=[self._user_api_key],
                outputs=[self._user_api_key],
                js=fetch_api_key_js,
            ).then(
                fn=self.chat_control.toggle_demo_login_visibility,
                inputs=[self._user_api_key],
                outputs=[
                    self.chat_control.cb_suggest_chat,
                    self.chat_control.btn_new,
                    self.chat_control.btn_demo_logout,
                    self.chat_control.btn_demo_login,
                ],
            ).then(
                fn=None,
                inputs=None,
                js=chat_input_focus_js,
            )

    def persist_data_source(
        self,
        convo_id,
        user_id,
        retrieval_msg,
        plot_data,
        retrival_history,
        plot_history,
        messages,
        state,
        *selecteds,
    ):
        """Update the data source"""
        if not convo_id:
            gr.Warning("No conversation selected")
            return

        # if not regen, then append the new message
        if not state["app"].get("regen", False):
            retrival_history = retrival_history + [retrieval_msg]
            plot_history = plot_history + [plot_data]
        else:
            if retrival_history:
                print("Updating retrieval history (regen=True)")
                retrival_history[-1] = retrieval_msg
                plot_history[-1] = plot_data

        # reset regen state
        state["app"]["regen"] = False

        selecteds_ = {}
        for index in self._app.index_manager.indices:
            if index.selector is None:
                continue
            if isinstance(index.selector, int):
                selecteds_[str(index.id)] = selecteds[index.selector]
            else:
                selecteds_[str(index.id)] = [selecteds[i] for i in index.selector]

        with Session(engine) as session:
            statement = select(Conversation).where(Conversation.id == convo_id)
            result = session.exec(statement).one()

            data_source = result.data_source
            old_selecteds = data_source.get("selected", {})
            is_owner = result.user == user_id

            # Write down to db
            result.data_source = {
                "selected": selecteds_ if is_owner else old_selecteds,
                "messages": messages,
                "retrieval_messages": retrival_history,
                "plot_history": plot_history,
                "state": state,
                "likes": deepcopy(data_source.get("likes", [])),
            }
            session.add(result)
            session.commit()

        return retrival_history, plot_history

    def reasoning_changed(self, reasoning_type):
        if reasoning_type != DEFAULT_SETTING:
            # override app settings state (temporary)
            gr.Info("Reasoning type changed to `{}`".format(reasoning_type))
        return reasoning_type

    def is_liked(self, convo_id, liked: gr.LikeData):
        with Session(engine) as session:
            statement = select(Conversation).where(Conversation.id == convo_id)
            result = session.exec(statement).one()

            data_source = deepcopy(result.data_source)
            likes = data_source.get("likes", [])
            likes.append([liked.index, liked.value, liked.liked])
            data_source["likes"] = likes

            result.data_source = data_source
            session.add(result)
            session.commit()

    def message_selected(self, retrieval_history, plot_history, msg: gr.SelectData):
        index = msg.index[0]
        try:
            retrieval_content, plot_content = (
                retrieval_history[index],
                plot_history[index],
            )
        except IndexError:
            retrieval_content, plot_content = gr.update(), None

        return retrieval_content, plot_content

    @staticmethod
    def _live_doc_text(doc: Any) -> str:
        text = getattr(doc, "text", None)
        if text is not None:
            return str(text)
        content = getattr(doc, "content", None)
        if content is not None:
            return str(content)
        return str(doc)

    @staticmethod
    def _live_doc_source(doc: Any) -> str:
        metadata = getattr(doc, "metadata", None) or {}
        for key in ("file_path", "source", "file_name", "filename", "path"):
            value = metadata.get(key)
            if value:
                return str(value)
        file_id = metadata.get("file_id")
        if file_id:
            return str(file_id)
        return str(getattr(doc, "doc_id", ""))

    def _live_doc_generation_text(self, doc: Any) -> str:
        metadata = getattr(doc, "metadata", None) or {}
        if "window" in metadata:
            text = metadata["window"]
        elif metadata.get("type") == "table":
            text = metadata.get("table_origin", self._live_doc_text(doc))
        elif metadata.get("type") == "chatbot":
            text = metadata.get("window", self._live_doc_text(doc))
        elif metadata.get("type") == "image":
            text = metadata.get("image_origin", self._live_doc_text(doc))
        else:
            text = self._live_doc_text(doc)
        return str(text).replace("\n", " ")

    @staticmethod
    def _live_contains_flags(text: str) -> dict:
        lowered = str(text or "").lower()
        return {
            "contains_six_semesters": "six semesters" in lowered,
            "contains_6_semester": "6 semester" in lowered,
            "contains_standard_length": "standard length" in lowered,
            "contains_180_ects": "180 ects" in lowered,
        }

    def _live_context_debug_item(self, rank: int, doc: Any) -> dict:
        text = self._live_doc_text(doc)
        item = {
            "rank": rank,
            "source": self._live_doc_source(doc),
            "context_id": str(getattr(doc, "doc_id", "")),
            "score": str(getattr(doc, "score", "")),
            "text_preview": text[:500],
            "full_text": text,
        }
        item.update(self._live_contains_flags(text))
        return item

    def _serialize_live_message(self, message: Any) -> str:
        role = message.__class__.__name__
        content = getattr(message, "content", str(message))
        return f"[{role}]\n{content}"

    @staticmethod
    def _approx_live_tokens(text: str) -> int:
        return max(1, int(len(str(text or "")) / 4))

    @property
    def _ku_d3b_eval_root(self) -> Path:
        # libs/ktem/ktem/pages/chat/__init__.py -> repository root
        return Path(__file__).resolve().parents[5] / "ku_d3b_eval"

    def _describe_live_selected_state(self, selecteds: tuple[Any, ...]) -> dict:
        """Describe the exact index selector values passed to the live pipeline."""
        selected_indices = []
        selected_file_ids: list[str] = []
        selected_filenames: list[str] = []
        selected_file_id_to_filename: dict[str, str] = {}
        selected_state = {}

        for index in self._app.index_manager.indices:
            selector_value: Any = None
            if index.selector is None:
                selector_value = None
            elif isinstance(index.selector, int):
                selector_value = selecteds[index.selector] if len(selecteds) > index.selector else None
            elif isinstance(index.selector, tuple):
                selector_value = [
                    selecteds[i] if len(selecteds) > i else None for i in index.selector
                ]

            selected_state[str(index.id)] = selector_value

            resolved_ids: list[str] = []
            try:
                selector_ui = index.get_selector_component_ui()
                if selector_ui and hasattr(selector_ui, "get_selected_ids"):
                    ids = selector_ui.get_selected_ids(selector_value) or []
                    resolved_ids = [str(each) for each in ids if each is not None]
            except Exception:
                resolved_ids = []

            id_to_filename = self._live_source_filename_map(index, resolved_ids)
            resolved_filenames = [
                id_to_filename.get(file_id, "") for file_id in resolved_ids
            ]
            selected_file_id_to_filename.update(id_to_filename)

            selected_indices.append(
                {
                    "id": str(index.id),
                    "name": getattr(index, "name", str(index.id)),
                    "selector": selector_value,
                    "resolved_file_ids": resolved_ids,
                    "resolved_filenames": resolved_filenames,
                    "resolved_file_id_to_filename": id_to_filename,
                }
            )
            selected_file_ids.extend(resolved_ids)
            selected_filenames.extend([name for name in resolved_filenames if name])

        return {
            "selected_index": ", ".join(
                f"{item['name']}({item['id']})" for item in selected_indices
            ),
            "selected_indices": selected_indices,
            "selected_file_ids": selected_file_ids,
            "selected_filenames": selected_filenames,
            "selected_file_id_to_filename": selected_file_id_to_filename,
            "selected_state": selected_state,
        }

    def _live_source_filename_map(self, index: Any, file_ids: list[str]) -> dict[str, str]:
        if not file_ids:
            return {}
        try:
            Source = index._resources["Source"]
            with Session(engine) as session:
                rows = session.execute(sa_select(Source).where(Source.id.in_(file_ids))).all()
            mapping = {}
            for row in rows:
                source = row[0]
                filename = getattr(source, "name", None) or getattr(source, "path", "")
                mapping[str(source.id)] = str(filename or "")
            return mapping
        except Exception:
            return {}

    @staticmethod
    def _fold_debug_text(text: str) -> str:
        return (
            str(text or "")
            .lower()
            .replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )

    def _count_vectors_for_chunks(self, vector_store: Any, chunk_ids: list[str]) -> int | None:
        if not chunk_ids:
            return 0
        try:
            collection = getattr(vector_store, "_collection", None)
            if collection is not None and hasattr(collection, "get"):
                got = collection.get(ids=chunk_ids, include=[])
                return len(got.get("ids", []) if isinstance(got, dict) else [])
        except Exception:
            pass
        try:
            client = getattr(getattr(vector_store, "_client", None), "client", None)
            if client is not None and hasattr(client, "get"):
                got = client.get(ids=chunk_ids, include=[])
                return len(got.get("ids", []) if isinstance(got, dict) else [])
        except Exception:
            pass
        return None

    def _build_index_state_debug(
        self,
        selected_info: dict | None,
        question: str,
        pipeline: Any,
    ) -> dict:
        selected_info = selected_info or {}
        debug: dict = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "selected_file_ids": selected_info.get("selected_file_ids", []),
            "selected_filenames": selected_info.get("selected_filenames", []),
            "selected_file_id_to_filename": selected_info.get(
                "selected_file_id_to_filename", {}
            ),
            "files": [],
            "exact_indexed_text_search": {},
            "retrieval_comparison": [],
        }

        exact_terms = [
            "Ostenstraße 26",
            "Ostenstrasse 26",
            "Prüfungsamt",
            "Pruefungsamt",
            "Marktplatz 7",
            "85072 Eichstätt",
            "85072 Eichstaett",
        ]
        selected_ids = set(debug["selected_file_ids"])
        all_selected_chunk_ids: set[str] = set()
        exact_hits = {
            term: {"files": [], "chunk_ids": [], "chunks": []} for term in exact_terms
        }

        for index in self._app.index_manager.indices:
            index_selected_ids: list[str] = []
            for item in selected_info.get("selected_indices", []) or []:
                if str(item.get("id")) == str(index.id):
                    index_selected_ids = [str(each) for each in item.get("resolved_file_ids", [])]
                    break
            if not index_selected_ids:
                continue

            try:
                Source = index._resources["Source"]
                Index = index._resources["Index"]
                docstore = index._resources["DocStore"]
                vectorstore = index._resources["VectorStore"]
                with Session(engine) as session:
                    sources = {
                        str(row[0].id): row[0]
                        for row in session.execute(
                            sa_select(Source).where(Source.id.in_(index_selected_ids))
                        ).all()
                    }
                    index_rows = session.execute(
                        sa_select(Index).where(
                            Index.relation_type == "document",
                            Index.source_id.in_(index_selected_ids),
                        )
                    ).all()
            except Exception as exc:
                debug.setdefault("errors", []).append(
                    f"Could not inspect index {getattr(index, 'id', '')}: {exc}"
                )
                continue

            file_to_chunk_ids: dict[str, list[str]] = {file_id: [] for file_id in index_selected_ids}
            chunk_to_file_id: dict[str, str] = {}
            for row in index_rows:
                idx_row = row[0]
                file_id = str(idx_row.source_id)
                chunk_id = str(idx_row.target_id)
                file_to_chunk_ids.setdefault(file_id, []).append(chunk_id)
                chunk_to_file_id[chunk_id] = file_id
                all_selected_chunk_ids.add(chunk_id)

            all_chunk_ids = [chunk_id for ids in file_to_chunk_ids.values() for chunk_id in ids]
            try:
                docs = docstore.get(all_chunk_ids) if all_chunk_ids else []
            except Exception as exc:
                docs = []
                debug.setdefault("errors", []).append(
                    f"Could not read selected chunks for index {getattr(index, 'id', '')}: {exc}"
                )

            docs_by_file: dict[str, list[Any]] = {file_id: [] for file_id in index_selected_ids}
            for doc in docs:
                metadata = getattr(doc, "metadata", None) or {}
                file_id = str(metadata.get("file_id") or chunk_to_file_id.get(str(getattr(doc, "doc_id", "")), ""))
                docs_by_file.setdefault(file_id, []).append(doc)

                raw_text = self._live_doc_text(doc)
                raw_lower = raw_text.lower()
                folded = self._fold_debug_text(raw_text)
                for term in exact_terms:
                    term_lower = term.lower()
                    term_folded = self._fold_debug_text(term)
                    if term_lower in raw_lower or term_folded in folded:
                        hit = exact_hits[term]
                        filename = (
                            selected_info.get("selected_file_id_to_filename", {}).get(file_id)
                            or getattr(sources.get(file_id), "name", "")
                        )
                        if filename and filename not in hit["files"]:
                            hit["files"].append(filename)
                        chunk_id = str(getattr(doc, "doc_id", ""))
                        if chunk_id not in hit["chunk_ids"]:
                            hit["chunk_ids"].append(chunk_id)
                            hit["chunks"].append(
                                {
                                    "chunk_id": chunk_id,
                                    "file_id": file_id,
                                    "filename": filename,
                                    "belongs_to_selected_file_ids": file_id in selected_ids,
                                    "text_preview": raw_text[:500],
                                }
                            )

            for file_id in index_selected_ids:
                source = sources.get(file_id)
                filename = (
                    selected_info.get("selected_file_id_to_filename", {}).get(file_id)
                    or getattr(source, "name", "")
                    or getattr(source, "path", "")
                )
                file_docs = docs_by_file.get(file_id, [])
                combined_text = "\n".join(self._live_doc_text(doc) for doc in file_docs)
                combined_folded = self._fold_debug_text(combined_text)
                chunk_ids = file_to_chunk_ids.get(file_id, [])
                debug["files"].append(
                    {
                        "file_id": file_id,
                        "filename": str(filename or ""),
                        "num_chunks_in_docstore": len(file_docs),
                        "num_vectors_in_vectorstore": self._count_vectors_for_chunks(
                            vectorstore, chunk_ids
                        ),
                        "sample_chunk_previews": [
                            self._live_doc_text(doc)[:500] for doc in file_docs[:3]
                        ],
                        "contains_ostenstrasse": "ostenstrasse" in combined_folded,
                        "contains_ostenstraße": "ostenstraße" in combined_text.lower(),
                        "contains_pruefungsamt": "pruefungsamt" in combined_folded,
                        "contains_marktplatz_7": "marktplatz 7" in combined_folded,
                        "contains_85072_eichstaett": "85072 eichstaett" in combined_folded,
                    }
                )

        debug["exact_indexed_text_search"] = {
            term: {
                **payload,
                "found": bool(payload["chunk_ids"]),
                "all_hits_belong_to_selected_file_ids": all(
                    chunk.get("belongs_to_selected_file_ids") for chunk in payload["chunks"]
                )
                if payload["chunks"]
                else False,
            }
            for term, payload in exact_hits.items()
        }

        if self._should_build_retrieval_comparison(question):
            for retriever in getattr(pipeline, "retrievers", []) or []:
                compare = getattr(retriever, "debug_retrieval_comparison", None)
                if callable(compare):
                    try:
                        debug["retrieval_comparison"].append(compare(question, top_k=10))
                    except Exception as exc:
                        debug["retrieval_comparison"].append(
                            {"error": f"{exc.__class__.__name__}: {exc}"}
                        )

        return debug

    @staticmethod
    def _should_build_retrieval_comparison(question: str) -> bool:
        folded = ChatPage._fold_debug_text(question)
        return any(
            term in folded
            for term in (
                "postal address",
                "address",
                "examination office",
                "pruefungsamt",
                "ostenstrasse",
                "eichstaett",
                "ingolstadt",
                "schanz",
            )
        )

    def _write_index_state_debug(self, payload: dict) -> Path:
        debug_dir = self._ku_d3b_eval_root / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / "index_state_latest.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def _write_live_chat_debug(self, payload: dict) -> None:
        debug_dir = self._ku_d3b_eval_root / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        latest = debug_dir / "live_chat_latest.json"
        history = debug_dir / "live_chat_debug.jsonl"
        latest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _install_live_debug_hooks(self, pipeline, debug_state: dict) -> list:
        """Capture the exact live pipeline artifacts without replacing the pipeline."""
        restore_callbacks = []

        original_retrieve = getattr(pipeline, "retrieve", None)
        if callable(original_retrieve):

            def capture_retrieve(message, history):
                docs, infos = original_retrieve(message, history)
                docs = list(docs or [])
                infos = list(infos or [])
                debug_state["retrieved_docs"] = docs
                debug_state["information_panel_docs"] = infos
                return docs, infos

            try:
                object.__setattr__(pipeline, "retrieve", capture_retrieve)
                restore_callbacks.append(
                    lambda: object.__setattr__(pipeline, "retrieve", original_retrieve)
                )
            except Exception as exc:
                debug_state.setdefault("hook_warnings", []).append(
                    f"Could not hook retrieve(): {exc}"
                )

        answering_pipeline = getattr(pipeline, "answering_pipeline", None)
        if answering_pipeline is not None:
            original_get_prompt = getattr(answering_pipeline, "get_prompt", None)
            if callable(original_get_prompt):

                def capture_get_prompt(question, evidence, evidence_mode):
                    prompt, evidence_out = original_get_prompt(
                        question, evidence, evidence_mode
                    )
                    debug_state["answer_evidence"] = evidence_out
                    debug_state["answer_prompt"] = prompt
                    debug_state["answer_evidence_mode"] = evidence_mode
                    return prompt, evidence_out

                try:
                    object.__setattr__(
                        answering_pipeline, "get_prompt", capture_get_prompt
                    )
                    restore_callbacks.append(
                        lambda: object.__setattr__(
                            answering_pipeline, "get_prompt", original_get_prompt
                        )
                    )
                except Exception as exc:
                    debug_state.setdefault("hook_warnings", []).append(
                        f"Could not hook get_prompt(): {exc}"
                    )

            llm = getattr(answering_pipeline, "llm", None)
            original_stream = getattr(llm, "stream", None)
            if callable(original_stream):

                def capture_stream(messages, *args, **kwargs):
                    if not debug_state.get("answer_messages"):
                        debug_state["answer_messages"] = list(messages)
                    yield from original_stream(messages, *args, **kwargs)

                try:
                    object.__setattr__(llm, "stream", capture_stream)
                    restore_callbacks.append(
                        lambda: object.__setattr__(llm, "stream", original_stream)
                    )
                except Exception as exc:
                    debug_state.setdefault("hook_warnings", []).append(
                        f"Could not hook llm.stream(): {exc}"
                    )

        return restore_callbacks

    def _build_live_debug_payload(
        self,
        sample_id: str,
        mode: str,
        question: str,
        pipeline,
        debug_state: dict,
        final_answer: str,
        selected_info: dict | None = None,
        selected_pipeline: str | None = None,
        selected_reasoning: str | None = None,
    ) -> dict:
        retrieved_docs = debug_state.get("retrieved_docs", []) or []
        retrieved_contexts = [
            self._live_context_debug_item(rank, doc)
            for rank, doc in enumerate(retrieved_docs, start=1)
        ]

        evidence = str(debug_state.get("answer_evidence") or "")
        contexts_sent = []
        for rank, doc in enumerate(retrieved_docs, start=1):
            text = self._live_doc_generation_text(doc)
            if text and (text[:200].replace("\n", " ") in evidence or text in evidence):
                item = self._live_context_debug_item(rank, doc)
                item["text_preview"] = text[:500]
                item["full_text"] = text
                item.update(self._live_contains_flags(text))
                contexts_sent.append(item)
        if not contexts_sent and evidence:
            item = {
                "rank": 1,
                "source": "prepared_evidence",
                "context_id": "",
                "text_preview": evidence[:500],
                "full_text": evidence,
            }
            item.update(self._live_contains_flags(evidence))
            contexts_sent.append(item)

        messages = debug_state.get("answer_messages") or []
        if messages:
            final_prompt_or_messages = "\n\n".join(
                self._serialize_live_message(message) for message in messages
            )
        else:
            final_prompt_or_messages = str(debug_state.get("answer_prompt") or "")

        info_contexts = []
        # The Information panel renders the same retrieved docs (possibly after
        # citation/highlight processing). Use the raw retrieved docs here so debug
        # output has inspectable source/text instead of rendered HTML only.
        for rank, doc in enumerate(retrieved_docs, start=1):
            info_contexts.append(self._live_context_debug_item(rank, doc))

        prompt_flags = self._live_contains_flags(final_prompt_or_messages)
        generation_debug = getattr(pipeline, "_generation_debug", {}) or {}
        answer_generation_debug = getattr(pipeline, "_answer_generation_debug", {}) or {}
        llm_relevance_scorer_enabled = generation_debug.get(
            "llm_relevance_scorer_enabled",
            getattr(pipeline, "_llm_relevance_scorer_enabled", False),
        )
        llm_relevance_scorer_failed = generation_debug.get(
            "llm_relevance_scorer_failed",
            getattr(pipeline, "_llm_relevance_scorer_failed", False),
        )
        llm_relevance_scorer_errors_count = generation_debug.get(
            "llm_relevance_scorer_errors_count",
            getattr(pipeline, "_llm_relevance_scorer_errors_count", 0),
        )
        retrieved_text = "\n".join(item["full_text"] for item in retrieved_contexts)
        info_text = "\n".join(item["full_text"] for item in info_contexts)
        answer_l = final_answer.lower()
        answer_says_not_explicitly_stated = any(
            marker in answer_l
            for marker in (
                "not explicitly stated",
                "does not explicitly state",
                "not explicitly state",
            )
        )

        warnings = []
        warnings.extend(debug_state.get("hook_warnings", []))
        for phrase_key, phrase in (
            ("six_semesters", "six semesters"),
            ("6_semester", "6 semester"),
        ):
            if phrase in retrieved_text.lower() and phrase not in final_prompt_or_messages.lower():
                warnings.append(
                    f"Retrieved contexts contain '{phrase}' but the final answer prompt does not."
                )
            if phrase in info_text.lower() and phrase not in final_prompt_or_messages.lower():
                warnings.append(
                    f"Information panel contains '{phrase}' but the final answer prompt does not."
                )

        if (
            any(prompt_flags.values())
            and answer_says_not_explicitly_stated
        ):
            warnings.append(
                "Final prompt contains target evidence but the answer says it is not explicitly stated."
            )
        if mode == "evaluation_ui" and not (selected_info or {}).get("selected_file_ids"):
            warnings.append(
                "Evaluation UI selected_file_ids is empty. Confirm that the Evaluation tab is using the same file/index selection state as the Chat tab."
            )
        if mode == "evaluation_ui" and not retrieved_contexts:
            warnings.append(
                "Evaluation UI has empty retrieved_contexts. No retrieved contexts were captured."
            )
        if mode == "evaluation_ui" and not [
            item for item in retrieved_contexts if item.get("source")
        ]:
            warnings.append("Evaluation UI has empty retrieved_sources.")

        payload = {
            "id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "question": question,
            "selected_pipeline": selected_pipeline or pipeline.__class__.__name__,
            "selected_reasoning": selected_reasoning or (
                pipeline.get_info().get("id")
                if hasattr(pipeline, "get_info")
                else pipeline.__class__.__name__
            ),
            "selected_index": (selected_info or {}).get("selected_index", ""),
            "selected_file_ids": (selected_info or {}).get("selected_file_ids", []),
            "selected_filenames": (selected_info or {}).get("selected_filenames", []),
            "selected_file_id_to_filename": (selected_info or {}).get(
                "selected_file_id_to_filename", {}
            ),
            "selected_indices": (selected_info or {}).get("selected_indices", []),
            "selected_state": (selected_info or {}).get("selected_state", {}),
            "pipeline_used": pipeline.__class__.__name__,
            "reasoning_used": (
                pipeline.get_info().get("id")
                if hasattr(pipeline, "get_info")
                else pipeline.__class__.__name__
            ),
            "retrieved_contexts_available": retrieved_contexts,
            "contexts_actually_sent_to_answer_llm": contexts_sent,
            "final_prompt_or_messages_sent_to_answer_llm": final_prompt_or_messages,
            **{f"prompt_{key}": value for key, value in prompt_flags.items()},
            "final_answer": final_answer,
            "answer_says_not_explicitly_stated": answer_says_not_explicitly_stated,
            "information_panel_contexts": info_contexts,
            "llm_relevance_scorer_enabled": bool(llm_relevance_scorer_enabled),
            "llm_relevance_scorer_failed": bool(llm_relevance_scorer_failed),
            "llm_relevance_scorer_errors_count": int(
                llm_relevance_scorer_errors_count or 0
            ),
            "model_context_window": generation_debug.get(
                "model_context_window", answer_generation_debug.get("model_context_window")
            ),
            "answer_max_tokens": generation_debug.get(
                "answer_max_tokens", answer_generation_debug.get("answer_max_tokens")
            ),
            "reserved_prompt_tokens": generation_debug.get("reserved_prompt_tokens"),
            "available_context_tokens": generation_debug.get("available_context_tokens"),
            "estimated_context_tokens": generation_debug.get("estimated_context_tokens"),
            "estimated_final_prompt_tokens": generation_debug.get("estimated_final_prompt_tokens"),
            "approximate_final_prompt_tokens": self._approx_live_tokens(
                final_prompt_or_messages
            ),
            "prompt_exceeds_context_window": generation_debug.get(
                "prompt_exceeds_context_window"
            ),
            "num_retrieved_contexts": generation_debug.get(
                "num_retrieved_contexts", len(retrieved_contexts)
            ),
            "num_contexts_sent_to_llm": generation_debug.get(
                "num_contexts_sent_to_llm", len(contexts_sent)
            ),
            "num_contexts_before_filtering": generation_debug.get(
                "num_contexts_before_filtering", len(retrieved_contexts)
            ),
            "num_contexts_after_filtering": generation_debug.get(
                "num_contexts_after_filtering", len(contexts_sent)
            ),
            "cleaned_contexts_sent_to_llm": generation_debug.get(
                "cleaned_contexts_sent_to_llm", []
            ),
            "final_prompt_after_context_budgeting": final_prompt_or_messages,
            "retry_triggered": answer_generation_debug.get("retry_triggered", False),
            "retry_reason": answer_generation_debug.get("retry_reason", ""),
            "retry_contexts_sent": answer_generation_debug.get("retry_contexts_sent", []),
            "retry_prompt": answer_generation_debug.get("retry_prompt", ""),
            "retry_answer": answer_generation_debug.get("retry_answer", ""),
            "final_selected_answer": final_answer,
            "warnings": warnings,
        }
        try:
            index_state_debug = self._build_index_state_debug(
                selected_info, question, pipeline
            )
            debug_path = self._write_index_state_debug(index_state_debug)
            payload["index_state_debug_path"] = str(debug_path)
        except Exception as exc:
            payload.setdefault("warnings", []).append(
                f"Could not write index_state_latest.json: {exc}"
            )
        return payload

    def stream_ui_chat_query(
        self,
        message: str,
        history: list | None,
        conversation_id: str | None,
        settings: dict,
        reasoning_type: str,
        llm_type: str,
        use_mind_map: bool | str,
        use_citation: str,
        language: str,
        chat_state: dict,
        command_state: str | None,
        user_id: int,
        *selecteds,
        debug: bool = False,
        sample_id: str | None = None,
        debug_mode: str = "normal_chat_ui",
    ):
        """Shared live chat execution path used by normal Chat UI and Evaluation UI."""
        queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
        selected_info = self._describe_live_selected_state(selecteds)
        pipeline, reasoning_state = self.create_pipeline(
            settings,
            reasoning_type,
            llm_type,
            use_mind_map,
            use_citation,
            language,
            chat_state,
            command_state,
            user_id,
            *selecteds,
        )
        pipeline.set_output_queue(queue)

        debug_state = {}
        restore_callbacks = (
            self._install_live_debug_hooks(pipeline, debug_state) if debug else []
        )

        text, refs, plot, plot_gr = "", "", None, gr.update(visible=False)
        try:
            for response in pipeline.stream(message, conversation_id or "", history or []):
                if not isinstance(response, Document):
                    continue
                if response.channel is None:
                    continue
                if response.channel == "chat":
                    if response.content is None:
                        text = ""
                    else:
                        text += response.content
                if response.channel == "info":
                    if response.content is None:
                        refs = ""
                    else:
                        refs += response.content
                if response.channel == "plot":
                    plot = response.content
                    plot_gr = self._json_to_plot(plot)

                chat_state[pipeline.get_info()["id"]] = reasoning_state["pipeline"]
                yield {
                    "response": text,
                    "info_panel": refs,
                    "plot": plot,
                    "plot_gr": plot_gr,
                    "chat_state": chat_state,
                    "pipeline": pipeline,
                    "debug": None,
                }
        finally:
            for restore in reversed(restore_callbacks):
                restore()

        if not debug:
            return

        debug_payload = None
        debug_payload = self._build_live_debug_payload(
            sample_id or "sample",
            debug_mode,
            message,
            pipeline,
            debug_state,
            text,
            selected_info=selected_info,
            selected_pipeline=pipeline.__class__.__name__,
            selected_reasoning=(
                pipeline.get_info().get("id")
                if hasattr(pipeline, "get_info")
                else pipeline.__class__.__name__
            ),
        )
        yield {
            "response": text,
            "info_panel": refs,
            "plot": plot,
            "plot_gr": plot_gr,
            "chat_state": chat_state,
            "pipeline": pipeline,
            "debug": debug_payload,
        }

    def run_ui_chat_query(
        self,
        message: str,
        history: list | None,
        conversation_id: str | None,
        settings: dict,
        reasoning_type: str,
        llm_type: str,
        use_mind_map: bool | str,
        use_citation: str,
        language: str,
        chat_state: dict,
        command_state: str | None,
        user_id: int,
        *selecteds,
        debug: bool = False,
        sample_id: str | None = None,
        debug_mode: str = "normal_chat_ui",
    ) -> dict:
        """Run the same live Chat UI pipeline to completion and return final artifacts."""
        result = {}
        for result in self.stream_ui_chat_query(
            message,
            history,
            conversation_id,
            settings,
            reasoning_type,
            llm_type,
            use_mind_map,
            use_citation,
            language,
            chat_state,
            command_state,
            user_id,
            *selecteds,
            debug=debug,
            sample_id=sample_id,
            debug_mode=debug_mode,
        ):
            pass
        return result

    def create_pipeline(
        self,
        settings: dict,
        session_reasoning_type: str,
        session_llm: str,
        session_use_mindmap: bool | str,
        session_use_citation: str,
        session_language: str,
        state: dict,
        command_state: str | None,
        user_id: int,
        *selecteds,
    ):
        """Create the pipeline from settings

        Args:
            settings: the settings of the app
            state: the state of the app
            selected: the list of file ids that will be served as context. If None, then
                consider using all files

        Returns:
            - the pipeline objects
        """
        # override reasoning_mode by temporary chat page state
        print(
            "Session reasoning type",
            session_reasoning_type,
            "use mindmap",
            session_use_mindmap,
            "use citation",
            session_use_citation,
            "language",
            session_language,
        )
        print("Session LLM", session_llm)
        reasoning_mode = (
            settings["reasoning.use"]
            if session_reasoning_type in (DEFAULT_SETTING, None)
            else session_reasoning_type
        )
        reasoning_cls = reasonings[reasoning_mode]
        print("Reasoning class", reasoning_cls)
        reasoning_id = reasoning_cls.get_info()["id"]

        settings = deepcopy(settings)
        llm_setting_key = f"reasoning.options.{reasoning_id}.llm"
        if llm_setting_key in settings and session_llm not in (
            DEFAULT_SETTING,
            None,
            "",
        ):
            settings[llm_setting_key] = session_llm

        if session_use_mindmap not in (DEFAULT_SETTING, None):
            settings["reasoning.options.simple.create_mindmap"] = session_use_mindmap

        if session_use_citation not in (DEFAULT_SETTING, None):
            settings[
                "reasoning.options.simple.highlight_citation"
            ] = session_use_citation

        if session_language not in (DEFAULT_SETTING, None):
            settings["reasoning.lang"] = session_language

        # get retrievers
        retrievers = []

        if command_state == WEB_SEARCH_COMMAND:
            # set retriever for web search
            if not WebSearch:
                raise ValueError("Web search back-end is not available.")

            web_search = WebSearch()
            retrievers.append(web_search)
        else:
            for index in self._app.index_manager.indices:
                index_selected = []
                if isinstance(index.selector, int):
                    index_selected = selecteds[index.selector]
                if isinstance(index.selector, tuple):
                    for i in index.selector:
                        index_selected.append(selecteds[i])
                iretrievers = index.get_retriever_pipelines(
                    settings, user_id, index_selected
                )
                retrievers += iretrievers

        # prepare states
        reasoning_state = {
            "app": deepcopy(state["app"]),
            "pipeline": deepcopy(state.get(reasoning_id, {})),
        }

        pipeline = reasoning_cls.get_pipeline(settings, reasoning_state, retrievers)

        return pipeline, reasoning_state

    def chat_fn(
        self,
        conversation_id,
        chat_history,
        settings,
        reasoning_type,
        llm_type,
        use_mind_map,
        use_citation,
        language,
        chat_state,
        command_state,
        user_id,
        *selecteds,
    ):
        """Chat function"""
        chat_input, chat_output = chat_history[-1]
        chat_history = chat_history[:-1]

        # if chat_input is empty, assume regen mode
        if chat_output:
            chat_state["app"]["regen"] = True

        text, refs, plot, plot_gr = "", "", None, gr.update(visible=False)
        msg_placeholder = getattr(
            flowsettings, "KH_CHAT_MSG_PLACEHOLDER", "Thinking ..."
        )
        print(msg_placeholder)
        yield (
            chat_history + [(chat_input, text or msg_placeholder)],
            refs,
            plot_gr,
            plot,
            chat_state,
        )

        try:
            for result in self.stream_ui_chat_query(
                chat_input,
                chat_history,
                conversation_id,
                settings,
                reasoning_type,
                llm_type,
                use_mind_map,
                use_citation,
                language,
                chat_state,
                command_state,
                user_id,
                *selecteds,
                debug=True,
                sample_id="normal_chat_ui",
                debug_mode="normal_chat_ui",
            ):
                text = result["response"]
                refs = result["info_panel"]
                plot = result["plot"]
                plot_gr = result["plot_gr"]
                chat_state = result["chat_state"]
                if result.get("debug"):
                    try:
                        self._write_live_chat_debug(result["debug"])
                    except Exception as exc:
                        print(f"Failed to write live chat debug log: {exc}")
                yield (
                    chat_history + [(chat_input, text or msg_placeholder)],
                    refs,
                    plot_gr,
                    plot,
                    chat_state,
                )
        except ValueError as e:
            print(e)

        if not text:
            empty_msg = getattr(
                flowsettings, "KH_CHAT_EMPTY_MSG_PLACEHOLDER", "(Sorry, I don't know)"
            )
            print(f"Generate nothing: {empty_msg}")
            yield (
                chat_history + [(chat_input, text or empty_msg)],
                refs,
                plot_gr,
                plot,
                chat_state,
            )

    def check_and_suggest_name_conv(self, chat_history):
        suggest_pipeline = SuggestConvNamePipeline()
        new_name = gr.update()
        renamed = False

        # check if this is a newly created conversation
        if len(chat_history) == 1:
            suggested_name = suggest_pipeline(chat_history).text
            suggested_name = strip_think_tag(suggested_name)
            suggested_name = suggested_name.replace('"', "").replace("'", "")[:40]
            new_name = gr.update(value=suggested_name)
            renamed = True

        return new_name, renamed

    def suggest_chat_conv(
        self,
        settings,
        session_language,
        chat_history,
        use_suggestion,
    ):
        target_language = (
            session_language
            if session_language not in (DEFAULT_SETTING, None)
            else settings["reasoning.lang"]
        )
        if use_suggestion:
            suggest_pipeline = SuggestFollowupQuesPipeline()
            suggest_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(
                target_language, "English"
            )
            suggested_questions = [[each] for each in ChatSuggestion.CHAT_SAMPLES]

            if len(chat_history) >= 1:
                suggested_resp = suggest_pipeline(chat_history).text
                if ques_res := re.search(
                    r"\[(.*?)\]", re.sub("\n", "", suggested_resp)
                ):
                    ques_res_str = ques_res.group()
                    try:
                        suggested_questions = json.loads(ques_res_str)
                        suggested_questions = [[x] for x in suggested_questions]
                    except Exception:
                        pass

            return gr.update(visible=True), suggested_questions

        return gr.update(visible=False), gr.update()
