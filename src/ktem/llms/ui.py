from copy import deepcopy

import gradio as gr
import pandas as pd
import yaml
from ktem.app import BasePage
from ktem.utils.file import YAMLNoDateSafeLoader
from theflow.utils.modules import deserialize

from ktem.embeddings.manager import embedding_models_manager
from ktem.rerankings.manager import reranking_models_manager

from .manager import llms


def format_description(cls):
    params = cls.describe()["params"]
    params_lines = ["| Name | Type | Description |", "| --- | --- | --- |"]
    for key, value in params.items():
        if isinstance(value["auto_callback"], str):
            continue
        params_lines.append(f"| {key} | {value['type']} | {value['help']} |")
    return f"{cls.__doc__}\n\n" + "\n".join(params_lines)


class LocalModelAddBlock(BasePage):
    """Shared Resources UI for adding local LLM, embedding, and reranker specs."""

    MODEL_TYPES = ["LLM", "Embedding", "Reranker"]
    RUNTIMES = ["Localhost", "Docker"]

    def __init__(self, app, llm_page=None, embedding_page=None, reranking_page=None):
        self._app = app
        self._model_pages = {
            "llm": llm_page,
            "embedding": embedding_page,
            "reranking": reranking_page,
        }
        self.on_building_ui()

    def on_building_ui(self):
        gr.Markdown("### Add Local Model")
        with gr.Row():
            self.local_runtime = gr.Radio(
                self.RUNTIMES,
                value="Localhost",
                label="Runtime",
            )
            self.local_model_type = gr.Dropdown(
                self.MODEL_TYPES,
                value="LLM",
                label="Model Type",
            )
        self.local_model_name = gr.Textbox(
            label="Model Name",
            placeholder="e.g., qwen3:8b",
        )
        self.btn_add_local_model = gr.Button("Add", variant="primary")
        self.local_model_status = gr.Label(label="Status")

    @staticmethod
    def _base_urls(runtime: str) -> tuple[str, str]:
        if runtime == "Docker":
            return (
                "http://host.docker.internal:11434/v1",
                "http://host.docker.internal:8080",
            )
        return "http://localhost:11434/v1", "http://localhost:8080"

    @classmethod
    def _local_spec(cls, runtime: str, model_type: str, model_name: str) -> dict:
        base_llm, base_tei = cls._base_urls(runtime)
        if model_type == "LLM":
            return {
                "__type__": "kotaemon.llms.ChatOpenAI",
                "api_key": "ollama",
                "base_url": base_llm,
                "model": model_name,
                "temperature": 0,
                "timeout": 120,
            }
        if model_type == "Embedding":
            return {
                "__type__": "kotaemon.embeddings.OpenAIEmbeddings",
                "api_key": "ollama",
                "base_url": base_llm,
                "model": model_name,
                "timeout": 600,
            }
        if model_type == "Reranker":
            return {
                "__type__": "kotaemon.rerankings.TeiFastReranking",
                "endpoint_url": base_tei,
                "model_name": model_name,
                "is_truncated": True,
            }
        raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _manager_for(model_type: str):
        if model_type == "LLM":
            return llms
        if model_type == "Embedding":
            return embedding_models_manager
        if model_type == "Reranker":
            return reranking_models_manager
        raise ValueError(f"Unknown model type: {model_type}")

    def add_local_model(self, runtime: str, model_type: str, model_name: str):
        try:
            model_name = (model_name or "").strip()
            if not model_name:
                raise ValueError("Model name cannot be empty")

            spec = self._local_spec(runtime, model_type, model_name)
            manager = self._manager_for(model_type)
            manager.add(model_name, spec=spec, default=True)
            return f"✅ {model_type} '{model_name}' added and set as default"
        except ValueError as exc:
            return f"❌ {exc}"
        except Exception as exc:
            return f"❌ Failed to add model: {exc}"

    def on_register_events(self):
        event = self.btn_add_local_model.click(
            self.add_local_model,
            inputs=[self.local_runtime, self.local_model_type, self.local_model_name],
            outputs=[self.local_model_status],
        )
        llm_page = self._model_pages["llm"]
        embedding_page = self._model_pages["embedding"]
        reranking_page = self._model_pages["reranking"]

        if llm_page is not None:
            event = event.then(
                llm_page.list_llms,
                inputs=[],
                outputs=[llm_page.llm_list],
            )
        if embedding_page is not None:
            event = event.then(
                embedding_page.list_embeddings,
                inputs=[],
                outputs=[embedding_page.emb_list],
            )
        if reranking_page is not None:
            event.then(
                reranking_page.list_rerankings,
                inputs=[],
                outputs=[reranking_page.rerank_list],
            )


class LLMManagement(BasePage):
    def __init__(self, app):
        self._app = app
        self.spec_desc_default = (
            "# Spec description\n\nSelect an LLM to view the spec description."
        )
        self.on_building_ui()

    def on_building_ui(self):
        with gr.Tab(label="View"):
            self.llm_list = gr.DataFrame(
                headers=["name", "adapter", "default"],
                interactive=False,
                column_widths=[30, 40, 30],
            )

            with gr.Column(visible=False) as self._selected_panel:
                self.selected_llm_name = gr.Textbox(value="", visible=False)
                with gr.Row():
                    with gr.Column():
                        self.edit_default = gr.Checkbox(
                            label="Set default",
                            info=(
                                "Set this LLM as default. If no default is set, "
                                "a random LLM will be used. "
                                "This default LLM will be used by other components "
                                "by default if no LLM is specified for such components."
                            ),
                        )
                        self.edit_name = gr.Textbox(
                            label="Name",
                            info="Edit to rename this LLM.",
                        )
                        self.edit_spec = gr.Textbox(
                            label="Specification",
                            info="Specification of the LLM in YAML format",
                            lines=10,
                        )

                        with gr.Accordion(
                            label="Test connection", visible=False, open=False
                        ) as self._check_connection_panel:
                            with gr.Row():
                                with gr.Column(scale=4):
                                    self.connection_logs = gr.HTML("Logs")

                                with gr.Column(scale=1):
                                    self.btn_test_connection = gr.Button(
                                        "Test",
                                    )

                        with gr.Row(visible=False) as self._selected_panel_btn:
                            with gr.Column():
                                self.btn_edit_save = gr.Button(
                                    "Save", min_width=10, variant="primary"
                                )
                            with gr.Column():
                                self.btn_delete = gr.Button(
                                    "Delete", min_width=10, variant="stop"
                                )
                                with gr.Row():
                                    self.btn_delete_yes = gr.Button(
                                        "Confirm Delete",
                                        variant="stop",
                                        visible=False,
                                        min_width=10,
                                    )
                                    self.btn_delete_no = gr.Button(
                                        "Cancel", visible=False, min_width=10
                                    )
                            with gr.Column():
                                self.btn_close = gr.Button("Close", min_width=10)

                    with gr.Column():
                        self.edit_spec_desc = gr.Markdown("# Spec description")

        with gr.Tab(label="Advanced local YAML"):
            with gr.Row():
                with gr.Column(scale=2):
                    self.name = gr.Textbox(
                        label="LLM name",
                        info=(
                            "Must be unique. The name will be used to identify the LLM."
                        ),
                    )
                    gr.Markdown(
                        (
                            "**Provider:** OpenAI-compatible local endpoint "
                            "(`kotaemon.llms.ChatOpenAI`).\n\n"
                            "Example:\n"
                            "```yaml\n"
                            "api_key: ollama\n"
                            "base_url: http://localhost:11434/v1\n"
                            "model: qwen3:8b\n"
                            "temperature: 0\n"
                            "timeout: 120\n"
                            "```"
                        )
                    )
                    self.spec = gr.Textbox(
                        label="Specification",
                        info="Specification of the LLM in YAML format",
                        lines=7,
                    )
                    self.default = gr.Checkbox(
                        label="Set default",
                        info=(
                            "Set this LLM as default. This default LLM will be used "
                            "by default across the application."
                        ),
                    )
                    self.btn_new = gr.Button("Add LLM", variant="primary")

                with gr.Column(scale=3):
                    self.spec_desc = gr.Markdown(self.spec_desc_default)

    def _on_app_created(self):
        """Called when the app is created"""
        self._app.app.load(
            self.list_llms,
            inputs=[],
            outputs=[self.llm_list],
        )

    def on_register_events(self):
        self.btn_new.click(
            self.create_llm,
            inputs=[self.name, self.spec, self.default],
            outputs=[],
        ).success(self.list_llms, inputs=[], outputs=[self.llm_list]).success(
            lambda: ("", "", False, self.spec_desc_default),
            outputs=[
                self.name,
                self.spec,
                self.default,
                self.spec_desc,
            ],
        )
        self.llm_list.select(
            self.select_llm,
            inputs=self.llm_list,
            outputs=[self.selected_llm_name],
            show_progress="hidden",
        )
        self.selected_llm_name.change(
            self.on_selected_llm_change,
            inputs=[self.selected_llm_name],
            outputs=[
                self._selected_panel,
                self._selected_panel_btn,
                # delete section
                self.btn_delete,
                self.btn_delete_yes,
                self.btn_delete_no,
                # edit section
                self.edit_name,
                self.edit_spec,
                self.edit_spec_desc,
                self.edit_default,
            ],
            show_progress="hidden",
        ).success(lambda: gr.update(value=""), outputs=[self.connection_logs])

        self.btn_delete.click(
            self.on_btn_delete_click,
            inputs=[],
            outputs=[self.btn_delete, self.btn_delete_yes, self.btn_delete_no],
            show_progress="hidden",
        )
        self.btn_delete_yes.click(
            self.delete_llm,
            inputs=[self.selected_llm_name],
            outputs=[self.selected_llm_name],
            show_progress="hidden",
        ).then(
            self.list_llms,
            inputs=[],
            outputs=[self.llm_list],
        )
        self.btn_delete_no.click(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            ),
            inputs=[],
            outputs=[self.btn_delete, self.btn_delete_yes, self.btn_delete_no],
            show_progress="hidden",
        )
        self.btn_edit_save.click(
            self.save_llm,
            inputs=[
                self.selected_llm_name,
                self.edit_name,
                self.edit_default,
                self.edit_spec,
            ],
            outputs=[self.selected_llm_name],
            show_progress="hidden",
        ).then(
            self.list_llms,
            inputs=[],
            outputs=[self.llm_list],
        )
        self.btn_close.click(
            lambda: "",
            outputs=[self.selected_llm_name],
        )

        self.btn_test_connection.click(
            self.check_connection,
            inputs=[self.selected_llm_name, self.edit_spec],
            outputs=[self.connection_logs],
        )

    def create_llm(self, name, spec, default):
        try:
            name = name.strip()
            spec = yaml.load(spec, Loader=YAMLNoDateSafeLoader) or {}
            spec["__type__"] = "kotaemon.llms.ChatOpenAI"

            llms.add(name, spec=spec, default=default)
            gr.Info(f"LLM '{name}' created successfully")
        except ValueError as e:
            raise gr.Error(str(e))
        except Exception as e:
            raise gr.Error(f"Failed to create LLM '{name}': {e}")

    def list_llms(self):
        """List the LLMs"""
        items = []
        for item in llms.info().values():
            record = {}
            record["name"] = item["name"]
            record["adapter"] = item["spec"].get("__type__", "-").split(".")[-1]
            record["default"] = item["default"]
            items.append(record)

        if items:
            llm_list = pd.DataFrame.from_records(items)
        else:
            llm_list = pd.DataFrame.from_records(
                [{"name": "-", "adapter": "-", "default": "-"}]
            )

        return llm_list

    def select_llm(self, llm_list, ev: gr.SelectData):
        if ev.value == "-" and ev.index[0] == 0:
            gr.Info("No LLM is loaded. Please add LLM first")
            return ""

        if not ev.selected:
            return ""

        return llm_list["name"][ev.index[0]]

    def on_selected_llm_change(self, selected_llm_name):
        if selected_llm_name == "":
            _selected_panel = gr.update(visible=False)
            _selected_panel_btn = gr.update(visible=False)
            btn_delete = gr.update(visible=True)
            btn_delete_yes = gr.update(visible=False)
            btn_delete_no = gr.update(visible=False)
            edit_name = gr.update(value="")
            edit_spec = gr.update(value="")
            edit_spec_desc = gr.update(value="")
            edit_default = gr.update(value=False)
        else:
            _selected_panel = gr.update(visible=True)
            _selected_panel_btn = gr.update(visible=True)
            btn_delete = gr.update(visible=True)
            btn_delete_yes = gr.update(visible=False)
            btn_delete_no = gr.update(visible=False)

            info = deepcopy(llms.info()[selected_llm_name])
            vendor_str = info["spec"].pop("__type__", "-").split(".")[-1]
            vendor = llms.vendors().get(vendor_str)

            edit_name = selected_llm_name
            edit_spec = yaml.dump(info["spec"])
            edit_spec_desc = (
                format_description(vendor)
                if vendor is not None
                else f"# Spec description\n\n{vendor_str}"
            )
            edit_default = info["default"]

        return (
            _selected_panel,
            _selected_panel_btn,
            btn_delete,
            btn_delete_yes,
            btn_delete_no,
            edit_name,
            edit_spec,
            edit_spec_desc,
            edit_default,
        )

    def on_btn_delete_click(self):
        btn_delete = gr.update(visible=False)
        btn_delete_yes = gr.update(visible=True)
        btn_delete_no = gr.update(visible=True)

        return btn_delete, btn_delete_yes, btn_delete_no

    def check_connection(self, selected_llm_name: str, selected_spec):
        log_content: str = ""

        try:
            log_content += f"- Testing model: {selected_llm_name}<br>"
            yield log_content

            # Parse content & init model
            info = deepcopy(llms.info()[selected_llm_name])

            # Parse content & create dummy embedding
            spec = yaml.load(selected_spec, Loader=YAMLNoDateSafeLoader)
            info["spec"].update(spec)

            llm = deserialize(info["spec"], safe=False)

            if llm is None:
                raise Exception(f"Can not found model: {selected_llm_name}")

            log_content += "- Sending a message `Hi`<br>"
            yield log_content
            respond = llm("Hi")

            log_content += (
                f"<mark style='background: green; color: white'>- Connection success. "
                f"Got response:\n {respond}</mark><br>"
            )
            yield log_content

            gr.Info(f"LLM {selected_llm_name} connect successfully")
        except Exception as e:
            log_content += (
                f"<mark style='color: yellow; background: red'>- Connection failed. "
                f"Got error:\n {e}</mark>"
            )
            yield log_content

        return log_content

    def save_llm(self, selected_llm_name, edit_name, default, spec):
        try:
            new_name = edit_name.strip()
            spec = yaml.load(spec, Loader=YAMLNoDateSafeLoader)
            spec["__type__"] = llms.info()[selected_llm_name]["spec"]["__type__"]
            llms.update(
                selected_llm_name, spec=spec, default=default, new_name=new_name
            )
            final_name = (
                new_name if new_name != selected_llm_name else selected_llm_name
            )
            gr.Info(f"LLM '{final_name}' saved successfully")
            return final_name
        except ValueError as e:
            raise gr.Error(str(e))
        except Exception as e:
            raise gr.Error(f"Failed to save LLM '{selected_llm_name}': {e}")

    def delete_llm(self, selected_llm_name):
        try:
            llms.delete(selected_llm_name)
        except Exception as e:
            gr.Error(f"Failed to delete LLM {selected_llm_name}: {e}")
            return selected_llm_name

        return ""
