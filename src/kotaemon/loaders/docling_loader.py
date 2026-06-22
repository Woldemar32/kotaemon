import base64
import json
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from kotaemon.base import Document, Param

from .azureai_document_intelligence_loader import crop_image
from .base import BaseReader
from .utils.adobe import generate_single_figure_caption, make_markdown_table


_BROKEN_WORD = re.compile(
    r"([A-Za-zÄÖÜäöüß]+)-\s+([A-Za-zÄÖÜäöüß]+)"
)


def _repair_broken_word(match: re.Match[str]) -> str:
    left, right = match.groups()
    if right.lower() in {"und", "oder", "and", "or", "bzw"}:
        return f"{left}- {right}"
    return f"{left}{right}"


def _clean_table_text(value: object) -> str:
    """Repair common PDF line-wrap artefacts without changing table semantics."""

    text = str(value or "").replace("\u00ad", "")
    text = _BROKEN_WORD.sub(_repair_broken_word, text)
    text = re.sub(r"(?<=\s)-\s+(?=[A-Za-zÄÖÜäöüß])", "-", text)
    return " ".join(text.split())


def _table_structure(table_obj: dict) -> str:
    """Keep Docling's cell topology in vector-store-safe JSON metadata.

    Metadata backends used by Kotaemon only accept scalar values, therefore the
    structure is serialized instead of storing nested lists directly.
    """

    rows = []
    for row_index, row in enumerate(table_obj.get("data", {}).get("grid", [])):
        cells = []
        for column_index, cell in enumerate(row):
            compact = {
                "text": _clean_table_text(cell.get("text", "")),
                "row": row_index,
                "column": column_index,
            }
            # Docling versions use slightly different names for cell spans and
            # header roles. Preserve whichever fields are available.
            for key in (
                "start_row_offset_idx",
                "end_row_offset_idx",
                "start_col_offset_idx",
                "end_col_offset_idx",
                "row_span",
                "col_span",
                "column_header",
                "row_header",
            ):
                value = cell.get(key)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    compact[key] = value
            cells.append(compact)
        rows.append(cells)
    return json.dumps(
        {"version": 1, "rows": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _page_element_payload(text_obj: dict, index: int) -> dict:
    """Return scalar-safe layout evidence consumed by ingestion v2."""

    provenance = (text_obj.get("prov") or [{}])[0]
    bbox = provenance.get("bbox") or {}
    compact_bbox = {
        key: value
        for key, value in bbox.items()
        if key in {"l", "t", "r", "b", "coord_origin"}
        and isinstance(value, (str, int, float, bool))
    }
    return {
        "id": f"text-{index}",
        "label": str(text_obj.get("label") or "text"),
        "text": str(text_obj.get("text") or ""),
        "bbox": compact_bbox,
    }


def _nearest_section_heading(table_obj: dict, text_objs: list[dict]) -> str:
    """Return the closest section header above a Docling table.

    Docling exports tables separately from page text.  Retaining this layout link
    prevents two tables on the same page from both inheriting the page's final
    heading later in the indexing pipeline.
    """

    table_prov = (table_obj.get("prov") or [{}])[0]
    table_page = table_prov.get("page_no")
    table_bbox = table_prov.get("bbox") or {}
    if not table_bbox:
        return ""
    table_origin = table_bbox.get("coord_origin", "BOTTOMLEFT")
    candidates: list[tuple[float, str]] = []
    for text_obj in text_objs:
        if text_obj.get("label") != "section_header":
            continue
        text_prov = (text_obj.get("prov") or [{}])[0]
        if text_prov.get("page_no") != table_page:
            continue
        text_bbox = text_prov.get("bbox") or {}
        if text_bbox.get("coord_origin", table_origin) != table_origin:
            continue
        if table_origin == "TOPLEFT":
            distance = float(table_bbox.get("t", 0)) - float(text_bbox.get("b", 0))
        else:
            distance = float(text_bbox.get("b", 0)) - float(table_bbox.get("t", 0))
        if distance >= -1.0:
            candidates.append((max(0.0, distance), str(text_obj.get("text") or "")))
    return min(candidates, default=(0.0, ""), key=lambda item: item[0])[1].strip()


class DoclingReader(BaseReader):
    """Using Docling to extract document structure and content"""

    _dependencies = ["docling"]

    vlm_endpoint: str = Param(
        help=(
            "Default VLM endpoint for figure captioning. "
            "If not provided, will not caption the figures"
        )
    )

    max_figure_to_caption: int = Param(
        100,
        help=(
            "The maximum number of figures to caption. "
            "The rest will be indexed without captions."
        ),
    )

    figure_friendly_filetypes: list[str] = Param(
        [".pdf", ".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".heif", ".tif"],
        help=(
            "File types that we can reliably open and extract figures. "
            "For files like .docx or .html, the visual layout may be different "
            "when viewed from different tools, hence we cannot use Azure DI location "
            "to extract figures."
        ),
    )

    @Param.auto(cache=True)
    def converter_(self):
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError("Please install docling: 'pip install docling'")

        return DocumentConverter()

    def run(
        self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
    ) -> List[Document]:
        return self.load_data(file_path, extra_info, **kwargs)

    def load_data(
        self, file_path: str | Path, extra_info: Optional[dict] = None, **kwargs
    ) -> List[Document]:
        """Extract the input file, allowing multi-modal extraction"""

        metadata = extra_info or {}

        result = self.converter_.convert(file_path)
        result_dict = result.document.export_to_dict()

        file_path = Path(file_path)
        file_name = file_path.name

        # extract the figures
        figures = []
        gen_caption_count = 0
        for figure_obj in result_dict.get("pictures", []):
            if not self.vlm_endpoint:
                continue
            if file_path.suffix.lower() not in self.figure_friendly_filetypes:
                continue

            # retrieve extractive captions provided by docling
            caption_refs = [caption["$ref"] for caption in figure_obj["captions"]]
            extractive_captions = []
            for caption_ref in caption_refs:
                text_id = caption_ref.split("/")[-1]
                try:
                    caption_text = result_dict["texts"][int(text_id)]["text"]
                    extractive_captions.append(caption_text)
                except (ValueError, TypeError, IndexError) as e:
                    print(e)
                    continue

            # read & crop image
            page_number = figure_obj["prov"][0]["page_no"]

            try:
                page_number_text = str(page_number)
                page_width = result_dict["pages"][page_number_text]["size"]["width"]
                page_height = result_dict["pages"][page_number_text]["size"]["height"]

                bbox_obj = figure_obj["prov"][0]["bbox"]
                bbox: list[float] = [
                    bbox_obj["l"],
                    bbox_obj["t"],
                    bbox_obj["r"],
                    bbox_obj["b"],
                ]
                if bbox_obj["coord_origin"] == "BOTTOMLEFT":
                    bbox = self._convert_bbox_bl_tl(bbox, page_width, page_height)

                img = crop_image(file_path, bbox, page_number - 1)
            except KeyError as e:
                print(e, list(result_dict["pages"].keys()))
                continue

            # convert img to base64
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
            img_base64 = f"data:image/png;base64,{img_base64}"

            # generate the generative caption
            if gen_caption_count >= self.max_figure_to_caption:
                gen_caption = ""
            else:
                gen_caption_count += 1
                gen_caption = generate_single_figure_caption(
                    figure=img_base64, vlm_endpoint=self.vlm_endpoint
                )

            # join the extractive and generative captions
            caption = "\n".join(extractive_captions + [gen_caption])

            # store the image into document
            figure_metadata = {
                "image_origin": img_base64,
                "type": "image",
                "page_label": page_number,
                "file_name": file_name,
                "file_path": file_path,
            }
            figure_metadata.update(metadata)

            figures.append(
                Document(
                    text=caption,
                    metadata=figure_metadata,
                )
            )

        # extract the tables
        tables = []
        for table_index, table_obj in enumerate(result_dict.get("tables", [])):
            # convert the tables into markdown format
            markdown_table = self._parse_table(table_obj)
            caption_refs = [caption["$ref"] for caption in table_obj["captions"]]

            extractive_captions = []
            for caption_ref in caption_refs:
                text_id = caption_ref.split("/")[-1]
                try:
                    caption_text = result_dict["texts"][int(text_id)]["text"]
                    extractive_captions.append(caption_text)
                except (ValueError, TypeError, IndexError) as e:
                    print(e)
                    continue
            # join the extractive and generative captions
            caption = "\n".join(extractive_captions)
            markdown_table = f"{caption}\n{markdown_table}"

            page_number = table_obj["prov"][0].get("page_no", 1)
            section_heading = _nearest_section_heading(
                table_obj, result_dict.get("texts", [])
            )

            table_metadata = {
                "type": "table",
                "page_label": page_number,
                "table_origin": markdown_table,
                "docling_table_index": table_index,
                "docling_table_structure": _table_structure(table_obj),
                "section_heading": section_heading,
                "table_heading": section_heading,
                "file_name": file_name,
                "file_path": file_path,
            }
            table_metadata.update(metadata)

            tables.append(
                Document(
                    text=markdown_table,
                    metadata=table_metadata,
                )
            )

        # join plain text elements
        texts = []
        page_number_to_text = defaultdict(list)
        page_number_to_elements = defaultdict(list)

        for text_index, text_obj in enumerate(result_dict["texts"]):
            page_number = text_obj["prov"][0].get("page_no", 1)
            page_number_to_text[page_number].append(text_obj["text"])
            page_number_to_elements[page_number].append(
                _page_element_payload(text_obj, text_index)
            )

        for page_number, txts in page_number_to_text.items():
            texts.append(
                Document(
                    text="\n".join(txts),
                    metadata={
                        "page_label": page_number,
                        "file_name": file_name,
                        "file_path": file_path,
                        "docling_page_elements": json.dumps(
                            page_number_to_elements[page_number],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        **metadata,
                    },
                )
            )

        return texts + tables + figures

    def _convert_bbox_bl_tl(
        self, bbox: list[float], page_width: int, page_height: int
    ) -> list[float]:
        """Convert bbox from bottom-left to top-left"""
        x0, y0, x1, y1 = bbox
        return [
            x0 / page_width,
            (page_height - y1) / page_height,
            x1 / page_width,
            (page_height - y0) / page_height,
        ]

    def _parse_table(self, table_obj: dict) -> str:
        """Convert docling table object to markdown table"""
        table_as_list: List[List[str]] = []
        grid = table_obj["data"]["grid"]
        for row in grid:
            table_as_list.append([])
            for cell in row:
                table_as_list[-1].append(_clean_table_text(cell["text"]))

        return make_markdown_table(table_as_list)
