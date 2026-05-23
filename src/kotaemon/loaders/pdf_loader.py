import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from decouple import config
from fsspec import AbstractFileSystem
from llama_index.readers.file import PDFReader
from PIL import Image

from kotaemon.base import Document

PDF_LOADER_DPI = config("PDF_LOADER_DPI", default=40, cast=int)
logger = logging.getLogger(__name__)


def _pdf_log(message: str, level: int = logging.INFO) -> None:
    """Log PDF loading progress to logger and terminal."""

    logger.log(level, message)
    print(f"[pdf-loader] {message}", flush=True)


def get_page_thumbnails(
    file_path: Path, pages: list[int], dpi: int = PDF_LOADER_DPI
) -> List[str]:
    """Get image thumbnails of the pages in the PDF file.

    Args:
        file_path (Path): path to the image file
        page_number (list[int]): list of page numbers to extract

    Returns:
        list[str]: list of page thumbnails encoded as base64 data URLs
    """

    img: Image.Image
    suffix = file_path.suffix.lower()
    assert suffix == ".pdf", "This function only supports PDF files."
    try:
        import fitz
    except ImportError:
        raise ImportError("Please install PyMuPDF: 'pip install PyMuPDF'")

    _pdf_log(
        f"{file_path.name}: rendering {len(pages)} page thumbnails at {dpi} DPI"
    )
    start_time = time.time()
    output_imgs = []
    doc = fitz.open(file_path)
    try:
        for idx, page_number in enumerate(pages, start=1):
            page = doc.load_page(page_number)
            pm = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            output_imgs.append(convert_image_to_base64(img))
            _pdf_log(
                f"{file_path.name}: rendered thumbnail {idx}/{len(pages)} "
                f"(page_index={page_number})"
            )
    finally:
        doc.close()

    _pdf_log(
        f"{file_path.name}: rendered {len(output_imgs)} thumbnails "
        f"in {time.time() - start_time:.2f}s"
    )

    return output_imgs


def convert_image_to_base64(img: Image.Image) -> str:
    # convert the image into base64
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    img_base64 = f"data:image/png;base64,{img_base64}"

    return img_base64


class PDFThumbnailReader(PDFReader):
    """PDF parser with thumbnail for each page."""

    def __init__(self) -> None:
        """
        Initialize PDFReader.
        """
        super().__init__(return_full_document=False)

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        start_time = time.time()
        _pdf_log(f"{file.name}: text extraction started")
        documents = super().load_data(file, extra_info, fs)
        _pdf_log(
            f"{file.name}: text extraction produced {len(documents)} raw documents "
            f"in {time.time() - start_time:.2f}s"
        )

        page_numbers_str = []
        filtered_docs = []
        is_int_page_number: dict[str, bool] = {}

        for doc in documents:
            if "page_label" in doc.metadata:
                page_num_str = doc.metadata["page_label"]
                page_numbers_str.append(page_num_str)
                try:
                    _ = int(page_num_str)
                    is_int_page_number[page_num_str] = True
                    filtered_docs.append(doc)
                except ValueError:
                    is_int_page_number[page_num_str] = False
                    continue

        documents = filtered_docs
        text_page_count = len(documents)
        page_numbers = list(range(len(page_numbers_str)))

        _pdf_log(
            f"{file.name}: page labels found={len(page_numbers_str)}, "
            f"usable_text_pages={len(documents)}"
        )
        print("Page numbers:", len(page_numbers), flush=True)
        page_thumbnails = get_page_thumbnails(file, page_numbers)

        documents.extend(
            [
                Document(
                    text="Page thumbnail",
                    metadata={
                        "image_origin": page_thumbnail,
                        "type": "thumbnail",
                        "page_label": page_number,
                        **(extra_info if extra_info is not None else {}),
                    },
                )
                for (page_thumbnail, page_number) in zip(
                    page_thumbnails, page_numbers_str
                )
                if is_int_page_number[page_number]
            ]
        )

        _pdf_log(
            f"{file.name}: returning {len(documents)} documents "
            f"(text_pages={text_page_count}, thumbnails={len(page_thumbnails)})"
        )
        return documents
