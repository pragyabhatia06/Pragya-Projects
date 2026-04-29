import os

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def _looks_like_pdf(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
        return header.startswith(b"%PDF-")
    except OSError:
        return False


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extract text page by page from a PDF.
    Returns page-level text with metadata.
    """

    pages = []
    parser_errors = []

    if not file_path:
        raise ValueError("No file path provided for PDF extraction.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise ValueError("Uploaded PDF is empty.")

    if not _looks_like_pdf(file_path):
        raise ValueError("Uploaded file is not a valid PDF (missing %PDF header).")

    if fitz is not None:
        try:
            with fitz.open(file_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text")
                    pages.append({
                        "page_number": page_number,
                        "text": text
                    })
        except Exception as exc:
            # Fall back to pypdf parser when PyMuPDF cannot open/safely parse a file.
            pages = []
            parser_errors.append(f"PyMuPDF: {exc}")

    if not pages and PdfReader is not None:
        try:
            pdf = PdfReader(file_path, strict=False)

            if pdf.is_encrypted:
                decrypt_status = pdf.decrypt("")
                if decrypt_status == 0:
                    raise ValueError("Encrypted PDF is not supported without a password.")

            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": page_number,
                    "text": text
                })
        except Exception as exc:
            pages = []
            parser_errors.append(f"pypdf: {exc}")

    if not pages and fitz is None and PdfReader is None:
        raise ImportError(
            "No PDF parser is installed. Install 'pymupdf' or 'pypdf' to extract PDF text."
        )

    if not pages:
        error_suffix = f" Parser details: {' | '.join(parser_errors)}" if parser_errors else ""
        raise ValueError(
            "Unable to extract text from PDF. The file may be corrupted, encrypted, or unsupported."
            f"{error_suffix}"
        )

    return {
        "total_pages": len(pages),
        "pages": pages
    }