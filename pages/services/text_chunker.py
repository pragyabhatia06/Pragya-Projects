from typing import List, Dict


def chunk_text(
    pages: List[Dict],
    file_name: str,
    chunk_size: int = 800,
    overlap: int = 150
) -> List[Dict]:
    """
    Split page text into chunks with metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    # Ensure forward progress even with invalid overlap values.
    step = max(1, chunk_size - max(0, overlap))

    chunks = []
    chunk_id = 1

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        if not text:
            continue

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text_value = text[start:end]

            chunks.append({
                "chunk_id": f"{file_name}_page_{page_number}_chunk_{chunk_id}",
                "file_name": file_name,
                "page_number": page_number,
                "chunk_index": chunk_id,
                "text": chunk_text_value
            })

            chunk_id += 1
            start += step

    return chunks