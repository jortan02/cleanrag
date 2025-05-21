from typing import List, Dict, Any, Tuple
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_to_text


@dataclass
class DocumentChunk:
    text: str
    start_idx: int
    end_idx: int
    chunk_id: int


def extract_text_from_file(file) -> str:
    """Extract text from various file formats using unstructured."""
    # Save the uploaded file temporarily
    temp_path = Path("temp") / file.name
    temp_path.parent.mkdir(exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(file.getvalue())

    try:
        # Use unstructured to extract text
        elements = partition(str(temp_path))
        text = elements_to_text(elements)
        return text
    finally:
        # Clean up temporary file
        temp_path.unlink(missing_ok=True)


def process_document(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[DocumentChunk]:
    """Process document text into chunks with position information."""
    chunks = []
    words = text.split()
    current_pos = 0

    for i in range(0, len(words), chunk_size - overlap):
        chunk_text = " ".join(words[i : i + chunk_size])
        start_idx = text.find(chunk_text, current_pos)
        end_idx = start_idx + len(chunk_text)

        chunks.append(
            DocumentChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_id=len(chunks),
            )
        )

        current_pos = end_idx

    return chunks


def get_chunk_statistics(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Get statistics about the chunks."""
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(len(chunk.text.split()) for chunk in chunks)
        / len(chunks),
        "total_words": sum(len(chunk.text.split()) for chunk in chunks),
        "chunk_sizes": [len(chunk.text.split()) for chunk in chunks],
    }


def create_chunk_dataframe(chunks: List[DocumentChunk]) -> pd.DataFrame:
    """Create a DataFrame with chunk information."""
    return pd.DataFrame(
        [
            {
                "Chunk ID": chunk.chunk_id,
                "Text": chunk.text,
                "Word Count": len(chunk.text.split()),
                "Start Position": chunk.start_idx,
                "End Position": chunk.end_idx,
            }
            for chunk in chunks
        ]
    )
