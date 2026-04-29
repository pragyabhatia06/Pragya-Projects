import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from pages.services.pdf_extractor import extract_text_from_pdf
from pages.services.text_cleaner import clean_text
from pages.services.text_chunker import chunk_text
from pages.services.embedding_service import EmbeddingService
from pages.services.vector_store import VectorStore
from pages.services.rag_service import build_context_from_results, generate_basic_rag_answer


st.set_page_config(
    page_title="LLM Data Ingestion Pipeline",
    layout="wide"
)

UPLOAD_DIR = "data/uploaded_docs"
VECTOR_DB_DIR = "data/chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


st.title("LLM Data Ingestion & RAG Pipeline")
st.caption(
    "Data Engineering perspective: PDF ingestion, metadata capture, text extraction, "
    "cleaning, chunking, embeddings, vector database, semantic search and RAG."
)

if "last_embed_latency_s" not in st.session_state:
    st.session_state.last_embed_latency_s = None

if "last_inserted_count" not in st.session_state:
    st.session_state.last_inserted_count = 0


with st.sidebar:
    st.header("Pipeline Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=300,
        max_value=2000,
        value=800,
        step=100
    )

    overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=150,
        step=50
    )

    top_k = st.slider(
        "Top-K Search Results",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()
    st.subheader("Governance & SLA")

    pipeline_owner = st.text_input("Pipeline Owner", value="Data Engineering Team")
    source_system = st.selectbox(
        "Source System",
        options=["Manual PDF Upload", "SharePoint", "S3", "Azure Blob", "Confluence"],
        index=0
    )
    ingestion_batch_id = st.text_input(
        "Ingestion Batch ID",
        value=f"batch-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    target_embed_latency_s = st.slider(
        "Target Embed Latency (seconds)",
        min_value=1,
        max_value=120,
        value=25
    )
    max_empty_chunk_pct = st.slider(
        "Max Empty Chunk %",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1
    )
    max_duplicate_chunk_pct = st.slider(
        "Max Duplicate Chunk %",
        min_value=0.0,
        max_value=15.0,
        value=2.0,
        step=0.5
    )

    st.divider()

    clear_db = st.button("Clear Vector Database")


vector_store = VectorStore(
    persist_directory=VECTOR_DB_DIR,
    collection_name="llm_pdf_documents"
)

if clear_db:
    vector_store.delete_collection_data()
    st.success("Vector database cleared.")


st.subheader("1. Upload PDF")

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


if uploaded_file:
    safe_file_name = os.path.basename(uploaded_file.name)
    timestamp_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    stored_file_name = f"{timestamp_suffix}_{safe_file_name}"
    file_path = os.path.join(UPLOAD_DIR, stored_file_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if os.path.getsize(file_path) == 0:
        st.error("Uploaded PDF is empty. Please upload a valid non-empty PDF file.")
        st.stop()

    file_metadata = {
        "file_name": safe_file_name,
        "stored_file_name": stored_file_name,
        "file_size_kb": round(uploaded_file.size / 1024, 2),
        "upload_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_system": source_system,
        "storage_layer": "Raw Document Layer",
        "pipeline_owner": pipeline_owner,
        "ingestion_batch_id": ingestion_batch_id
    }

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("File Name", safe_file_name)
    col2.metric("File Size KB", file_metadata["file_size_kb"])
    col3.metric("Source", file_metadata["source_system"])
    col4.metric("Vector DB Count", vector_store.count())

    with st.expander("Show Metadata & Lineage"):
        st.json(file_metadata)

    st.subheader("2. Data Engineering Pipeline DAG")

    st.graphviz_chart("""
    digraph {
        rankdir=LR;

        UploadPDF [label="Upload PDF"];
        RawStorage [label="Raw Storage"];
        ExtractText [label="Extract Text"];
        CleanText [label="Clean Text"];
        ChunkDocument [label="Chunk Document"];
        DataQuality [label="Data Quality Checks"];
        GenerateEmbeddings [label="Generate Embeddings"];
        StoreVectorDB [label="Store in Vector DB"];
        SemanticSearch [label="Semantic Search"];
        RAGAnswer [label="RAG Answer"];

        UploadPDF -> RawStorage;
        RawStorage -> ExtractText;
        ExtractText -> CleanText;
        CleanText -> ChunkDocument;
        ChunkDocument -> DataQuality;
        DataQuality -> GenerateEmbeddings;
        GenerateEmbeddings -> StoreVectorDB;
        StoreVectorDB -> SemanticSearch;
        SemanticSearch -> RAGAnswer;
    }
    """)

    st.subheader("3. Extract Text")

    try:
        extracted_pdf = extract_text_from_pdf(file_path)
    except Exception as exc:
        st.error(f"PDF extraction failed: {exc}")
        st.stop()

    st.write(f"Total pages extracted: **{extracted_pdf['total_pages']}**")

    raw_pages = extracted_pdf["pages"]
    extracted_chars = sum(len((page.get("text") or "").strip()) for page in raw_pages)
    if extracted_chars == 0:
        st.error(
            "No selectable text found in this PDF. It may be a scanned/image-only document. "
            "Run OCR first, then upload the OCR-processed PDF."
        )
        st.stop()

    cleaned_pages = []

    for page in raw_pages:
        cleaned_pages.append({
            "page_number": page["page_number"],
            "text": clean_text(page["text"])
        })

    with st.expander("Preview Extracted Text"):
        for page in cleaned_pages[:3]:
            st.markdown(f"**Page {page['page_number']}**")
            st.write(page["text"][:1500])

    st.subheader("4. Chunk Document")

    chunks = chunk_text(
        pages=cleaned_pages,
        file_name=safe_file_name,
        chunk_size=chunk_size,
        overlap=overlap
    )

    chunk_df = pd.DataFrame(chunks)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Chunks", len(chunks))
    col2.metric("Chunk Size", chunk_size)
    col3.metric("Overlap", overlap)

    if not chunk_df.empty:
        st.dataframe(
            chunk_df[["chunk_id", "file_name", "page_number", "chunk_index", "text"]],
            use_container_width=True
        )

    st.subheader("5. Data Quality Checks")

    empty_chunks = [c for c in chunks if not c["text"].strip()]
    duplicate_chunks = chunk_df["text"].duplicated().sum() if not chunk_df.empty else 0
    total_chunks = len(chunks)
    empty_chunk_pct = (len(empty_chunks) / total_chunks * 100) if total_chunks else 0.0
    duplicate_chunk_pct = (float(duplicate_chunks) / total_chunks * 100) if total_chunks else 0.0

    token_estimate = sum(len(c["text"]) // 4 for c in chunks)
    avg_chunk_chars = int(chunk_df["text"].str.len().mean()) if not chunk_df.empty else 0
    pages_touched = chunk_df["page_number"].nunique() if not chunk_df.empty else 0

    dq_col1, dq_col2, dq_col3 = st.columns(3)

    dq_col1.metric("Empty Chunks", len(empty_chunks))
    dq_col2.metric("Duplicate Chunks", int(duplicate_chunks))
    dq_col3.metric("Valid Chunks", len(chunks) - len(empty_chunks))

    ops_col1, ops_col2, ops_col3, ops_col4 = st.columns(4)
    ops_col1.metric("Estimated Tokens", f"{token_estimate:,}")
    ops_col2.metric("Avg Chunk Chars", avg_chunk_chars)
    ops_col3.metric("Pages Covered", pages_touched)
    ops_col4.metric("Batch ID", ingestion_batch_id)

    st.write("Quality Threshold Tracking")
    st.progress(
        min(100, int((empty_chunk_pct / max_empty_chunk_pct * 100) if max_empty_chunk_pct else 100)),
        text=f"Empty Chunk %: {empty_chunk_pct:.2f}% (limit {max_empty_chunk_pct:.2f}%)"
    )
    st.progress(
        min(100, int((duplicate_chunk_pct / max_duplicate_chunk_pct * 100) if max_duplicate_chunk_pct else 100)),
        text=f"Duplicate Chunk %: {duplicate_chunk_pct:.2f}% (limit {max_duplicate_chunk_pct:.2f}%)"
    )

    if empty_chunk_pct <= max_empty_chunk_pct and duplicate_chunk_pct <= max_duplicate_chunk_pct:
        st.success("Data quality check passed: No empty chunks found.")
    else:
        st.warning("Data quality threshold breached. Review chunking parameters before indexing.")

    if not chunk_df.empty:
        chunk_profile = (
            chunk_df.assign(chunk_chars=chunk_df["text"].str.len())
            .groupby("page_number", as_index=True)["chunk_chars"]
            .mean()
            .fillna(0)
            .astype(int)
        )
        st.caption("Chunk profile by page")
        st.bar_chart(chunk_profile)

    st.subheader("6. Generate Embeddings & Store in Vector DB")

    if st.button("Run Embedding Pipeline"):
        if not chunks:
            st.error("No chunks available for embedding.")
        else:
            try:
                with st.spinner("Generating embeddings and storing in vector database..."):
                    start_time = time.perf_counter()
                    embedding_service = EmbeddingService()

                    texts = [chunk["text"] for chunk in chunks]
                    embeddings = embedding_service.embed_texts(texts)

                    inserted_count = vector_store.add_chunks(
                        chunks=chunks,
                        embeddings=embeddings
                    )
                    elapsed_s = round(time.perf_counter() - start_time, 2)
                    st.session_state.last_embed_latency_s = elapsed_s
                    st.session_state.last_inserted_count = inserted_count
            except Exception as exc:
                st.error(f"Embedding pipeline failed: {exc}")
                st.stop()

            st.success(f"{inserted_count} chunks stored in Chroma Vector DB.")
            st.metric("Total Vector DB Records", vector_store.count())
            embed_sla_ok = st.session_state.last_embed_latency_s <= target_embed_latency_s
            st.metric(
                "Embedding Runtime (s)",
                st.session_state.last_embed_latency_s,
                delta=f"target {target_embed_latency_s}s"
            )
            if embed_sla_ok:
                st.success("Embedding SLA met.")
            else:
                st.warning("Embedding SLA missed. Consider reducing chunk size or batching strategy.")


st.divider()

st.subheader("7. Semantic Search / RAG Query")

query = st.text_input(
    "Ask a question from uploaded documents",
    placeholder="Example: What is this document about?"
)

if query:
    try:
        embedding_service = EmbeddingService()
        query_embedding = embedding_service.embed_query(query)
    except Exception as exc:
        st.error(f"Query embedding failed: {exc}")
        st.stop()

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k
    )

    context = build_context_from_results(results)
    answer = generate_basic_rag_answer(query, context)

    st.subheader("Retrieved Chunks")

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    result_rows = []

    for doc, meta, distance in zip(documents, metadatas, distances):
        result_rows.append({
            "file_name": meta.get("file_name"),
            "page_number": meta.get("page_number"),
            "chunk_index": meta.get("chunk_index"),
            "similarity_distance": distance,
            "text": doc
        })

    if result_rows:
        avg_distance = sum([r["similarity_distance"] for r in result_rows]) / len(result_rows)
        confidence_score = max(0.0, 1.0 - min(1.0, avg_distance))

        r_col1, r_col2, r_col3 = st.columns(3)
        r_col1.metric("Results Returned", len(result_rows))
        r_col2.metric("Avg Similarity Distance", f"{avg_distance:.4f}")
        r_col3.metric("Retrieval Confidence", f"{confidence_score * 100:.1f}%")

        st.dataframe(
            pd.DataFrame(result_rows),
            use_container_width=True
        )

    st.subheader("RAG-Style Answer")

    st.write(answer)

    with st.expander("Context Passed to LLM"):
        st.write(context)


st.divider()

st.subheader("Data Engineering Talking Points")

st.markdown("""
Technical Data Engineering Flow (RAG):

- Source ingestion: PDF lands in raw storage with batch ID, owner, and source metadata.
- Parsing and normalization: page-level extraction, text cleanup, and schema-like chunk structure.
- Data quality gates: empty/duplicate thresholds, chunk size controls, and coverage metrics.
- Feature generation: chunk embeddings with runtime tracking and SLA visibility.
- Serving layer: vectors + metadata written to Chroma for low-latency semantic retrieval.
- Retrieval operation: query embedding, top-k similarity search, context assembly, and answer draft.
""")