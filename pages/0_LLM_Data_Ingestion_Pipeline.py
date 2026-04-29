import os
import re
import time
from html import escape
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

from pages.services.pdf_extractor import extract_text_from_pdf
from pages.services.text_cleaner import clean_text
from pages.services.text_chunker import chunk_text
from pages.services.embedding_service import EmbeddingService
from pages.services.vector_store import VectorStore
from pages.services.rag_service import build_context_from_results, generate_basic_rag_answer


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "will", "with",
    "this", "these", "those", "into", "their", "there", "about", "what", "which", "when",
    "where", "who", "why", "how", "your", "you", "they", "them", "than", "then", "also"
}


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40]


def _count_term_in_text(text: str, term: str) -> int:
    return len(re.findall(rf"\b{re.escape(term.lower())}\b", text.lower()))


def _top_terms_from_pages(pages: list[dict], limit: int = 6) -> list[str]:
    tokens = []
    for page in pages:
        tokens.extend(
            token for token in _tokenize_words(page.get("text", ""))
            if token not in STOP_WORDS and not token.isdigit()
        )

    ranked_terms = [term for term, _ in Counter(tokens).most_common(limit * 3)]
    unique_terms = []

    for term in ranked_terms:
        if term not in unique_terms:
            unique_terms.append(term)
        if len(unique_terms) >= limit:
            break

    return unique_terms


def build_suggested_questions(pages: list[dict]) -> list[str]:
    top_terms = _top_terms_from_pages(pages)
    suggested_questions = [
        "What is this document about?",
        "What are the main topics covered in the document?",
        "What key metrics or figures are mentioned?",
        "What actions, recommendations, or next steps are described?"
    ]

    for term in top_terms[:3]:
        suggested_questions.append(f"What does the document say about {term}?")

    deduped_questions = []
    for question in suggested_questions:
        if question not in deduped_questions:
            deduped_questions.append(question)

    return deduped_questions[:6]


def answer_question_from_pages(question: str, pages: list[dict], max_sentences: int = 3) -> str:
    question_terms = {
        token for token in _tokenize_words(question)
        if token not in STOP_WORDS
    }
    scored_sentences = []

    for page in pages:
        for sentence in _split_sentences(page.get("text", "")):
            sentence_terms = set(_tokenize_words(sentence))
            overlap = len(question_terms & sentence_terms)
            numeric_bonus = 1 if re.search(r"\d", sentence) else 0
            score = overlap * 3 + numeric_bonus

            if score > 0 or not question_terms:
                scored_sentences.append((score, page["page_number"], sentence))

    if not scored_sentences:
        fallback = []
        for page in pages[:2]:
            page_sentences = _split_sentences(page.get("text", ""))
            if page_sentences:
                fallback.append(f"Page {page['page_number']}: {page_sentences[0]}")
        return "\n\n".join(fallback) if fallback else "No answer could be derived from the extracted PDF text."

    top_matches = sorted(scored_sentences, key=lambda item: (-item[0], item[1]))[:max_sentences]
    return "\n\n".join(
        [f"Page {page_number}: {sentence}" for _, page_number, sentence in top_matches]
    )


def build_follow_up_questions(
    question: str,
    result_rows: list[dict],
    preferred_terms: list[str] | None = None
) -> list[str]:
    preferred_terms = preferred_terms or []
    follow_ups = [
        "Can you summarize this in 5 concise bullet points?",
        "What assumptions, risks, or constraints are mentioned?",
        "What actions or next steps are recommended?"
    ]

    if result_rows:
        first_row = result_rows[0]
        source_file = first_row.get("file_name") or "the document"
        source_page = first_row.get("page_number")
        if source_page is not None:
            follow_ups.append(f"What are the key insights from page {source_page} in {source_file}?")

    for term in preferred_terms[:3]:
        follow_ups.append(f"Can you explain the section related to {term} in simple terms?")

    deduped = []
    for candidate in follow_ups:
        if candidate not in deduped and candidate.strip().lower() != question.strip().lower():
            deduped.append(candidate)

    return deduped[:6]


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

if "suggested_pdf_question" not in st.session_state:
    st.session_state.suggested_pdf_question = ""

if "latest_cleaned_pages" not in st.session_state:
    st.session_state.latest_cleaned_pages = []

if "latest_top_terms" not in st.session_state:
    st.session_state.latest_top_terms = []

if "manual_query_input" not in st.session_state:
    st.session_state.manual_query_input = ""

st.markdown(
    """
    <style>
    .qa-card {
        border: 1px solid rgba(49, 51, 63, 0.20);
        border-radius: 14px;
        padding: 12px 14px;
        margin-top: 8px;
        background: linear-gradient(180deg, rgba(49, 51, 63, 0.06), rgba(49, 51, 63, 0.02));
    }
    .qa-role {
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.75;
        margin-bottom: 6px;
    }
    .qa-content {
        font-size: 0.95rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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

    st.subheader("4. Executive Summary")

    page_metrics_df = pd.DataFrame([
        {
            "page_number": page["page_number"],
            "characters": len(page["text"]),
            "estimated_words": len(page["text"].split()),
            "estimated_sentences": max(1, len(_split_sentences(page["text"])))
        }
        for page in cleaned_pages
    ])

    pages_with_text = int((page_metrics_df["characters"] > 0).sum())
    coverage_pct = (pages_with_text / len(page_metrics_df) * 100) if len(page_metrics_df) else 0
    avg_words_per_page = int(page_metrics_df["estimated_words"].mean()) if len(page_metrics_df) else 0

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Document Coverage", f"{coverage_pct:.1f}%")
    summary_col2.metric("Pages With Text", pages_with_text)
    summary_col3.metric("Avg Words/Page", f"{avg_words_per_page:,}")

    exec_col1, exec_col2 = st.columns(2)

    with exec_col1:
        coverage_chart_df = page_metrics_df[["page_number", "characters"]].copy()
        coverage_chart_df["coverage_flag"] = coverage_chart_df["characters"].apply(lambda value: 1 if value > 0 else 0)
        st.caption("Document coverage by page")
        st.area_chart(
            coverage_chart_df.set_index("page_number")[["coverage_flag"]],
            use_container_width=True
        )

    top_terms = _top_terms_from_pages(cleaned_pages)
    st.session_state.latest_cleaned_pages = cleaned_pages
    st.session_state.latest_top_terms = top_terms

    with exec_col2:
        topic_counter = Counter(
            token
            for page in cleaned_pages
            for token in _tokenize_words(page["text"])
            if token not in STOP_WORDS
        )
        topic_df = pd.DataFrame(topic_counter.most_common(8), columns=["topic", "count"])
        if not topic_df.empty:
            topic_df["share_pct"] = (topic_df["count"] / max(1, topic_df["count"].sum()) * 100).round(2)
            st.caption("Topic concentration (top terms share)")
            st.bar_chart(topic_df.set_index("topic")[["share_pct"]], use_container_width=True)

    if top_terms:
        trend_terms = top_terms[:3]
        trend_records = []
        for page in cleaned_pages:
            row = {"page_number": page["page_number"]}
            for term in trend_terms:
                row[term] = _count_term_in_text(page["text"], term)
            trend_records.append(row)

        trend_df = pd.DataFrame(trend_records)
        if not trend_df.empty:
            st.caption("Keyword trend lines across pages")
            st.line_chart(
                trend_df.set_index("page_number"),
                use_container_width=True
            )

    st.subheader("5. Document Analytics")

    analytics_col1, analytics_col2 = st.columns(2)

    with analytics_col1:
        st.caption("Text volume by page")
        st.bar_chart(
            page_metrics_df.set_index("page_number")[["characters", "estimated_words"]],
            use_container_width=True
        )

    with analytics_col2:
        st.caption("Sentence density by page")
        st.line_chart(
            page_metrics_df.set_index("page_number")[["estimated_sentences"]],
            use_container_width=True
        )

    if top_terms:
        keyword_df = pd.DataFrame(
            Counter(
                token
                for page in cleaned_pages
                for token in _tokenize_words(page["text"])
                if token in top_terms
            ).most_common(),
            columns=["term", "count"]
        )
        if not keyword_df.empty:
            st.caption("Top repeated document terms")
            st.bar_chart(keyword_df.set_index("term"), use_container_width=True)

    suggested_questions = build_suggested_questions(cleaned_pages)

    st.subheader("6. Suggested Questions from PDF")
    st.caption("Auto-generated prompts and quick evidence-based answers derived from extracted text.")

    question_cols = st.columns(2)
    for index, suggested_question in enumerate(suggested_questions):
        with question_cols[index % 2]:
            if st.button(suggested_question, key=f"suggested_question_{index}"):
                st.session_state.suggested_pdf_question = suggested_question

    selected_suggested_question = st.session_state.suggested_pdf_question or (
        suggested_questions[0] if suggested_questions else ""
    )

    if selected_suggested_question:
        suggested_answer = answer_question_from_pages(selected_suggested_question, cleaned_pages)
        suggested_answer_html = escape(suggested_answer).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class=\"qa-card\">
                <div class=\"qa-role\">Suggested Question</div>
                <div class=\"qa-content\">{escape(selected_suggested_question)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class=\"qa-card\">
                <div class=\"qa-role\">Answer (From PDF Evidence)</div>
                <div class=\"qa-content\">{suggested_answer_html}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("7. Chunk Document")

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

    st.subheader("8. Data Quality Checks")

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

    st.subheader("9. Generate Embeddings & Store in Vector DB")

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

st.subheader("10. Semantic Search / RAG Query")

query = st.text_input(
    "Ask a question from uploaded documents",
    placeholder="Example: What is this document about?",
    key="manual_query_input"
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

    follow_up_questions = build_follow_up_questions(
        question=query,
        result_rows=result_rows,
        preferred_terms=st.session_state.latest_top_terms
    )

    if follow_up_questions:
        st.subheader("Suggested Follow-up Questions")
        st.caption("Click any follow-up to auto-fill and rerun semantic search.")
        follow_up_cols = st.columns(2)
        for index, follow_up_question in enumerate(follow_up_questions):
            with follow_up_cols[index % 2]:
                if st.button(follow_up_question, key=f"follow_up_question_{index}"):
                    st.session_state.manual_query_input = follow_up_question
                    st.rerun()

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