def build_context_from_results(results: dict) -> str:
    """
    Build context from vector search results.
    """

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    context_parts = []

    for doc, meta in zip(documents, metadatas):
        context_parts.append(
            f"Source: {meta.get('file_name')} | Page: {meta.get('page_number')}\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_basic_rag_answer(question: str, context: str) -> str:
    """
    Basic local RAG-style answer.
    This does not call an LLM.
    It shows how retrieved context would be passed to an LLM.
    """

    if not context:
        return "No relevant context found."

    return f"""
Based on the retrieved document context, the answer should be generated from the following evidence.

Question:
{question}

Retrieved Context:
{context}

In a production system, this context would be passed to an LLM such as Azure OpenAI, OpenAI, Claude, Gemini, or a local model.
"""