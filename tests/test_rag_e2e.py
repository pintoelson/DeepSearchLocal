"""End-to-end integration test for the full Section 3.3 Dynamic Hybrid RAG pipeline."""

import uuid

from rag import (
    CrossEncoderReranker,
    EphemeralVectorStore,
    HybridRetriever,
    RecursiveMarkdownChunker,
)


def test_full_rag_pipeline_lifecycle():
    """Verify end-to-end RAG lifecycle: chunking -> ephemeral indexing -> hybrid retrieval -> reranking -> cleanup."""
    session_id = f"e2e_{uuid.uuid4().hex[:8]}"

    # 1. Simulated scraped documents
    docs = [
        (
            """# NVIDIA GeForce RTX 3050 Laptop
The RTX 3050 Mobile is an entry-level laptop GPU based on the Ampere GA107 chip.
It comes with 4 GB or 6 GB GDDR6 VRAM with a 96-bit memory bus width.

## Memory Bandwidth and Capacity
The 6GB variant offers 168 GB/s memory bandwidth.
This capacity is critical when loading small language models like Qwen 2.5 3B with 4-bit quantization.
""",
            "https://notebookcheck.net/rtx-3050",
            "RTX 3050 Mobile Review",
        ),
        (
            """# QLoRA Parameter-Efficient Fine-Tuning
QLoRA backpropagates gradients through a frozen 4-bit quantized base model into Low-Rank Adapters (LoRA).
It uses NF4 (NormalFloat4) data type and Double Quantization to minimize memory footprint.

## VRAM Optimization on 6GB GPUs
By applying paged optimizers and gradient checkpointing, a 3B model can be fine-tuned in ~5.2 GB VRAM.
""",
            "https://huggingface.co/blog/qlora",
            "QLoRA Overview",
        ),
        (
            """# Artisan Sourdough Fermentation
The starter culture is a symbiotic culture of lactic acid bacteria and yeasts.
Ambient temperature during bulk fermentation dictates acetic vs lactic acid ratios.
""",
            "https://kingarthurbaking.com/sourdough",
            "Artisan Sourdough Guide",
        ),
    ]

    # 2. Chunking
    chunker = RecursiveMarkdownChunker(max_tokens=256, overlap_tokens=32)
    all_chunks = []
    for content, url, title in docs:
        all_chunks.extend(chunker.chunk_document(content, url, page_title=title))

    assert len(all_chunks) >= 4

    # 3. Ephemeral Vector Store
    store = EphemeralVectorStore(use_memory=True)
    with store.session(session_id):
        # 4. Ingestion & Embedding
        upserted_count = store.add_documents(session_id, all_chunks)
        assert upserted_count == len(all_chunks)
        assert store.count_documents(session_id) == len(all_chunks)

        # 5. Hybrid Retrieval (Dense + BM25 + Server-side RRF)
        retriever = HybridRetriever(vector_store=store, rrf_k=60)
        query = "Can you fine-tune a 3B parameter model on a 6GB RTX 3050 using QLoRA?"
        candidates = retriever.retrieve(session_id=session_id, query=query, top_k=5)

        assert len(candidates) > 0
        assert any("QLoRA" in c.chunk.raw_content for c in candidates)
        assert any("RTX 3050" in c.chunk.raw_content for c in candidates)

        # 6. Cross-Encoder Re-Ranking (Top-15 down to Top-4)
        reranker = CrossEncoderReranker()
        top_passages = reranker.rerank(query=query, candidates=candidates, top_n=2)

        assert len(top_passages) == 2
        for passage in top_passages:
            assert passage.reranker_score is not None
            assert 0.0 <= passage.reranker_score <= 1.0
            assert passage.chunk.chunk_id != ""
            assert passage.chunk.source_url != ""

        # Verify sourdough is completely absent from top passages
        for passage in top_passages:
            assert "Sourdough" not in passage.chunk.raw_content

    # 7. Verification of Ephemeral Cleanup
    assert store._resolve_collection_name(session_id) is None
