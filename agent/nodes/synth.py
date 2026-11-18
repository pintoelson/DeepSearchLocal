"""Synthesis loop node: Section-by-section hybrid retrieval, CPU reranking, and grounded drafting."""

from __future__ import annotations

import logging
from typing import Any

from agent.llm import BaseAgentLLM
from agent.prompts.synth_prompts import (
    SYNTH_REVISION_SYSTEM_PROMPT,
    SYNTH_SYSTEM_PROMPT,
    build_synth_revision_user_prompt,
    build_synth_user_prompt,
)
from agent.state import ResearchState
from rag.hybrid_retriever import HybridRetriever
from rag.models import ScoredChunk
from rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


async def synthesis_loop(
    state: ResearchState,
    llm: BaseAgentLLM,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    top_candidates: int = 15,
    top_reranked: int = 4,
) -> dict[str, Any]:
    """Execute section-by-section grounded drafting loop.

    Hardware & Context Preservation:
    - Restricts prompt contents to only immediate section outline and top-4 reranked chunks.
    - Operates sequentially to strictly preserve single-context 6 GB VRAM budget on consumer GPU.
    - Employs temporary chunk identifiers ([^chunk_id]) to prevent arithmetic hallucination.
    - Emits newly surfaced evidence chunks into state.evidence_chunks channel reducer.
    """
    logger.info(
        "Executing synthesis_loop (iteration=%d). Total outline sections=%d",
        state.iteration_count,
        len(state.outline),
    )

    # Determine sections to synthesize
    if state.iteration_count > 0 and state.sections_to_revise:
        # Targeted revision mode
        sections_to_process = [
            s for s in state.outline
            if any(rev.lower() in s.lower() or s.lower() in rev.lower() for rev in state.sections_to_revise)
        ]
        if not sections_to_process:
            sections_to_process = state.sections_to_revise
        logger.info("Targeted revision mode: revising %d sections", len(sections_to_process))
    else:
        # Full initial synthesis
        sections_to_process = list(state.outline)
        if not sections_to_process:
            sections_to_process = [f"Overview of {state.research_topic}"]

    updated_sections: dict[str, str] = dict(state.sections)
    all_used_chunks: list[dict[str, Any]] = []

    for idx, section_title in enumerate(sections_to_process, 1):
        logger.info(
            "Synthesizing section [%d/%d]: %s",
            idx,
            len(sections_to_process),
            section_title,
        )

        query = f"{state.research_topic} {section_title}"

        # 1. Hybrid dense + sparse retrieval (Server-side RRF on CPU)
        try:
            candidates: list[ScoredChunk] = retriever.retrieve(
                session_id=state.session_id,
                query=query,
                top_k=top_candidates,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Retrieval failed for query '%s': %s", query, e)
            candidates = []

        # 2. CPU Cross-Encoder Re-Ranking (top-15 -> top-4)
        if candidates:
            try:
                reranked_chunks: list[ScoredChunk] = reranker.rerank(
                    query=query,
                    candidates=candidates,
                    top_n=top_reranked,
                )
                evidence_chunk_dicts = [sc.chunk.model_dump() for sc in reranked_chunks]
            except Exception as e:  # noqa: BLE001
                logger.warning("Reranking failed for '%s': %s. Using top un-reranked chunks.", query, e)
                evidence_chunk_dicts = [sc.chunk.model_dump() for sc in candidates[:top_reranked]]
        elif state.evidence_chunks:
            # Fallback to in-state evidence chunks if direct retrieval returned no results
            evidence_chunk_dicts = state.evidence_chunks[:top_reranked]
        else:
            evidence_chunk_dicts = []

        all_used_chunks.extend(evidence_chunk_dicts)

        # 3. Formulate grounded generation prompt
        is_revision = section_title in state.sections
        if is_revision:
            messages = [
                {"role": "system", "content": SYNTH_REVISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_synth_revision_user_prompt(
                        section_title=section_title,
                        research_topic=state.research_topic,
                        current_draft=state.sections[section_title],
                        unresolved_gaps=state.unresolved_gaps,
                        new_evidence_chunks=evidence_chunk_dicts,
                    ),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_synth_user_prompt(
                        section_title=section_title,
                        research_topic=state.research_topic,
                        outline=state.outline,
                        evidence_chunks=evidence_chunk_dicts,
                    ),
                },
            ]

        # 4. Invoke SLM
        try:
            draft_text = await llm.agenerate(messages, temperature=0.2, max_tokens=1500)
        except Exception as e:  # noqa: BLE001
            logger.error("Synthesis LLM call failed for section '%s': %s", section_title, e)
            draft_text = f"*(Synthesis failed for this section due to generation error: {e})*"

        updated_sections[section_title] = draft_text.strip()

    # Deduplicate newly used chunks against existing state.evidence_chunks
    existing_cids = {str(c.get("chunk_id", "")) for c in state.evidence_chunks}
    new_chunks_to_record = []
    for chunk_dict in all_used_chunks:
        cid = str(chunk_dict.get("chunk_id", ""))
        if cid and cid not in existing_cids:
            new_chunks_to_record.append(chunk_dict)
            existing_cids.add(cid)

    return {
        "sections": updated_sections,
        "sections_to_revise": [],
        "evidence_chunks": new_chunks_to_record,
    }
