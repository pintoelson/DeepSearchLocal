"""LangGraph cyclic state graph topology and routing for DeepSearch-Local."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.llm import AgentLLM, BaseAgentLLM
from agent.nodes.compile import compiler_node
from agent.nodes.ingest import ingestion_node
from agent.nodes.planner import planner_node
from agent.nodes.reflect import reflection_node
from agent.nodes.search import search_scrape_node
from agent.nodes.synth import synthesis_loop
from agent.state import ResearchState
from rag.chunking import RecursiveMarkdownChunker
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker
from rag.vector_store import EphemeralVectorStore

logger = logging.getLogger(__name__)


def build_research_graph(
    llm: BaseAgentLLM | None = None,
    vector_store: EphemeralVectorStore | None = None,
    retriever: HybridRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
    chunker: RecursiveMarkdownChunker | None = None,
) -> Any:
    """Build and compile the LangGraph cyclic state machine.

    Topology:
        1. planner -> search_scrape
        2. search_scrape -> ingestion
        3. ingestion -> synthesis
        4. synthesis -> reflection
        5. reflection -> search_scrape (conditional: if gaps exist and iteration_count < max_iterations)
                      -> compiler (conditional: if audit passes or max_iterations reached)
        6. compiler -> END
    """
    llm_instance = llm or AgentLLM()
    store_instance = vector_store or EphemeralVectorStore()
    retriever_instance = retriever or HybridRetriever(vector_store=store_instance)
    reranker_instance = reranker or CrossEncoderReranker()
    chunker_instance = chunker or RecursiveMarkdownChunker(max_tokens=512, overlap_tokens=64)

    workflow = StateGraph(ResearchState)

    # 1. Define Node Executors with Dependency Injection
    async def _planner_step(state: ResearchState) -> dict[str, Any]:
        return await planner_node(state, llm=llm_instance)

    async def _search_step(state: ResearchState) -> dict[str, Any]:
        return await search_scrape_node(state)

    def _ingest_step(state: ResearchState) -> dict[str, Any]:
        return ingestion_node(
            state,
            vector_store=store_instance,
            chunker=chunker_instance,
        )

    async def _synth_step(state: ResearchState) -> dict[str, Any]:
        return await synthesis_loop(
            state,
            llm=llm_instance,
            retriever=retriever_instance,
            reranker=reranker_instance,
        )

    async def _reflect_step(state: ResearchState) -> dict[str, Any]:
        return await reflection_node(state, llm=llm_instance)

    def _compile_step(state: ResearchState) -> dict[str, Any]:
        return compiler_node(state, vector_store=store_instance)

    # 2. Add Nodes to Graph
    workflow.add_node("planner", _planner_step)
    workflow.add_node("search_scrape", _search_step)
    workflow.add_node("ingestion", _ingest_step)
    workflow.add_node("synthesis", _synth_step)
    workflow.add_node("reflection", _reflect_step)
    workflow.add_node("compiler", _compile_step)

    # 3. Wire Linear Directed Edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "search_scrape")
    workflow.add_edge("search_scrape", "ingestion")
    workflow.add_edge("ingestion", "synthesis")
    workflow.add_edge("synthesis", "reflection")

    # 4. Wire Conditional Self-Correction Routing Edge
    def should_continue_research(state: ResearchState) -> str:
        """Route conditionally based on reflection audit and iteration guardrail."""
        has_revisions = len(state.sections_to_revise) > 0
        within_limits = state.iteration_count < state.max_iterations

        if has_revisions and within_limits:
            logger.info(
                "Conditional Edge: Routing to search_scrape for revision iteration %d/%d (sections=%s)",
                state.iteration_count,
                state.max_iterations,
                state.sections_to_revise,
            )
            return "search_scrape"

        logger.info(
            "Conditional Edge: Routing to compiler (iteration=%d, max=%d, sections_to_revise=%d)",
            state.iteration_count,
            state.max_iterations,
            len(state.sections_to_revise),
        )
        return "compiler"

    workflow.add_conditional_edges(
        "reflection",
        should_continue_research,
        {
            "search_scrape": "search_scrape",
            "compiler": "compiler",
        },
    )

    workflow.add_edge("compiler", END)

    logger.info("Compiled DeepSearch-Local LangGraph cyclic state machine.")
    return workflow.compile()
