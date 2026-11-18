"""Unit tests for compiler node and Two-Pass Citation Resolution."""

from unittest.mock import MagicMock

from agent.nodes.compile import compiler_node
from agent.state import ResearchState


def test_compiler_two_pass_citations_normalization():
    state = ResearchState(
        research_topic="AVX-512 Power Comparison",
        outline=["1. Architecture", "2. Power Draw"],
        evidence_chunks=[
            {
                "chunk_id": "c8a1f29b",
                "source_url": "https://anandtech.com/ryzen7000",
                "page_title": "AnandTech Ryzen 7000 Review",
                "raw_content": "Ryzen 7950X consumes 230W under full AVX-512 workload.",
            },
            {
                "chunk_id": "b3e9a112",
                "source_url": "https://tomshardware.com/13900k",
                "page_title": "Tom's Hardware 13900K Review",
                "raw_content": "Core i9-13900K reaches 310W at peak package power.",
            },
        ],
        sections={
            "1. Architecture": "Ryzen uses dual 256-bit FADD pipes [^c8a1f29b].",
            "2. Power Draw": "Under full load, 7950X reaches 230W [^c8a1f29b] while 13900K pulls 310W [^b3e9a112].",
        },
    )

    mock_store = MagicMock()
    result = compiler_node(state, vector_store=mock_store)

    final_report = result["final_report"]
    citation_map = result["citation_map"]

    # 1. Verify sequential [1], [2] normalization
    assert "Ryzen uses dual 256-bit FADD pipes [1]." in final_report
    # Repeated citation [^c8a1f29b] gets same marker [1], new citation gets [2]
    assert "7950X reaches 230W [1] while 13900K pulls 310W [2]." in final_report

    # 2. Verify References section
    assert "## References" in final_report
    assert "[1] **AnandTech Ryzen 7000 Review** - [https://anandtech.com/ryzen7000](https://anandtech.com/ryzen7000)" in final_report
    assert "[2] **Tom's Hardware 13900K Review** - [https://tomshardware.com/13900k](https://tomshardware.com/13900k)" in final_report

    # 3. Verify citation_map
    assert "1" in citation_map
    assert citation_map["1"]["title"] == "AnandTech Ryzen 7000 Review"
    assert citation_map["1"]["chunk_id"] == "c8a1f29b"
    assert "2" in citation_map

    # 4. Verify collection deletion was called
    mock_store.delete_session_collection.assert_called_once_with(state.session_id)


def test_compiler_orphan_citation_stripping():
    state = ResearchState(
        research_topic="Test Topic",
        outline=["1. Section"],
        evidence_chunks=[
            {
                "chunk_id": "c8a1f29b",
                "source_url": "https://example.com/valid",
                "page_title": "Valid Source",
                "raw_content": "Valid chunk content.",
            }
        ],
        sections={
            "1. Section": "Real claim [^c8a1f29b] and fabricated claim [^hallucinated_chunk_xyz].",
        },
    )

    result = compiler_node(state, vector_store=None)
    final_report = result["final_report"]

    # The real chunk is converted to [1]
    assert "[1]" in final_report
    # The hallucinated chunk is stripped
    assert "[^hallucinated_chunk_xyz]" not in final_report
    assert "fabricated claim ." in final_report or "fabricated claim" in final_report
    assert len(result["citation_map"]) == 1


def test_compiler_retain_collection():
    state = ResearchState(
        research_topic="Test Topic",
        outline=["1. Sec"],
        sections={"1. Sec": "Content"},
        retain_collection=True,
    )
    mock_store = MagicMock()
    compiler_node(state, vector_store=mock_store)
    # delete_session_collection must not be called when retain_collection is True
    mock_store.delete_session_collection.assert_not_called()
