"""Unit tests for agent LLM client and structured output parser."""

import pytest

from agent.llm import MockAgentLLM, StructuredOutputParser


def test_structured_output_parser_direct_json():
    raw = '{"outline": ["A", "B"], "search_queries": ["Q1"]}'
    res = StructuredOutputParser.extract_json(raw)
    assert res["outline"] == ["A", "B"]
    assert res["search_queries"] == ["Q1"]


def test_structured_output_parser_markdown_fence():
    raw = """Here is the research plan:
```json
{
  "outline": ["Sec 1", "Sec 2"],
  "search_queries": ["Q1", "Q2"]
}
```
Hope this helps!"""
    res = StructuredOutputParser.extract_json(raw)
    assert res["outline"] == ["Sec 1", "Sec 2"]
    assert res["search_queries"] == ["Q1", "Q2"]


def test_structured_output_parser_trailing_comma():
    raw = """{
      "outline": ["Sec 1", "Sec 2",],
      "search_queries": ["Q1",],
    }"""
    res = StructuredOutputParser.extract_json(raw)
    assert res["outline"] == ["Sec 1", "Sec 2"]


def test_structured_output_parser_invalid():
    with pytest.raises(ValueError):
        StructuredOutputParser.extract_json("Not a json at all")


@pytest.mark.asyncio
async def test_mock_agent_llm():
    mock_llm = MockAgentLLM(default_response='{"status": "ok"}')
    messages = [{"role": "user", "content": "Hello"}]
    res = await mock_llm.agenerate(messages)
    assert res == '{"status": "ok"}'
    assert len(mock_llm.call_history) == 1
    assert mock_llm.call_history[0] == messages
