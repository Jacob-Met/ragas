"""Pre-chunked knowledge graphs should score their own chunk summaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms.filters import CustomNodeFilter, QuestionPotentialOutput


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("node_type", "parent_summary", "child_summary", "expected_summary"),
    [
        (NodeType.CHUNK, None, "Pre-chunked summary", "Pre-chunked summary"),
        (NodeType.CHUNK, "Parent summary", "Chunk summary", "Parent summary"),
        (NodeType.CHUNK, "", "Chunk summary", "Chunk summary"),
        (NodeType.DOCUMENT, None, "Own document summary", "Own document summary"),
    ],
)
async def test_filter_uses_available_summary(
    node_type, parent_summary, child_summary, expected_summary
):
    node = Node(
        type=node_type,
        properties={"summary": child_summary, "page_content": "Real chunk content"},
    )
    graph = KnowledgeGraph(nodes=[node])
    if parent_summary is not None:
        parent = Node(type=NodeType.DOCUMENT, properties={"summary": parent_summary})
        graph.nodes.append(parent)
        graph.relationships.append(
            Relationship(type="child", source=parent, target=node)
        )

    scorer = SimpleNamespace(
        generate=AsyncMock(return_value=QuestionPotentialOutput(score=3))
    )
    metric = CustomNodeFilter(llm=object(), scoring_prompt=scorer)

    result = await metric.custom_filter(node, graph)
    assert result is False
    scorer.generate.assert_awaited_once()
    prompt_input = scorer.generate.await_args.kwargs["data"]
    assert prompt_input.document_summary == expected_summary
    assert prompt_input.node_content == "Real chunk content"


@pytest.mark.asyncio
async def test_filter_only_skips_node_when_no_summary_exists(caplog):
    node = Node(type=NodeType.CHUNK, properties={"page_content": "Chunk content"})
    graph = KnowledgeGraph(nodes=[node])
    scorer = SimpleNamespace(
        generate=AsyncMock(return_value=QuestionPotentialOutput(score=1))
    )
    metric = CustomNodeFilter(llm=object(), scoring_prompt=scorer)

    result = await metric.custom_filter(node, graph)

    assert result is False
    scorer.generate.assert_not_awaited()
    assert "does not have a summary" in caplog.text


@pytest.mark.asyncio
async def test_filter_can_remove_unhelpful_prechunked_node():
    node = Node(
        type=NodeType.CHUNK,
        properties={"summary": "Pre-chunked summary", "page_content": "Off-topic"},
    )
    graph = KnowledgeGraph(nodes=[node])
    scorer = SimpleNamespace(
        generate=AsyncMock(return_value=QuestionPotentialOutput(score=1))
    )
    metric = CustomNodeFilter(llm=object(), scoring_prompt=scorer)

    assert await metric.custom_filter(node, graph) is True
    scorer.generate.assert_awaited_once()
