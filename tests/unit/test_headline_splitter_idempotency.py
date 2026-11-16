import pytest

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms.engine import apply_transforms
from ragas.testset.transforms.splitters import HeadlineSplitter


def _document_with_headlines():
    text = (
        "First headline\n"
        + "First section content. " * 40
        + "\nSecond headline\n"
        + "Second section content. " * 40
    )
    return Node(
        type=NodeType.DOCUMENT,
        properties={
            "page_content": text,
            "headlines": ["First headline", "Second headline"],
        },
    )


@pytest.mark.asyncio
async def test_repeating_headline_split_keeps_same_chunk_and_edge_ids():
    document = _document_with_headlines()
    splitter = HeadlineSplitter(min_tokens=1, max_tokens=3000)

    first_nodes, first_edges = await splitter.split(document)
    second_nodes, second_edges = await splitter.split(document)

    assert len(first_nodes) >= 2
    assert [node.id for node in first_nodes] == [node.id for node in second_nodes]
    assert [edge.id for edge in first_edges] == [edge.id for edge in second_edges]


def test_repeating_headline_transform_does_not_duplicate_graph():
    document = _document_with_headlines()
    graph = KnowledgeGraph(nodes=[document])
    splitter = HeadlineSplitter(min_tokens=1, max_tokens=3000)

    apply_transforms(graph, splitter)
    first_counts = (len(graph.nodes), len(graph.relationships))
    first_ids = (
        [node.id for node in graph.nodes],
        [edge.id for edge in graph.relationships],
    )

    apply_transforms(graph, splitter)

    assert (len(graph.nodes), len(graph.relationships)) == first_counts
    assert [node.id for node in graph.nodes] == first_ids[0]
    assert [edge.id for edge in graph.relationships] == first_ids[1]


def test_unsplit_source_document_is_not_added_twice():
    source = _document_with_headlines()
    graph = KnowledgeGraph(nodes=[source])
    splitter = HeadlineSplitter(min_tokens=10000)

    apply_transforms(graph, splitter)
    apply_transforms(graph, splitter)

    assert graph.nodes == [source]
    assert not graph.relationships


@pytest.mark.asyncio
async def test_identical_text_from_distinct_sources_keeps_distinct_chunks():
    first_source = _document_with_headlines()
    second_source = _document_with_headlines()
    splitter = HeadlineSplitter(min_tokens=1, max_tokens=3000)

    first_nodes, first_edges = await splitter.split(first_source)
    second_nodes, second_edges = await splitter.split(second_source)

    assert len(first_nodes) == len(second_nodes)
    assert {node.id for node in first_nodes}.isdisjoint(
        {node.id for node in second_nodes}
    )
    assert {edge.id for edge in first_edges}.isdisjoint(
        {edge.id for edge in second_edges}
    )


def test_existing_chunk_enrichment_is_preserved_across_repeated_split():
    source = _document_with_headlines()
    graph = KnowledgeGraph(nodes=[source])
    splitter = HeadlineSplitter(min_tokens=1, max_tokens=3000)

    apply_transforms(graph, splitter)
    child = next(node for node in graph.nodes if node.type == NodeType.CHUNK)
    child.add_property("summary", "enriched by downstream extractor")
    apply_transforms(graph, splitter)

    retained = graph.get_node_by_id(child.id)
    assert retained is child
    assert retained.get_property("summary") == "enriched by downstream extractor"
