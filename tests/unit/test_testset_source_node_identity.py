"""Document and chunk identities should survive independent graph rebuilds."""

from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from ragas.testset.graph import NodeType
from ragas.testset.synthesizers.generate import TestsetGenerator as Generator


def _generator():
    return Generator(llm=object(), embedding_model=object())


def _graph_for(generator, method, sources):
    # Exercise the public source-ingestion entry point without model calls.
    with (
        patch("ragas.testset.synthesizers.generate.apply_transforms"),
        patch.object(Generator, "generate", return_value="generated"),
    ):
        output = getattr(generator, method)(
            sources, testset_size=1, transforms=[object()]
        )
    assert output == "generated"
    return generator.knowledge_graph


def test_independent_langchain_graphs_have_identical_document_ids():
    documents = [
        Document(page_content="Alpha source", metadata={"tag": "x", "year": 2024}),
        Document(page_content="Beta source", metadata={"tag": "y"}),
    ]
    graph_a = _graph_for(_generator(), "generate_with_langchain_docs", documents)
    graph_b = _graph_for(
        _generator(), "generate_with_langchain_docs", list(reversed(documents))
    )
    ids_a = {node.properties["page_content"]: node.id for node in graph_a.nodes}
    ids_b = {node.properties["page_content"]: node.id for node in graph_b.nodes}
    assert ids_a == ids_b
    assert all(node.type == NodeType.DOCUMENT for node in graph_a.nodes)


def test_metadata_key_order_does_not_change_identity_but_value_does():
    generator = _generator()
    method = "generate_with_langchain_docs"
    a = Document(page_content="Same content", metadata={"x": 1, "y": [2, 3]})
    reordered = Document(page_content="Same content", metadata={"y": [2, 3], "x": 1})
    changed = Document(page_content="Same content", metadata={"x": 1, "y": [3, 2]})

    a_id = _graph_for(generator, method, [a]).nodes[0].id
    reordered_id = _graph_for(generator, method, [reordered]).nodes[0].id
    changed_id = _graph_for(generator, method, [changed]).nodes[0].id
    assert a_id == reordered_id
    assert a_id != changed_id


def test_repeated_identical_documents_remain_distinct_and_stable():
    sources = [
        Document(page_content="Duplicate"),
        Document(page_content="Duplicate"),
        Document(page_content="Other"),
    ]
    graph_a = _graph_for(_generator(), "generate_with_langchain_docs", sources)
    graph_b = _graph_for(_generator(), "generate_with_langchain_docs", sources)

    ids_a = [node.id for node in graph_a.nodes]
    ids_b = [node.id for node in graph_b.nodes]
    assert ids_a == ids_b
    assert len(set(ids_a)) == len(sources)
    assert [node.properties["page_content"] for node in graph_a.nodes] == [
        "Duplicate",
        "Duplicate",
        "Other",
    ]


def test_chunk_graphs_rebuild_with_stable_distinct_ids():
    sources = [
        "Shared chunk text",
        Document(page_content="Shared chunk text"),
        Document(page_content="Second chunk", metadata={"origin": "a"}),
    ]
    method = "generate_with_chunks"
    graph_a = _graph_for(_generator(), method, sources)
    graph_b = _graph_for(_generator(), method, sources)

    assert [node.id for node in graph_a.nodes] == [node.id for node in graph_b.nodes]
    assert len({node.id for node in graph_a.nodes}) == 3
    assert all(node.type == NodeType.CHUNK for node in graph_a.nodes)


def test_llamaindex_and_langchain_document_ingestion_share_identity():
    metadata = {"source": "https://example.org", "n": 1}
    langchain = Document(page_content="Page content", metadata=metadata)
    llamaindex = SimpleNamespace(text="Page content", metadata=dict(metadata))
    graph_a = _graph_for(_generator(), "generate_with_langchain_docs", [langchain])
    graph_b = _graph_for(_generator(), "generate_with_llamaindex_docs", [llamaindex])

    assert graph_a.nodes[0].id == graph_b.nodes[0].id
    assert graph_a.nodes[0].properties == graph_b.nodes[0].properties


def test_blank_llamaindex_docs_and_chunks_keep_existing_skip_behavior():
    graph = _graph_for(
        _generator(),
        "generate_with_llamaindex_docs",
        [
            SimpleNamespace(text="  ", metadata={}),
            SimpleNamespace(text="Valid", metadata={}),
        ],
    )
    assert [node.get_property("page_content") for node in graph.nodes] == ["Valid"]

    chunks = _graph_for(_generator(), "generate_with_chunks", ["  ", "Valid"])
    assert [node.get_property("page_content") for node in chunks.nodes] == ["Valid"]


def test_nested_mixed_key_metadata_does_not_change_source_identity():
    metadata = {"nested": {1: "a", "1": "b"}, "tags": {"orange", "apple"}}
    generator = _generator()
    method = "generate_with_langchain_docs"
    first = Document(page_content="Metadata", metadata=metadata)
    reordered = Document(
        page_content="Metadata",
        metadata={"tags": {"apple", "orange"}, "nested": {"1": "b", 1: "a"}},
    )
    first_id = _graph_for(generator, method, [first]).nodes[0].id
    second_id = _graph_for(generator, method, [reordered]).nodes[0].id
    assert first_id == second_id


def test_rebuilding_source_and_splitting_preserves_entire_graph_identity():
    from ragas.testset.transforms.engine import apply_transforms
    from ragas.testset.transforms.splitters import HeadlineSplitter

    text = (
        "First headline\n"
        + "First section content. " * 40
        + "\nSecond headline\n"
        + "Second section content. " * 40
    )
    document = Document(page_content=text, metadata={"chapter": 1})
    graphs = [
        _graph_for(_generator(), "generate_with_langchain_docs", [document])
        for _ in range(2)
    ]
    for graph in graphs:
        graph.nodes[0].add_property("headlines", ["First headline", "Second headline"])
        apply_transforms(graph, HeadlineSplitter(min_tokens=1, max_tokens=3000))

    assert [node.id for node in graphs[0].nodes] == [
        node.id for node in graphs[1].nodes
    ]
    assert [edge.id for edge in graphs[0].relationships] == [
        edge.id for edge in graphs[1].relationships
    ]
