"""Repeated relationship transforms must update a graph, not duplicate edges."""

import asyncio
import uuid
from dataclasses import dataclass

import pytest

from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms.base import RelationshipBuilder
from ragas.testset.transforms.engine import Parallel, apply_transforms
from ragas.testset.transforms.relationship_builders.cosine import (
    CosineSimilarityBuilder,
    SummaryCosineSimilarityBuilder,
)
from ragas.testset.transforms.relationship_builders.traditional import (
    JaccardSimilarityBuilder,
)


def _cosine_nodes():
    return [
        Node(
            id=uuid.UUID(int=1),
            type=NodeType.CHUNK,
            properties={"embedding": [1.0, 0.0]},
        ),
        Node(
            id=uuid.UUID(int=2),
            type=NodeType.CHUNK,
            properties={"embedding": [1.0, 0.0]},
        ),
    ]


def test_repeating_cosine_transform_does_not_duplicate_links():
    graph = KnowledgeGraph(nodes=_cosine_nodes())
    builder = CosineSimilarityBuilder(threshold=0.9)

    apply_transforms(graph, builder)
    first = graph.relationships[0]
    assert len(graph.relationships) == 1
    apply_transforms(graph, builder)

    assert len(graph.relationships) == 1
    assert graph.relationships[0] is first


def test_link_ids_stable_when_independent_source_order_changes():
    nodes = _cosine_nodes()
    graph_a = KnowledgeGraph(nodes=nodes)
    graph_b = KnowledgeGraph(nodes=list(reversed(nodes)))
    builder = CosineSimilarityBuilder(threshold=0.9)
    apply_transforms(graph_a, builder)
    apply_transforms(graph_b, builder)

    assert len(graph_a.relationships) == len(graph_b.relationships) == 1
    assert graph_a.relationships[0].id == graph_b.relationships[0].id


def test_repeating_jaccard_transform_keeps_one_link():
    graph = KnowledgeGraph(
        nodes=[
            Node(type=NodeType.CHUNK, properties={"entities": ["a", "b"]}),
            Node(type=NodeType.CHUNK, properties={"entities": ["a", "b"]}),
        ]
    )
    builder = JaccardSimilarityBuilder(threshold=0.5)
    apply_transforms(graph, builder)
    apply_transforms(graph, builder)

    assert len(graph.relationships) == 1


def test_summary_cosine_ignores_unrelated_chunks():
    graph = KnowledgeGraph(
        nodes=[
            Node(
                type=NodeType.DOCUMENT,
                properties={"summary_embedding": [1.0, 0.0]},
            ),
            Node(type=NodeType.CHUNK, properties={"page_content": "No summary"}),
            Node(
                type=NodeType.DOCUMENT,
                properties={"summary_embedding": [1.0, 0.0]},
            ),
        ]
    )
    apply_transforms(graph, SummaryCosineSimilarityBuilder(threshold=0.9))

    assert len(graph.relationships) == 1
    assert graph.relationships[0].type == "summary_cosine_similarity"


@dataclass
class UpdatingLinkBuilder(RelationshipBuilder):
    score: float = 0.5

    async def transform(self, graph: KnowledgeGraph):
        return [
            Relationship(
                type="custom-score",
                source=graph.nodes[0],
                target=graph.nodes[1],
                properties={"score": self.score},
            )
        ]


def test_reapplying_builder_updates_score_preserving_annotations_and_identity():
    graph = KnowledgeGraph(nodes=_cosine_nodes())
    builder = UpdatingLinkBuilder(score=0.5)
    apply_transforms(graph, builder)
    rel = graph.relationships[0]
    rel.properties["human_note"] = "Keep this"
    original_id = rel.id

    builder.score = 0.8
    apply_transforms(graph, builder)

    assert len(graph.relationships) == 1
    assert graph.relationships[0] is rel
    assert rel.id == original_id
    assert rel.properties == {"score": 0.8, "human_note": "Keep this"}


def test_directed_relationship_pairs_do_not_collapse():
    @dataclass
    class BothDirections(RelationshipBuilder):
        async def transform(self, graph: KnowledgeGraph):
            a, b = graph.nodes
            return [
                Relationship(type="links", source=a, target=b),
                Relationship(type="links", source=b, target=a),
            ]

    graph = KnowledgeGraph(nodes=_cosine_nodes())
    builder = BothDirections()
    apply_transforms(graph, builder)
    apply_transforms(graph, builder)

    assert len(graph.relationships) == 2
    assert graph.relationships[0].id != graph.relationships[1].id


def test_changed_cosine_similarity_updates_existing_edge_score():
    graph = KnowledgeGraph(nodes=_cosine_nodes())
    graph.nodes[1].properties["embedding"] = [0.8, 0.6]
    builder = CosineSimilarityBuilder(threshold=0.5)
    apply_transforms(graph, builder)

    rel = graph.relationships[0]
    assert rel.properties["cosine_similarity"] == pytest.approx(0.8)
    rel.properties["editor_note"] = "preserve annotation"

    graph.nodes[1].properties["embedding"] = [1.0, 0.0]
    apply_transforms(graph, builder)

    assert len(graph.relationships) == 1
    assert graph.relationships[0] is rel
    assert rel.properties["cosine_similarity"] == pytest.approx(1.0)
    assert rel.properties["editor_note"] == "preserve annotation"


def test_parallel_builders_deduplicate_racing_equivalent_edges():
    @dataclass
    class DelayedBuilder(RelationshipBuilder):
        label: str = "first"

        async def transform(self, graph: KnowledgeGraph):
            await asyncio.sleep(0.01)
            return [
                Relationship(
                    type="same",
                    source=graph.nodes[0],
                    target=graph.nodes[1],
                    properties={self.label: True},
                    bidirectional=True,
                )
            ]

    graph = KnowledgeGraph(nodes=_cosine_nodes())
    apply_transforms(
        graph,
        Parallel(
            DelayedBuilder(label="first"),
            DelayedBuilder(label="second"),
        ),
    )

    assert len(graph.relationships) == 1
    assert graph.relationships[0].properties == {"first": True, "second": True}


def test_loaded_graph_can_rebuild_links_without_duplication(tmp_path):
    graph = KnowledgeGraph(nodes=_cosine_nodes())
    builder = CosineSimilarityBuilder(threshold=0.9)
    apply_transforms(graph, builder)
    original_id = graph.relationships[0].id

    path = tmp_path / "graph.json"
    graph.save(path)
    restored = KnowledgeGraph.load(path)
    apply_transforms(restored, builder)

    assert len(restored.relationships) == 1
    assert restored.relationships[0].id == original_id
