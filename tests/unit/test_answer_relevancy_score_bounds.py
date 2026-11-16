"""Answer relevancy scores must stay within the documented 0–1 range."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragas.metrics._answer_relevance import (
    ResponseRelevanceOutput,
    ResponseRelevancy,
)
from ragas.metrics.collections.answer_relevancy.metric import (
    AnswerRelevancy as CollectionsAnswerRelevancy,
)
from ragas.metrics.collections.answer_relevancy.util import (
    AnswerRelevanceOutput,
    AnswerRelevancePrompt,
)


@pytest.mark.parametrize(
    ("generated_embeddings", "expected"),
    [
        ([[1.0, 0.0]], 1.0),
        ([[-1.0, 0.0]], 0.0),
        ([[0.0, 1.0]], 0.0),
        ([[0.0, 0.0]], 0.0),
        ([[1.0, 0.0], [-1.0, 0.0]], 0.0),
        ([[1.0, 0.0], [0.0, 1.0]], 0.5),
    ],
)
def test_legacy_relevancy_stays_bounded(generated_embeddings, expected):
    metric = ResponseRelevancy()
    metric.embeddings = SimpleNamespace(
        embed_query=lambda question: [1.0, 0.0],
        embed_documents=lambda questions: generated_embeddings,
    )
    answers = [
        ResponseRelevanceOutput(question="Generated question", noncommittal=0)
        for _ in generated_embeddings
    ]

    assert (
        metric._calculate_score(answers, {"user_input": "Original question"})
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("generated_embeddings", "expected"),
    [
        ([[1.0, 0.0]], 1.0),
        ([[-1.0, 0.0]], 0.0),
        ([[0.0, 1.0]], 0.0),
        ([[0.0, 0.0]], 0.0),
        ([[1.0, 0.0], [-1.0, 0.0]], 0.0),
        ([[1.0, 0.0], [0.0, 1.0]], 0.5),
    ],
)
async def test_collections_relevancy_stays_bounded(generated_embeddings, expected):
    metric = object.__new__(CollectionsAnswerRelevancy)
    metric.llm = SimpleNamespace(
        agenerate=AsyncMock(
            return_value=AnswerRelevanceOutput(
                question="Generated question", noncommittal=0
            )
        )
    )
    metric.embeddings = SimpleNamespace(
        aembed_text=AsyncMock(return_value=[1.0, 0.0]),
        aembed_texts=AsyncMock(return_value=generated_embeddings),
    )
    metric.strictness = len(generated_embeddings)
    metric.prompt = AnswerRelevancePrompt()

    result = await metric.ascore(user_input="Original question", response="Some answer")
    assert result.value == expected
