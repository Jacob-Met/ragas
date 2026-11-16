"""Answer correctness should report the available classifier reasoning."""

from unittest.mock import AsyncMock

import pytest

from ragas.metrics.collections.answer_correctness.metric import AnswerCorrectness
from ragas.metrics.collections.answer_correctness.util import (
    ClassificationWithReason,
    StatementsWithReason,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("weights", [[1.0, 0.0], [0.75, 0.25]])
async def test_answer_correctness_preserves_classifier_reasons(weights):
    classification = ClassificationWithReason(
        TP=[
            StatementsWithReason(
                statement="Paris is in France", reason="Matches reference"
            )
        ],
        FP=[
            StatementsWithReason(
                statement="Paris is in Germany", reason="Incorrect country"
            )
        ],
        FN=[
            StatementsWithReason(
                statement="Paris is a capital", reason="Omitted from answer"
            )
        ],
    )
    metric = object.__new__(AnswerCorrectness)
    metric.weights = weights
    metric.beta = 1.0
    metric._generate_statements = AsyncMock(
        side_effect=[
            ["Paris is in France", "Paris is in Germany"],
            ["Paris is in France", "Paris is a capital"],
        ]
    )
    metric._classify_statements = AsyncMock(return_value=classification)
    metric._calculate_similarity = AsyncMock(return_value=0.2)

    result = await metric.ascore(
        user_input="What is Paris?",
        response="Paris is in France and in Germany",
        reference="Paris is the capital of France",
    )

    expected = 0.5 if weights[1] == 0 else 0.425
    assert result.value == pytest.approx(expected)
    assert "Incorrect country" in result.reason
    assert result.traces["output"]["classification"] == classification.model_dump()
    assert result.traces["output"]["factuality"] == 0.5
    if weights[1]:
        assert "similarity" in result.reason.lower()
    else:
        metric._calculate_similarity.assert_not_called()


@pytest.mark.asyncio
async def test_answer_correctness_explains_missing_extracted_statements():
    metric = object.__new__(AnswerCorrectness)
    metric.weights = [1.0, 0.0]
    metric.beta = 1.0
    metric._generate_statements = AsyncMock(side_effect=[[], ["Reference claim"]])
    metric._classify_statements = AsyncMock()

    result = await metric.ascore(
        user_input="Question", response="No claim", reference="Reference claim"
    )

    assert result.value == 1.0  # Preserve the incumbent scoring behavior
    assert "no statements" in result.reason.lower()
    metric._classify_statements.assert_not_called()
