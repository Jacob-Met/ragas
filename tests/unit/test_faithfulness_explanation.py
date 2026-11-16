"""Faithfulness should expose the judge's existing verdict reasons to callers."""

import math
from unittest.mock import AsyncMock

import pytest

from ragas.metrics.collections.faithfulness.metric import Faithfulness
from ragas.metrics.collections.faithfulness.util import (
    NLIStatementOutput,
    StatementFaithfulnessAnswer,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verdicts", "expected_score", "unsupported_phrase"),
    [
        (
            [
                StatementFaithfulnessAnswer(
                    statement="Paris is in France.",
                    reason="The context states Paris is in France.",
                    verdict=1,
                ),
                StatementFaithfulnessAnswer(
                    statement="Paris is in Germany.",
                    reason="The context places Paris in France, not Germany.",
                    verdict=0,
                ),
            ],
            0.5,
            "not Germany",
        ),
        (
            [
                StatementFaithfulnessAnswer(
                    statement="Paris is in France.",
                    reason="The context directly supports this.",
                    verdict=1,
                )
            ],
            1.0,
            None,
        ),
        (
            [
                StatementFaithfulnessAnswer(
                    statement="Paris is in Germany.",
                    reason="No context supports Germany.",
                    verdict=0,
                )
            ],
            0.0,
            "No context supports Germany",
        ),
    ],
)
async def test_faithfulness_preserves_judge_explanations(
    verdicts, expected_score, unsupported_phrase
):
    metric = object.__new__(Faithfulness)
    metric._create_statements = AsyncMock(
        return_value=[item.statement for item in verdicts]
    )
    metric._create_verdicts = AsyncMock(
        return_value=NLIStatementOutput(statements=verdicts)
    )

    result = await metric.ascore(
        user_input="Where is Paris?",
        response="Paris is in France.",
        retrieved_contexts=["Paris is in France."],
    )

    assert result.value == expected_score
    assert result.reason
    assert result.traces
    assert result.traces["output"]["statements"] == [
        item.model_dump() for item in verdicts
    ]
    if unsupported_phrase:
        assert unsupported_phrase in result.reason
    assert result.to_dict()["reason"] == result.reason


@pytest.mark.asyncio
async def test_faithfulness_explains_missing_statements():
    metric = object.__new__(Faithfulness)
    metric._create_statements = AsyncMock(return_value=[])
    metric._create_verdicts = AsyncMock()
    result = await metric.ascore(
        user_input="Where is Paris?",
        response="I can't tell",
        retrieved_contexts=["Paris is in France"],
    )
    assert math.isnan(result.value)
    assert "no statements" in result.reason.lower()
    metric._create_verdicts.assert_not_called()


@pytest.mark.asyncio
async def test_faithfulness_explains_empty_nli_results():
    metric = object.__new__(Faithfulness)
    metric._create_statements = AsyncMock(return_value=["Claim"])
    metric._create_verdicts = AsyncMock(return_value=NLIStatementOutput(statements=[]))
    result = await metric.ascore(
        user_input="Question",
        response="Claim",
        retrieved_contexts=["Context"],
    )
    assert math.isnan(result.value)
    assert "no verdict" in result.reason.lower()
