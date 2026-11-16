"""Context metrics expose reasons that are already produced by their judge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from ragas.metrics.collections.context_precision.metric import (
    ContextPrecision,
    ContextPrecisionWithoutReference,
)
from ragas.metrics.collections.context_precision.util import ContextPrecisionOutput
from ragas.metrics.collections.context_recall.metric import ContextRecall
from ragas.metrics.collections.context_recall.util import (
    ContextRecallClassification,
    ContextRecallOutput,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metric_class", [ContextPrecision, ContextPrecisionWithoutReference]
)
async def test_context_precision_exposes_per_context_reasons(metric_class):
    judgments = [
        ContextPrecisionOutput(verdict=1, reason="First passage answers the question"),
        ContextPrecisionOutput(
            verdict=0, reason="Second passage is about a different topic"
        ),
    ]
    metric = object.__new__(metric_class)
    metric.llm = SimpleNamespace(agenerate=AsyncMock(side_effect=judgments))
    metric.prompt = SimpleNamespace(to_string=lambda prompt: "Judge context")
    options = (
        {"reference": "Paris"}
        if metric_class is ContextPrecision
        else {"response": "Paris"}
    )

    result = await metric.ascore(
        user_input="What is the capital of France?",
        retrieved_contexts=["Paris is the capital", "Oslo is in Norway"],
        **options,
    )

    assert np.isclose(result.value, 1.0)
    assert "1 of 2" in result.reason
    assert "different topic" in result.reason
    assert result.traces["output"]["contexts"] == [
        {"index": 0, **judgments[0].model_dump()},
        {"index": 1, **judgments[1].model_dump()},
    ]


@pytest.mark.asyncio
async def test_context_recall_exposes_claim_reasons():
    judgments = ContextRecallOutput(
        classifications=[
            ContextRecallClassification(
                statement="Paris is the capital",
                attributed=1,
                reason="Explicitly stated",
            ),
            ContextRecallClassification(
                statement="Paris is in Germany",
                attributed=0,
                reason="Only France mentioned",
            ),
        ]
    )
    metric = object.__new__(ContextRecall)
    metric.llm = SimpleNamespace(agenerate=AsyncMock(return_value=judgments))
    metric.prompt = SimpleNamespace(to_string=lambda prompt: "Judge reference")

    result = await metric.ascore(
        user_input="Where is Paris?",
        retrieved_contexts=["Paris is in France"],
        reference="Paris is the capital and in Germany",
    )

    assert result.value == 0.5
    assert "1 of 2" in result.reason
    assert "Only France mentioned" in result.reason
    assert result.traces["output"]["classifications"] == [
        item.model_dump() for item in judgments.classifications
    ]


@pytest.mark.asyncio
async def test_context_recall_explains_missing_classifications():
    metric = object.__new__(ContextRecall)
    metric.llm = SimpleNamespace(
        agenerate=AsyncMock(return_value=ContextRecallOutput(classifications=[]))
    )
    metric.prompt = SimpleNamespace(to_string=lambda prompt: "Judge reference")

    result = await metric.ascore(
        user_input="Question", retrieved_contexts=["Evidence"], reference="Claim"
    )

    assert np.isnan(result.value)
    assert "no classifications" in result.reason.lower()
