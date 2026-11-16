"""Precision and recall must count true positives in their own directions."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._factual_correctness import FactualCorrectness


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "beta", "expected"),
    [
        ("precision", 1.0, 0.75),
        ("recall", 1.0, 0.67),
        ("f1", 1.0, 0.71),
        ("f1", 2.0, 0.68),
    ],
)
async def test_factual_correctness_uses_directional_true_positives(
    mode, beta, expected
):
    metric = FactualCorrectness(mode=mode, beta=beta)
    metric.llm = object()
    metric.decompose_and_verify_claims = AsyncMock(
        side_effect=[
            np.array([True, True, True, False], dtype=bool),
            np.array([True, True, False], dtype=bool),
        ]
    )

    score = await metric._single_turn_ascore(
        SingleTurnSample(reference="Reference", response="Response"), callbacks=None
    )
    assert score == expected
    assert metric.decompose_and_verify_claims.await_count == (
        1 if mode == "precision" else 2
    )


@pytest.mark.asyncio
async def test_recall_cannot_exceed_reference_claim_support_ratio():
    metric = FactualCorrectness(mode="recall")
    metric.llm = object()
    metric.decompose_and_verify_claims = AsyncMock(
        side_effect=[
            np.array([True] * 9, dtype=bool),
            np.array([True, False, False], dtype=bool),
        ]
    )
    score = await metric._single_turn_ascore(
        SingleTurnSample(reference="Reference", response="Response"), callbacks=None
    )
    assert score == 0.33


@pytest.mark.asyncio
async def test_recall_empty_reference_claims_scores_zero():
    metric = FactualCorrectness(mode="recall")
    metric.llm = object()
    metric.decompose_and_verify_claims = AsyncMock(
        side_effect=[
            np.array([True] * 3, dtype=bool),
            np.array([], dtype=bool),
        ]
    )
    score = await metric._single_turn_ascore(
        SingleTurnSample(reference="Reference", response="Response"), callbacks=None
    )
    assert score == 0.0
