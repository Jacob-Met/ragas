"""Exact matches have a BLEU score of 1, even for short or CJK text."""

import pytest

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._bleu_score import BleuScore as LegacyBleuScore
from ragas.metrics.collections import BleuScore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "今天天气很不错",
        "Hello world",
        "Paris is in France. Berlin is in Germany.",
    ],
)
async def test_exact_matches_return_one_legacy(text):
    score = await LegacyBleuScore()._single_turn_ascore(
        SingleTurnSample(reference=text, response=text), callbacks=None
    )
    assert score == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "今天天气很不错",
        "Hello world",
        "Paris is in France. Berlin is in Germany.",
    ],
)
async def test_exact_matches_return_one_collections(text):
    result = await BleuScore().ascore(reference=text, response=text)
    assert result.value == 1.0


@pytest.mark.asyncio
async def test_nonmatching_short_text_not_scored_exact():
    result = await BleuScore().ascore(reference="Hello world", response="Hello there")
    assert 0.0 <= result.value < 1.0


@pytest.mark.asyncio
async def test_legacy_override_effective_order():
    score = await LegacyBleuScore(
        kwargs={"use_effective_order": False}
    )._single_turn_ascore(
        SingleTurnSample(reference="Hello world", response="Hello world"),
        callbacks=None,
    )
    assert score == 0.0


@pytest.mark.asyncio
async def test_chinese_character_tokenizer_opt_in_measures_partial_match():
    result = await BleuScore(kwargs={"tokenize": "zh"}).ascore(
        reference="今天天气很不错", response="今天天气不错"
    )
    assert 0.0 < result.value < 1.0
