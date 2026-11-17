import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager

from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import aevaluate
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM, SingleTurnMetric


@dataclass
class MarkerMetric(MetricWithLLM, SingleTurnMetric):
    started: asyncio.Queue | None = None
    release: asyncio.Event | None = None

    async def _single_turn_ascore(self, sample, callbacks):
        if self.started is not None:
            self.started.put_nowait(self.llm.marker)
        if self.release is not None:
            await self.release.wait()
        return float(self.llm.marker)


@pytest.fixture
def evaluation_dataset():
    return EvaluationDataset(
        samples=[SingleTurnSample(user_input="question", response="answer")]
    )


@pytest.mark.asyncio
async def test_deferred_evaluations_isolate_their_models(evaluation_dataset):
    """A deferred executor must not borrow a previous evaluation's LLM."""
    metric = MarkerMetric()
    first = await aevaluate(
        evaluation_dataset,
        metrics=[metric],
        llm=SimpleNamespace(marker=1.0),
        return_executor=True,
        show_progress=False,
    )
    second = await aevaluate(
        evaluation_dataset,
        metrics=[metric],
        llm=SimpleNamespace(marker=2.0),
        return_executor=True,
        show_progress=False,
    )

    assert metric.llm is None
    assert await second.aresults() == [2.0]
    assert await first.aresults() == [1.0]
    assert metric.llm is None


@pytest.mark.asyncio
async def test_overlapping_evaluations_isolate_models(evaluation_dataset):
    """Two simultaneous aevaluate calls with one metric must not cross models."""
    started = asyncio.Queue()
    release = asyncio.Event()
    metric = MarkerMetric(started=started, release=release)

    first = asyncio.create_task(
        aevaluate(
            evaluation_dataset,
            metrics=[metric],
            llm=SimpleNamespace(marker=1.0),
            show_progress=False,
        )
    )
    await asyncio.wait_for(started.get(), timeout=1)
    second = asyncio.create_task(
        aevaluate(
            evaluation_dataset,
            metrics=[metric],
            llm=SimpleNamespace(marker=2.0),
            show_progress=False,
        )
    )
    second_started = await asyncio.wait_for(started.get(), timeout=1)
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert second_started == 2.0
    assert first_result["marker_metric"] == [1.0]
    assert second_result["marker_metric"] == [2.0]
    assert metric.llm is None


@pytest.mark.asyncio
async def test_configured_metric_keeps_its_own_llm(evaluation_dataset):
    """Caller-specified metric LLM retains precedence over evaluation defaults."""
    own_llm = SimpleNamespace(marker=9.0)
    metric = MarkerMetric(llm=own_llm)
    result = await aevaluate(
        evaluation_dataset,
        metrics=[metric],
        llm=SimpleNamespace(marker=2.0),
        show_progress=False,
    )
    assert result["marker_metric"] == [9.0]
    assert metric.llm is own_llm


class RecordingHandler(BaseCallbackHandler):
    def __init__(self):
        self.calls = 0

    def on_chain_start(self, *args, **kwargs):
        self.calls += 1


@pytest.mark.asyncio
async def test_reused_callback_list_does_not_accumulate_internal_handlers(
    evaluation_dataset,
):
    """Internal tracer/cost handlers must be scoped to each evaluation."""
    handler = RecordingHandler()
    callbacks = [handler]
    metric = MarkerMetric(llm=SimpleNamespace(marker=1.0))

    for _ in range(2):
        result = await aevaluate(
            evaluation_dataset,
            metrics=[metric],
            callbacks=callbacks,
            show_progress=False,
        )
        assert result["marker_metric"] == [1.0]
        assert callbacks == [handler]
    assert handler.calls >= 2


@pytest.mark.asyncio
async def test_callback_manager_metadata_and_handlers_are_not_mutated(
    evaluation_dataset,
):
    """A caller's configured callback manager survives an evaluation unchanged."""
    handler = RecordingHandler()
    callbacks = CallbackManager(
        [handler], tags=["customer"], metadata={"session": "original"}
    )
    metric = MarkerMetric(llm=SimpleNamespace(marker=1.0))
    result = await aevaluate(
        evaluation_dataset, metrics=[metric], callbacks=callbacks, show_progress=False
    )
    assert result["marker_metric"] == [1.0]
    assert callbacks.handlers == [handler]
    assert callbacks.tags == ["customer"]
    assert callbacks.metadata == {"session": "original"}
    assert handler.calls >= 1


@dataclass
class MarkerEmbeddingsMetric(MetricWithEmbeddings, SingleTurnMetric):
    async def _single_turn_ascore(self, sample, callbacks):
        return float(self.embeddings.marker)


@pytest.mark.asyncio
async def test_deferred_evaluations_isolate_embedding_models(evaluation_dataset):
    """Embedding-backed metrics also need per-evaluation binding."""
    metric = MarkerEmbeddingsMetric()
    first = await aevaluate(
        evaluation_dataset,
        metrics=[metric],
        embeddings=SimpleNamespace(marker=3.0),
        return_executor=True,
        show_progress=False,
    )
    second = await aevaluate(
        evaluation_dataset,
        metrics=[metric],
        embeddings=SimpleNamespace(marker=4.0),
        return_executor=True,
        show_progress=False,
    )
    assert metric.embeddings is None
    assert await first.aresults() == [3.0]
    assert await second.aresults() == [4.0]
    assert metric.embeddings is None


@pytest.mark.asyncio
async def test_failing_metric_initialization_does_not_change_caller_metric(
    evaluation_dataset,
):
    """An init failure must not leave a transient LLM on a reusable metric."""

    @dataclass
    class FailingMetric(MarkerMetric):
        def init(self, run_config):
            raise RuntimeError("failed to initialize")

    metric = FailingMetric()
    with pytest.raises(RuntimeError, match="failed to initialize"):
        await aevaluate(
            evaluation_dataset,
            metrics=[metric],
            llm=SimpleNamespace(marker=7.0),
            show_progress=False,
        )
    assert metric.llm is None
