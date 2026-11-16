"""Langfuse v4 tracing must return real observations, not a fabricated trace."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ragas.integrations.tracing.langfuse import sync_trace


@pytest.mark.asyncio
async def test_sync_trace_reads_all_observation_pages():
    client = MagicMock()
    first = SimpleNamespace(name="retrieval", id="o1")
    second = SimpleNamespace(name="generation", id="o2")
    client.api.observations.get_many.side_effect = [
        SimpleNamespace(data=[first], meta=SimpleNamespace(cursor="page-2")),
        SimpleNamespace(data=[second], meta=SimpleNamespace(cursor=None)),
    ]

    with patch("ragas.integrations.tracing.langfuse.Langfuse", return_value=client):
        result = await sync_trace(trace_id="trace-123", max_retries=1, delay=0)

    assert result.trace.id == "trace-123"
    assert result.filter("retrieval") == [first]
    assert result.filter("generation") == [second]
    assert client.api.observations.get_many.call_count == 2
    assert result.get_url() == client.get_trace_url.return_value
    client.get_trace_url.assert_called_once_with(trace_id="trace-123")


@pytest.mark.asyncio
async def test_sync_trace_never_returns_an_empty_fabricated_trace():
    client = MagicMock()
    client.api.observations.get_many.return_value = SimpleNamespace(
        data=[], meta=SimpleNamespace(cursor=None)
    )
    with patch("ragas.integrations.tracing.langfuse.Langfuse", return_value=client):
        with pytest.raises(ValueError, match="not found after 2 attempts"):
            await sync_trace(trace_id="missing", max_retries=2, delay=0)
    assert client.api.observations.get_many.call_count == 2
