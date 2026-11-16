"""Utils to help interact with Langfuse traces."""

__all__ = ["observe", "logger", "LangfuseTrace", "sync_trace", "add_query_param"]

import asyncio
import logging
import typing as t
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    from langfuse import Langfuse, observe

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None  # type: ignore[assignment, misc]

    def observe(*args, **kwargs):  # type: ignore
        """Preserve importability when tracing is not installed."""

        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


@dataclass
class _ObservationTrace:
    """A trace assembled from the real v4 observations endpoint."""

    id: str
    observations: t.List[t.Any]


class LangfuseTrace:
    def __init__(self, trace: t.Any, client: t.Optional[t.Any] = None):
        self.trace = trace
        if client is None:
            if Langfuse is None:
                raise ImportError("Install ragas[tracing] to use Langfuse tracing")
            client = Langfuse()
        self._langfuse_client = client

    def get_url(self) -> t.Optional[str]:
        return self._langfuse_client.get_trace_url(trace_id=self.trace.id)

    def filter(self, span_name: str) -> t.List[t.Any]:
        return [
            observation
            for observation in self.trace.observations
            if observation.name == span_name
        ]


async def sync_trace(
    trace_id: t.Optional[str] = None, max_retries: int = 10, delay: float = 2
) -> LangfuseTrace:
    """Wait for a trace's observations to arrive in Langfuse.

    The v4 observations endpoint replaces the deprecated trace read endpoint.
    Fetching all pages happens off the event loop.
    """
    if not LANGFUSE_AVAILABLE or Langfuse is None:
        raise ImportError("Install ragas[tracing] to use Langfuse tracing")

    langfuse_client = Langfuse()
    if trace_id is None:
        trace_id = langfuse_client.get_current_trace_id()

    if not trace_id:
        raise ValueError(
            "No trace id found. Please ensure you are running this function within a function decorated with @observe()."
        )

    def fetch_observations() -> t.List[t.Any]:
        observations: t.List[t.Any] = []
        cursor: t.Optional[str] = None
        seen_cursors: t.Set[str] = set()

        while True:
            page = langfuse_client.api.observations.get_many(
                trace_id=trace_id, fields="core,basic,io", limit=100, cursor=cursor
            )
            observations.extend(page.data)
            cursor = page.meta.cursor
            if cursor is None:
                return observations
            if cursor in seen_cursors:
                raise RuntimeError(
                    f"Langfuse returned a repeated observations cursor for {trace_id}"
                )
            seen_cursors.add(cursor)

    for attempt in range(max_retries):
        observations = await asyncio.to_thread(fetch_observations)
        if observations:
            return LangfuseTrace(
                trace=_ObservationTrace(id=trace_id, observations=observations),
                client=langfuse_client,
            )
        if attempt + 1 < max_retries:
            await asyncio.sleep(delay)

    raise ValueError(f"Trace {trace_id} not found after {max_retries} attempts")


def add_query_param(url: str, param_name: str, param_value: str) -> str:
    """Add a query parameter to a URL."""
    url_parts = list(urlparse(url))
    query_dict = dict(parse_qsl(url_parts[4]))
    query_dict[param_name] = param_value
    url_parts[4] = urlencode(query_dict)
    return urlunparse(url_parts)
