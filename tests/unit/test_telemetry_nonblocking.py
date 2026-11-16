"""Evaluation must not wait on tracking DNS resolution or HTTP retries."""

import threading
import time

import requests

from ragas import _analytics as analytics


def _event():
    return analytics.EvaluationEvent(
        user_id="telemetry-test",
        event_type="evaluation",
        metrics=["faithfulness"],
        num_rows=1,
        language="english",
        evaluation_type="SINGLE_TURN",
    )


def test_track_does_not_wait_for_slow_dns(monkeypatch):
    blocked = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def blocked_post(*args, **kwargs):
        blocked.set()
        release.wait(timeout=0.4)
        completed.set()
        raise requests.ConnectionError("Temporary DNS failure")

    monkeypatch.setattr(analytics, "do_not_track", lambda: False)
    monkeypatch.setattr(analytics, "_usage_event_debugging", lambda: False)
    monkeypatch.setattr(analytics.requests, "post", blocked_post)

    started = time.monotonic()
    try:
        analytics.track(_event())
        elapsed = time.monotonic() - started
    finally:
        release.set()
        completed.wait(timeout=1.0)

    assert elapsed < 0.15, "Analytics must not block user work during DNS failure"
    assert blocked.wait(timeout=1.0)


def test_evaluation_batch_flush_does_not_wait_for_dns(monkeypatch):
    blocked = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def blocked_post(*args, **kwargs):
        blocked.set()
        release.wait(timeout=0.4)
        completed.set()
        raise requests.ConnectionError("Temporary DNS failure")

    monkeypatch.setattr(analytics, "do_not_track", lambda: False)
    monkeypatch.setattr(analytics, "_usage_event_debugging", lambda: False)
    monkeypatch.setattr(analytics.requests, "post", blocked_post)

    batcher = analytics.AnalyticsBatcher(batch_size=10, flush_interval=3600)
    batcher.add_evaluation(_event())
    started = time.monotonic()
    try:
        batcher.flush()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        completed.wait(timeout=1.0)
        batcher.shutdown()

    assert elapsed < 0.15, "Completed evaluations must not wait for telemetry"
    assert blocked.wait(timeout=1.0)


def test_telemetry_queue_is_bounded_when_dns_is_stuck(monkeypatch):
    blocked = threading.Event()
    release = threading.Event()
    sent = []

    def blocked_post(url, *, json, timeout):
        blocked.set()
        release.wait(timeout=0.5)
        sent.append(json)

    monkeypatch.setattr(analytics.requests, "post", blocked_post)
    transport = analytics._AnalyticsTransport(capacity=2)
    try:
        transport.submit({"sequence": 0})
        assert blocked.wait(timeout=0.5)

        started = time.monotonic()
        for sequence in range(1, 50):
            transport.submit({"sequence": sequence})
        elapsed = time.monotonic() - started

        assert elapsed < 0.15, "Queue saturation must not slow down evaluation"
        assert transport._queue.qsize() == 2
    finally:
        release.set()
        transport._queue.join()

    assert len(sent) == 3


def test_telemetry_worker_survives_dns_failure(monkeypatch):
    sent = []
    finished = threading.Event()

    def flaky_post(url, *, json, timeout):
        sent.append(json)
        if len(sent) == 1:
            raise requests.ConnectionError("Temporary DNS failure")
        finished.set()

    monkeypatch.setattr(analytics.requests, "post", flaky_post)
    transport = analytics._AnalyticsTransport(capacity=3)
    transport.submit({"sequence": 1})
    transport.submit({"sequence": 2})

    assert finished.wait(timeout=1.0)
    assert sent == [{"sequence": 1}, {"sequence": 2}]


def test_tracking_opt_out_never_queues_telemetry(monkeypatch):
    from unittest.mock import Mock

    submit = Mock()
    monkeypatch.setattr(analytics, "do_not_track", lambda: True)
    monkeypatch.setattr(analytics._analytics_transport, "submit", submit)

    analytics.track(_event())

    submit.assert_not_called()
