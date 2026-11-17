import asyncio
import threading
import time

import pytest

from ragas.executor import Executor
from ragas.run_config import RunConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 3, 20])
async def test_order_of_execution(batch_size):
    async def echo_order(index: int):
        await asyncio.sleep(1 / index)
        return index

    # Arrange
    executor = Executor(batch_size=batch_size)
    # add 10 jobs to the executor
    for i in range(1, 11):
        executor.submit(echo_order, i, name=f"echo_order_{i}")

    # Act
    results = executor.results()
    # Assert
    assert results == list(range(1, 11))


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 3, 20])
async def test_executor_in_script(batch_size):
    async def echo_order(index: int):
        await asyncio.sleep(1 / index)
        return index

    # Arrange
    executor = Executor(batch_size=batch_size)
    # add 10 jobs to the executor
    for i in range(1, 4):
        executor.submit(echo_order, i, name=f"echo_order_{i}")

    # Act
    results = executor.results()
    # Assert
    assert results == list(range(1, 4))


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 3, 20])
async def test_executor_with_running_loop(batch_size):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.sleep(0.1))

    async def echo_order(index: int):
        await asyncio.sleep(1 / index)
        return index

    # Arrange
    executor = Executor(batch_size=batch_size)
    for i in range(1, 4):
        executor.submit(echo_order, i, name=f"echo_order_{i}")

    # Act
    # add 10 jobs to the executor
    results = executor.results()
    # Assert
    assert results == list(range(1, 4))


def test_executor_timings():
    # if we submit n tasks that take 1 second each,
    # the total time taken should be close to 1 second

    executor = Executor()

    async def long_task():
        await asyncio.sleep(0.1)
        return 1

    n_tasks = 5
    for i in range(n_tasks):
        executor.submit(long_task, name=f"long_task_{i}")

    start_time = time.time()
    results = executor.results()
    end_time = time.time()
    assert len(results) == n_tasks
    assert all(r == 1 for r in results)
    assert end_time - start_time < 0.2


def test_executor_exception_handling():
    """Test that exceptions are returned as np.nan when raise_exceptions is False."""
    import numpy as np

    async def fail_task():
        raise ValueError("fail")

    executor = Executor()
    executor.submit(fail_task)
    results = executor.results()
    assert len(results) == 1
    assert np.isnan(results[0])


def test_executor_exception_raises():
    """Test that exceptions are raised when raise_exceptions is True."""

    async def fail_task():
        raise ValueError("fail")

    executor = Executor(raise_exceptions=True)
    executor.submit(fail_task)
    with pytest.raises(ValueError):
        executor.results()


def test_executor_empty_jobs():
    """Test that results() returns an empty list if no jobs are submitted."""
    executor = Executor()
    assert executor.results() == []


def test_executor_job_index_after_clear():
    """Test that job indices reset after clearing jobs."""

    async def echo(x):
        return x

    executor = Executor()
    executor.submit(echo, 1)
    executor.clear_jobs()
    executor.submit(echo, 42)
    results = executor.results()
    assert results == [42]


def test_executor_batch_size_edge_cases():
    """Test batch_size=1 and batch_size > number of jobs."""

    async def echo(x):
        return x

    # batch_size=1
    executor = Executor(batch_size=1)
    for i in range(3):
        executor.submit(echo, i)
    assert executor.results() == [0, 1, 2]
    # batch_size > jobs
    executor = Executor(batch_size=10)
    for i in range(3):
        executor.submit(echo, i)
    assert executor.results() == [0, 1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("max_workers", [1, 2])
async def test_executor_failure_stops_sibling_jobs(batch_size, max_workers):
    """A failed evaluation must not leave other jobs running after it raises."""
    blocker = asyncio.Event()
    escaped = asyncio.Event()

    async def fail():
        await asyncio.sleep(0)
        raise ValueError("bad metric")

    async def slow():
        await blocker.wait()
        escaped.set()

    executor = Executor(
        batch_size=batch_size,
        raise_exceptions=True,
        show_progress=False,
        run_config=RunConfig(max_workers=max_workers),
    )
    executor.submit(fail)
    executor.submit(slow)
    with pytest.raises(ValueError, match="bad metric"):
        await executor.aresults()

    blocker.set()
    await asyncio.sleep(0.02)
    assert not escaped.is_set(), "a sibling completed after aresults() raised"


@pytest.mark.asyncio
async def test_cancelled_executor_stops_sibling_jobs():
    """Cancelling the caller also stops the running evaluation jobs."""
    started = asyncio.Event()
    finished = asyncio.Event()
    blocker = asyncio.Event()

    async def slow():
        started.set()
        await blocker.wait()
        finished.set()

    executor = Executor(show_progress=False)
    executor.submit(slow)
    outer = asyncio.create_task(executor.aresults())
    await asyncio.wait_for(started.wait(), timeout=0.5)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    blocker.set()
    await asyncio.sleep(0.02)
    assert not finished.is_set(), "a cancelled executor left a job running"


@pytest.mark.asyncio
async def test_executor_failure_closes_unstarted_coroutines():
    """Cancelling one-worker queues must close jobs that never started."""
    import gc

    before = asyncio.all_tasks()
    started = []

    async def fail():
        raise ValueError("bad metric")

    async def queued(index):
        started.append(index)
        await asyncio.sleep(1)

    executor = Executor(
        raise_exceptions=True,
        show_progress=False,
        run_config=RunConfig(max_workers=1),
    )
    executor.submit(fail)
    for i in range(50):
        executor.submit(queued, i)

    with pytest.raises(ValueError, match="bad metric"):
        await executor.aresults()
    started_on_return = started.copy()
    await asyncio.sleep(0.02)
    gc.collect()

    assert started == started_on_return
    assert asyncio.all_tasks() - before == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("max_workers", [1, 2])
@pytest.mark.parametrize("from_thread", [False, True])
async def test_explicit_cancel_interrupts_blocked_evaluation(
    batch_size, max_workers, from_thread
):
    """cancel() must wake aresults() even when no metric finishes."""
    started = asyncio.Event()
    run_count = 0
    exit_count = 0

    async def blocked_metric():
        nonlocal run_count, exit_count
        run_count += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            exit_count += 1

    executor = Executor(
        show_progress=False,
        batch_size=batch_size,
        run_config=RunConfig(max_workers=max_workers),
    )
    executor.submit(blocked_metric)
    executor.submit(blocked_metric)
    result_task = asyncio.create_task(executor.aresults())
    await asyncio.wait_for(started.wait(), timeout=0.5)

    if from_thread:
        await asyncio.to_thread(executor.cancel)
    else:
        executor.cancel()

    assert await asyncio.wait_for(result_task, timeout=0.25) == []
    assert run_count == exit_count


@pytest.mark.asyncio
async def test_explicit_cancel_returns_completed_partial_results():
    """Cancellation retains finished scores but does not await blocked jobs."""
    completed = asyncio.Event()
    started = asyncio.Event()

    async def quick_metric():
        completed.set()
        return 42

    async def blocked_metric():
        started.set()
        await asyncio.Event().wait()

    executor = Executor(show_progress=False, run_config=RunConfig(max_workers=2))
    executor.submit(quick_metric)
    executor.submit(blocked_metric)
    result_task = asyncio.create_task(executor.aresults())
    await asyncio.wait_for(completed.wait(), timeout=0.5)
    await asyncio.wait_for(started.wait(), timeout=0.5)
    await asyncio.sleep(0)
    executor.cancel()

    assert await asyncio.wait_for(result_task, timeout=0.25) == [42]


@pytest.mark.asyncio
async def test_cancel_before_evaluation_starts():
    """An already-cancelled executor does not start newly submitted work."""
    started = asyncio.Event()

    async def blocked_metric():
        started.set()
        await asyncio.Event().wait()

    executor = Executor(show_progress=False)
    executor.cancel()
    executor.submit(blocked_metric)
    assert await asyncio.wait_for(executor.aresults(), timeout=0.5) == []
    assert not started.is_set()


def test_cancel_wakes_synchronous_results_from_another_thread():
    """The public sync API must be interruptible from a different thread."""
    started = threading.Event()
    ended = threading.Event()
    outcomes = []
    errors = []

    async def blocked_metric():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            ended.set()

    executor = Executor(show_progress=False)
    executor.submit(blocked_metric)

    def evaluate():
        try:
            outcomes.append(executor.results())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=evaluate, daemon=True)
    worker.start()
    assert started.wait(timeout=2)
    executor.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive(), "cancel() left the sync evaluation thread blocked"
    assert not errors
    assert outcomes == [[]]
    assert ended.is_set()
