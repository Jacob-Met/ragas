import pytest

from ragas.async_utils import as_completed
from ragas.run_config import RunConfig


@pytest.mark.parametrize("workers", [0, -2])
def test_run_config_rejects_invalid_worker_count(workers):
    with pytest.raises(ValueError, match="max_workers"):
        RunConfig(max_workers=workers)


@pytest.mark.parametrize("workers", [0, -2])
def test_as_completed_rejects_invalid_worker_count_before_scheduling(workers):
    with pytest.raises(ValueError, match="max_workers"):
        as_completed([], max_workers=workers)


@pytest.mark.parametrize("workers", [-1, 1, 16])
def test_run_config_accepts_supported_worker_counts(workers):
    assert RunConfig(max_workers=workers).max_workers == workers
