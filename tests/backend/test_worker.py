from backend.workers.tasks import compute_info_task


def test_compute_info_task_runs_eagerly():
    result = compute_info_task.delay().get(timeout=5)
    assert result["recommended_device"] in {"cpu", "mps", "cuda"}
