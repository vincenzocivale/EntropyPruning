"""Evaluation APIs with optional GPU/profiling dependencies loaded on demand."""

__all__ = ["evaluate", "compute_tar_at_far", "benchmark_model"]


def __getattr__(name):
    if name == "benchmark_model":
        from .benchmark import benchmark_model
        return benchmark_model
    if name in {"evaluate", "compute_tar_at_far"}:
        from . import metrics
        return getattr(metrics, name)
    raise AttributeError(name)
