"""MAS-OS — Agentic Operating System kernel package."""

__version__ = "2.0.0"
__author__  = "Jyotirmoy Singh — BITS Pilani Hyderabad"
__all__     = ["MASKernel", "create_kernel"]


def __getattr__(name: str):
    """
    Lazy-load kernel symbols to avoid pulling in langgraph/jsonschema
    when only mas.eval.* is imported (e.g. the eval CLI or figure scripts).

    ``from mas import MASKernel`` and ``import mas; mas.create_kernel``
    continue to work identically — the import is simply deferred until
    first use rather than at package load time.
    """
    if name in ("MASKernel", "create_kernel"):
        from mas.kernel import MASKernel, create_kernel  # noqa: PLC0415
        globals()["MASKernel"]    = MASKernel
        globals()["create_kernel"] = create_kernel
        return globals()[name]
    raise AttributeError(f"module 'mas' has no attribute {name!r}")
