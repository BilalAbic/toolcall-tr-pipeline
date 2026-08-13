"""Small, source-specific, lossless adapters."""

from toolcall_tr.adapters.base import SourceAdapter
from toolcall_tr.adapters.no_tool import NoToolAdapter
from toolcall_tr.adapters.when2call import When2CallAdapter
from toolcall_tr.adapters.when2call_training import When2CallTrainingAdapter
from toolcall_tr.adapters.xlam import XlamAdapter
from toolcall_tr.adapters.xlam60k import Xlam60kAdapter

ADAPTERS: dict[str, type[SourceAdapter]] = {
    "xlam": XlamAdapter,
    "no_tool": NoToolAdapter,
    "when2call": When2CallAdapter,
    "when2call_training": When2CallTrainingAdapter,
    "xlam60k": Xlam60kAdapter,
}


def get_adapter(name: str) -> SourceAdapter:
    try:
        return ADAPTERS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown adapter: {name}; choose one of {sorted(ADAPTERS)}") from exc


__all__ = [
    "ADAPTERS",
    "NoToolAdapter",
    "SourceAdapter",
    "When2CallAdapter",
    "When2CallTrainingAdapter",
    "Xlam60kAdapter",
    "XlamAdapter",
    "get_adapter",
]
