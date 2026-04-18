"""Songfuse — generate seamless transitions between two audio tracks."""

from songfuse.analysis import TrackAnalysis, analyze
from songfuse.fuse import FuseResult, fuse

__all__ = ["analyze", "fuse", "TrackAnalysis", "FuseResult"]
__version__ = "0.1.0"
