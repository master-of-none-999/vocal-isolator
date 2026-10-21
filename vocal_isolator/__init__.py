"""Local vocal isolation built on the Demucs inference path."""

from .pipeline import SeparationResult, separate_file

__all__ = ["SeparationResult", "separate_file"]
__version__ = "1.0.0"
