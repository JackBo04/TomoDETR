"""TomoDETR reference implementation for the review manuscript."""
from .detector import LocalDetector
from .cwqr import CWQR

__all__ = ["LocalDetector", "CWQR"]
