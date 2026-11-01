"""Volleyball analysis from one broadcast camera.

Detect (RF-DETR) -> track -> pose (RTMPose) -> court homography -> touches -> rallies ->
skills from touch order -> jersey identity (PARSeq + DINOv2 + colour) -> DataVolley .dvw.
See README.md and the notebooks.
"""
from .config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
