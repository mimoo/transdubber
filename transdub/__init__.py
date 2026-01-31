"""
transdub - Translate and dub videos using AI voice cloning.

Quick start:
    from transdub import TransdubConfig, TransdubPipeline

    config = TransdubConfig(video_path="video.mp4")
    pipeline = TransdubPipeline(config)
    output = pipeline.run()
"""
from .config import TransdubConfig
from .pipeline import TransdubPipeline

__all__ = ["TransdubConfig", "TransdubPipeline"]
__version__ = "0.1.0"
