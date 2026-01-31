"""Configuration for transdub pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TransdubConfig:
    """Configuration for the transdub pipeline."""

    # Input
    video_path: Path

    # Output
    output_dir: Optional[Path] = None  # Default: same as video

    # Target language
    target_language: str = "English"
    source_language: Optional[str] = None  # Auto-detect if None

    # Whisper settings
    whisper_model: str = "medium"

    # Voice reference (auto-detected if not provided)
    ref_start: Optional[float] = None
    ref_duration: float = 20.0
    ref_text: Optional[str] = None

    # TTS settings
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    max_chunk_duration: float = 12.0

    # Segments to skip (different speakers, intro, etc.)
    skip_segments: list[int] = field(default_factory=list)

    # Translation improvement
    improve_translation: bool = False
    anthropic_api_key: Optional[str] = None

    # Processing options
    parallel_workers: int = 1  # For future parallel processing

    def __post_init__(self):
        self.video_path = Path(self.video_path)
        if self.output_dir is None:
            self.output_dir = self.video_path.parent
        else:
            self.output_dir = Path(self.output_dir)

    @property
    def video_stem(self) -> str:
        return self.video_path.stem

    @property
    def transcribed_srt(self) -> Path:
        return self.output_dir / f"{self.video_stem}_transcribed.srt"

    @property
    def translated_srt(self) -> Path:
        return self.output_dir / f"{self.video_stem}_translated.srt"

    @property
    def improved_srt(self) -> Path:
        return self.output_dir / f"{self.video_stem}_improved.srt"

    @property
    def output_video(self) -> Path:
        return self.output_dir / f"{self.video_stem}_{self.target_language.lower()}.mp4"
