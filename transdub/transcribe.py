"""Transcription and translation using Whisper."""
import tempfile
from pathlib import Path

import ffmpeg
import whisper


def extract_audio(video_path: Path, output_path: str) -> None:
    """Extract audio from video as 16kHz WAV."""
    (
        ffmpeg
        .input(str(video_path))
        .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def transcribe_video(
    video_path: Path,
    output_path: Path,
    model_name: str = "medium",
    task: str = "transcribe",
    language: str | None = None,
) -> dict:
    """
    Transcribe or translate a video.

    Args:
        video_path: Path to video file
        output_path: Path to output SRT file
        model_name: Whisper model size
        task: "transcribe" or "translate"
        language: Source language code (auto-detect if None)

    Returns:
        Whisper result dict with segments and detected language
    """
    # Extract audio to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        extract_audio(video_path, tmp_path)

        # Load model and transcribe
        model = whisper.load_model(model_name)

        options = {"task": task}
        if language:
            options["language"] = language

        result = model.transcribe(tmp_path, **options)

        # Write SRT
        write_srt(result["segments"], output_path)

        return result

    finally:
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write_srt(segments: list, output_path: Path) -> None:
    """Write segments to SRT format."""
    def format_ts(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_ts(seg['start'])} --> {format_ts(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")
