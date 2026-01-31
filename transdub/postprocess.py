"""Post-processing utilities for dubbed videos."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def detect_silence_regions(
    audio_path: Path,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Detect regions of silence in audio.

    Returns list of (start, end) tuples in seconds.
    """
    audio, sr = sf.read(audio_path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Convert to dB
    eps = 1e-10
    db = 20 * np.log10(np.abs(audio) + eps)

    # Find silent samples
    is_silent = db < threshold_db

    # Find silence regions
    regions = []
    in_silence = False
    start = 0

    min_samples = int(min_silence_duration * sr)

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            start = i
        elif not silent and in_silence:
            in_silence = False
            if i - start >= min_samples:
                regions.append((start / sr, i / sr))

    if in_silence and len(audio) - start >= min_samples:
        regions.append((start / sr, len(audio) / sr))

    return regions


def create_cut_filter(
    silence_regions: list[tuple[float, float]],
    max_silence: float = 0.3,
    total_duration: float = None,
) -> list[tuple[float, float]]:
    """
    Create list of segments to keep, cutting long silences.

    Args:
        silence_regions: List of (start, end) silence regions
        max_silence: Maximum silence to keep (seconds)
        total_duration: Total video duration

    Returns:
        List of (start, end) segments to keep
    """
    if not silence_regions:
        return [(0, total_duration)] if total_duration else []

    keep_segments = []
    last_end = 0

    for silence_start, silence_end in silence_regions:
        silence_duration = silence_end - silence_start

        if silence_duration > max_silence:
            # Keep content before silence
            if silence_start > last_end:
                keep_segments.append((last_end, silence_start + max_silence / 2))

            # Skip to end of silence minus small buffer
            last_end = silence_end - max_silence / 2

    # Keep final segment
    if total_duration and last_end < total_duration:
        keep_segments.append((last_end, total_duration))

    return keep_segments


def cut_silences(
    input_video: Path,
    output_video: Path,
    max_silence: float = 0.3,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.5,
) -> Path:
    """
    Cut long silences from a video.

    Args:
        input_video: Input video path
        output_video: Output video path
        max_silence: Maximum silence duration to keep
        threshold_db: Silence threshold in dB
        min_silence_duration: Minimum silence duration to consider cutting

    Returns:
        Path to output video
    """
    import ffmpeg

    # Extract audio for analysis
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        # Extract audio
        (
            ffmpeg
            .input(str(input_video))
            .output(audio_path, acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Get video duration
        probe = ffmpeg.probe(str(input_video))
        duration = float(probe["format"]["duration"])

        # Detect silences
        silences = detect_silence_regions(
            Path(audio_path),
            threshold_db=threshold_db,
            min_silence_duration=min_silence_duration,
        )

        if not silences:
            # No silences to cut, just copy
            (
                ffmpeg
                .input(str(input_video))
                .output(str(output_video), c="copy")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_video

        # Create segments to keep
        segments = create_cut_filter(silences, max_silence, duration)

        if not segments:
            raise ValueError("No segments to keep after cutting")

        # Build ffmpeg filter for concatenation
        # Using select filter and concat
        inputs = []
        for i, (start, end) in enumerate(segments):
            inputs.append(
                ffmpeg
                .input(str(input_video), ss=start, t=end - start)
            )

        if len(inputs) == 1:
            # Single segment, just trim
            (
                ffmpeg
                .input(str(input_video), ss=segments[0][0], t=segments[0][1] - segments[0][0])
                .output(str(output_video), c="copy")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        else:
            # Multiple segments, need to concat
            # Write concat file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                concat_file = f.name
                for start, end in segments:
                    f.write(f"file '{input_video}'\n")
                    f.write(f"inpoint {start}\n")
                    f.write(f"outpoint {end}\n")

            try:
                (
                    ffmpeg
                    .input(concat_file, f="concat", safe=0)
                    .output(str(output_video), c="copy")
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            finally:
                Path(concat_file).unlink()

        return output_video

    finally:
        Path(audio_path).unlink(missing_ok=True)


def get_silence_stats(
    video_path: Path,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.5,
) -> dict:
    """Get statistics about silences in a video."""
    import ffmpeg

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(audio_path, acodec="pcm_s16le", ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        probe = ffmpeg.probe(str(video_path))
        duration = float(probe["format"]["duration"])

        silences = detect_silence_regions(
            Path(audio_path),
            threshold_db=threshold_db,
            min_silence_duration=min_silence_duration,
        )

        total_silence = sum(end - start for start, end in silences)

        return {
            "duration": duration,
            "silence_count": len(silences),
            "total_silence": total_silence,
            "silence_percent": (total_silence / duration) * 100 if duration > 0 else 0,
            "silences": silences,
        }
    finally:
        Path(audio_path).unlink(missing_ok=True)
