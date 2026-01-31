"""Post-processing utilities for dubbed videos."""
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
    min_silence_duration: float = 1.0,
    progress_callback=None,
) -> Path:
    """
    Cut long silences from a video using moviepy.

    Args:
        input_video: Input video path
        output_video: Output video path
        max_silence: Maximum silence duration to keep
        threshold_db: Silence threshold in dB
        min_silence_duration: Minimum silence duration to consider cutting

    Returns:
        Path to output video
    """
    from moviepy import VideoFileClip, concatenate_videoclips
    import ffmpeg

    # Extract audio for analysis
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        if progress_callback:
            progress_callback("Extracting audio for analysis...")

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

        if progress_callback:
            progress_callback("Detecting silences...")

        silences = detect_silence_regions(
            Path(audio_path),
            threshold_db=threshold_db,
            min_silence_duration=min_silence_duration,
        )

        if not silences:
            if progress_callback:
                progress_callback("No significant silences found, copying...")
            (
                ffmpeg
                .input(str(input_video))
                .output(str(output_video), c="copy")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_video

        segments = create_cut_filter(silences, max_silence, duration)

        if not segments:
            raise ValueError("No segments to keep after cutting")

        total_keep = sum(end - start for start, end in segments)
        if progress_callback:
            progress_callback(f"Keeping {len(segments)} segments ({total_keep:.0f}s of {duration:.0f}s)")

        # Load video with moviepy
        if progress_callback:
            progress_callback("Loading video...")
        video = VideoFileClip(str(input_video))

        # Extract segments
        if progress_callback:
            progress_callback(f"Extracting {len(segments)} segments...")

        clips = []
        for i, (start, end) in enumerate(segments):
            if progress_callback and i % 10 == 0:
                progress_callback(f"  Segment {i}/{len(segments)}")
            clips.append(video.subclipped(start, min(end, video.duration)))

        # Concatenate
        if progress_callback:
            progress_callback("Concatenating clips...")
        final = concatenate_videoclips(clips)

        # Write output
        if progress_callback:
            progress_callback("Writing video (this takes a few minutes)...")

        final.write_videofile(
            str(output_video),
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            logger=None,
        )

        video.close()
        final.close()

        return output_video

    finally:
        Path(audio_path).unlink(missing_ok=True)


def remap_subtitles(
    srt_path: Path,
    output_path: Path,
    segments: list[tuple[float, float]],
) -> Path:
    """
    Remap subtitle timings to match cut video.

    Args:
        srt_path: Original SRT file
        output_path: Output SRT file with remapped timings
        segments: List of (start, end) segments that were kept

    Returns:
        Path to output SRT
    """
    from .srt_utils import parse_srt, SRTSegment

    subs = parse_srt(srt_path)

    # Build a mapping from original time to new time
    # New time = original time minus all the gaps before it
    def map_time(t: float) -> float | None:
        """Map original time to new time, or None if in a cut region."""
        new_t = 0.0
        for seg_start, seg_end in segments:
            if t < seg_start:
                # Time is before this segment (in a cut region)
                return None
            elif t <= seg_end:
                # Time is within this segment
                return new_t + (t - seg_start)
            else:
                # Time is after this segment, accumulate duration
                new_t += seg_end - seg_start
        return None  # After all segments

    # Remap subtitles
    remapped = []
    for sub in subs:
        new_start = map_time(sub.start)
        new_end = map_time(sub.end)

        # Skip if subtitle is entirely in a cut region
        if new_start is None and new_end is None:
            continue

        # If partially in cut region, adjust
        if new_start is None:
            new_start = 0.0
        if new_end is None:
            # Find the end of the last segment before this sub
            new_end = new_start + 0.5  # Minimum duration

        if new_end <= new_start:
            new_end = new_start + 0.5

        # Format timestamps
        def fmt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

        remapped.append(SRTSegment(
            index=len(remapped) + 1,
            start=new_start,
            end=new_end,
            start_str=fmt(new_start),
            end_str=fmt(new_end),
            text=sub.text,
        ))

    # Write output
    from .srt_utils import write_srt
    write_srt(remapped, output_path)

    return output_path


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
