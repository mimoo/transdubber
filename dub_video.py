#!/usr/bin/env python3
"""
Video dubbing using Qwen3-TTS voice cloning.

Takes a video, reference audio segment, and translated SRT to produce
a dubbed video with cloned voice in the target language.
"""
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Suppress noisy warnings from transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*model of type qwen3_tts.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

import click
import numpy as np
import soundfile as sf
from tqdm import tqdm

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"


@dataclass
class SRTSegment:
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Chunk:
    """A group of segments to be synthesized together."""
    segments: list[SRTSegment]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


def parse_srt_timestamp(ts: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_srt(srt_path: str) -> list[SRTSegment]:
    """Parse an SRT file into segments."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newline (segment separator)
    blocks = re.split(r"\n\n+", content.strip())
    segments = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
            timestamps = lines[1]
            text = " ".join(lines[2:]).strip()

            # Parse timestamps
            match = re.match(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", timestamps)
            if not match:
                continue

            start = parse_srt_timestamp(match.group(1))
            end = parse_srt_timestamp(match.group(2))

            if text:  # Skip empty segments
                segments.append(SRTSegment(index, start, end, text))
        except (ValueError, IndexError):
            continue

    return segments


def parse_skip_segments(skip_str: str) -> set[int]:
    """Parse skip segments string like '1-5,10,15-20' into a set of segment numbers."""
    skip_set = set()
    if not skip_str:
        return skip_set

    for part in skip_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            skip_set.update(range(int(start), int(end) + 1))
        else:
            skip_set.add(int(part))

    return skip_set


def group_segments_into_chunks(segments: list[SRTSegment], max_gap: float = 2.0, max_duration: float = 30.0) -> list[Chunk]:
    """
    Group consecutive segments into chunks for more natural TTS.

    Args:
        max_gap: Maximum gap between segments to group them (seconds)
        max_duration: Maximum duration of a chunk (seconds)
    """
    if not segments:
        return []

    chunks = []
    current_chunk_segments = [segments[0]]

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        gap = curr.start - prev.end
        chunk_duration = curr.end - current_chunk_segments[0].start

        # Start new chunk if gap too large or chunk too long
        if gap > max_gap or chunk_duration > max_duration:
            chunks.append(Chunk(current_chunk_segments))
            current_chunk_segments = [curr]
        else:
            current_chunk_segments.append(curr)

    # Don't forget the last chunk
    if current_chunk_segments:
        chunks.append(Chunk(current_chunk_segments))

    return chunks


def extract_reference_audio(
    video_path: str,
    output_path: str,
    start: float,
    duration: float,
    sample_rate: int = 24000,
) -> None:
    """Extract reference audio segment from video."""
    import ffmpeg

    try:
        (
            ffmpeg
            .input(video_path, ss=start)
            .output(output_path, acodec="pcm_s16le", ac=1, ar=sample_rate, t=duration)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise click.ClickException(f"Failed to extract audio: {e.stderr.decode()}")


def generate_chunk_audio(
    model,
    text: str,
    ref_audio_path: str,
    ref_text: str,
    language: str,
    sample_rate: int = 24000,
) -> tuple[np.ndarray, int]:
    """Generate TTS audio for a chunk of text."""
    import mlx.core as mx

    results = list(model.generate(
        text=text,
        ref_audio=ref_audio_path,
        ref_text=ref_text,
        language=language,
    ))

    if not results:
        return np.array([], dtype=np.float32), sample_rate

    audio = results[0].audio
    if hasattr(results[0], 'sample_rate'):
        sample_rate = results[0].sample_rate

    # Convert MLX array to numpy
    if isinstance(audio, mx.array):
        audio = np.array(audio.tolist(), dtype=np.float32)
    else:
        audio = np.array(audio, dtype=np.float32)

    return audio, sample_rate


def place_audio_in_timeline(
    chunks: list[Chunk],
    audio_clips: list[np.ndarray],
    total_duration: float,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Place audio clips into a timeline based on chunk start times, avoiding overlaps."""
    total_samples = int(total_duration * sample_rate)
    combined = np.zeros(total_samples, dtype=np.float32)

    # Track where the last audio ended to prevent overlap
    last_end_sample = 0

    for chunk, audio in zip(chunks, audio_clips):
        if len(audio) == 0:
            continue

        start_sample = int(chunk.start * sample_rate)

        # Prevent overlap: if this chunk would start before last one ended,
        # push it to start after the last one (with small gap)
        if start_sample < last_end_sample:
            gap_samples = int(0.1 * sample_rate)  # 100ms gap
            start_sample = last_end_sample + gap_samples

        # Calculate how much audio we can fit before next chunk or end
        available_space = total_samples - start_sample
        audio_to_place = audio[:available_space] if len(audio) > available_space else audio

        # Place audio
        end_sample = start_sample + len(audio_to_place)
        if start_sample < total_samples:
            combined[start_sample:end_sample] = audio_to_place
            last_end_sample = end_sample

    return combined


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    import ffmpeg

    probe = ffmpeg.probe(video_path)
    duration = float(probe["format"]["duration"])
    return duration


def mux_audio_video(video_path: str, audio_path: str, output_path: str) -> None:
    """Combine new audio with original video."""
    import ffmpeg

    video = ffmpeg.input(video_path)
    audio = ffmpeg.input(audio_path)

    try:
        (
            ffmpeg
            .output(
                video.video,
                audio.audio,
                output_path,
                vcodec="copy",
                acodec="aac",
                ar=44100,  # Standard sample rate for better compatibility
                ac=2,      # Stereo for QuickTime compatibility
                audio_bitrate="192k",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise click.ClickException(f"Failed to mux audio/video: {e.stderr.decode()}")


@click.command()
@click.option("--video", "-v", required=True, type=click.Path(exists=True),
              help="Input video file")
@click.option("--srt", "-s", required=True, type=click.Path(exists=True),
              help="Translated SRT file")
@click.option("--ref-start", "-rs", required=True, type=float,
              help="Start time (seconds) for reference audio")
@click.option("--ref-duration", "-rd", default=20.0, type=float,
              help="Duration (seconds) for reference audio")
@click.option("--ref-text", "-rt", required=True,
              help="Transcript of reference audio (original language)")
@click.option("--language", "-l", default="English",
              type=click.Choice(["English", "Chinese", "Japanese", "Korean",
                                "German", "French", "Russian", "Portuguese",
                                "Spanish", "Italian"]),
              help="Target language for dubbing")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output video file")
@click.option("--model", "-m", default=DEFAULT_MODEL,
              help="TTS model to use")
@click.option("--start-segment", type=int, default=1,
              help="Start from this segment number (for resuming)")
@click.option("--end-segment", type=int, default=None,
              help="End at this segment number (for testing)")
@click.option("--max-chunk-duration", type=float, default=30.0,
              help="Maximum duration of a chunk in seconds")
@click.option("--skip-segments", type=str, default=None,
              help="Segments to skip, e.g., '1-5,10,15-20' (different speakers, etc.)")
def dub_video(
    video: str,
    srt: str,
    ref_start: float,
    ref_duration: float,
    ref_text: str,
    language: str,
    output: str,
    model: str,
    start_segment: int,
    end_segment: Optional[int],
    max_chunk_duration: float,
    skip_segments: Optional[str],
) -> None:
    """
    Dub a video with voice cloning.

    Groups subtitle segments into natural chunks for consistent voice quality.

    EXAMPLE:

        uv run dub_video.py \\
            --video jancovici.mp4 \\
            --srt jancovici_translated.srt \\
            --ref-start 38 --ref-duration 20 \\
            --ref-text "d'abord parce que vous rentrez de vacances..." \\
            --language English \\
            --output jancovici_english.mp4
    """
    from mlx_audio.tts.utils import load_model

    video_path = Path(video)
    srt_path = Path(srt)
    output_path = Path(output)

    click.echo(f"Video: {video_path.name}")
    click.echo(f"SRT: {srt_path.name}")
    click.echo(f"Target language: {language}")

    # Parse SRT
    click.echo("Parsing SRT file...")
    segments = parse_srt(str(srt_path))
    click.echo(f"Found {len(segments)} segments")

    # Filter segments by range
    if end_segment:
        segments = [s for s in segments if start_segment <= s.index <= end_segment]
    else:
        segments = [s for s in segments if s.index >= start_segment]

    # Skip specified segments (e.g., different speakers)
    if skip_segments:
        skip_set = parse_skip_segments(skip_segments)
        segments = [s for s in segments if s.index not in skip_set]
        click.echo(f"Skipping segments: {sorted(skip_set)}")

    click.echo(f"Processing {len(segments)} segments")

    # Group into chunks
    chunks = group_segments_into_chunks(segments, max_gap=2.0, max_duration=max_chunk_duration)
    click.echo(f"Grouped into {len(chunks)} chunks for natural speech")

    # Get video duration
    video_duration = get_video_duration(str(video_path))
    click.echo(f"Video duration: {video_duration:.1f}s")

    # Extract reference audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        ref_audio_path = tmp.name

    click.echo("Extracting reference audio...")
    extract_reference_audio(str(video_path), ref_audio_path, ref_start, ref_duration)

    # Load TTS model
    click.echo(f"Loading TTS model: {model}")
    tts_model = load_model(model)

    sample_rate = 24000

    # Generate audio for each chunk
    click.echo("Generating dubbed audio...")
    audio_clips = []

    for chunk in tqdm(chunks, desc="Dubbing chunks"):
        try:
            audio, sr = generate_chunk_audio(
                model=tts_model,
                text=chunk.text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                language=language,
                sample_rate=sample_rate,
            )
            sample_rate = sr
            audio_clips.append(audio)
        except Exception as e:
            click.echo(f"\nWarning: Failed chunk starting at {chunk.start:.1f}s: {e}")
            audio_clips.append(np.array([], dtype=np.float32))

    # Place audio in timeline
    click.echo("Combining audio...")
    combined_audio = place_audio_in_timeline(chunks, audio_clips, video_duration, sample_rate)

    # Save combined audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_output_path = tmp.name

    sf.write(audio_output_path, combined_audio, sample_rate)

    # Mux with video
    click.echo("Creating final video...")
    mux_audio_video(str(video_path), audio_output_path, str(output_path))

    # Cleanup
    os.unlink(ref_audio_path)
    os.unlink(audio_output_path)

    click.echo(f"Done! Output saved to: {output_path}")


if __name__ == "__main__":
    dub_video()
