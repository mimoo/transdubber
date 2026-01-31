"""Video dubbing utilities."""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import numpy as np
import soundfile as sf
from tqdm import tqdm

from .srt_utils import SRTSegment
from .tts import generate_speech


@dataclass
class Chunk:
    """A group of segments to synthesize together."""
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


def group_into_chunks(
    segments: list[SRTSegment],
    max_gap: float = 2.0,
    max_duration: float = 12.0,
) -> list[Chunk]:
    """Group consecutive segments into chunks."""
    if not segments:
        return []

    chunks = []
    current = [segments[0]]

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        gap = curr.start - prev.end
        chunk_duration = curr.end - current[0].start

        if gap > max_gap or chunk_duration > max_duration:
            chunks.append(Chunk(current))
            current = [curr]
        else:
            current.append(curr)

    if current:
        chunks.append(Chunk(current))

    return chunks


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    probe = ffmpeg.probe(str(video_path))
    return float(probe["format"]["duration"])


def create_dubbed_audio(
    chunks: list[Chunk],
    model,
    ref_audio_path: str,
    ref_text: str,
    language: str,
    total_duration: float,
    sample_rate: int = 24000,
    progress_callback=None,
) -> np.ndarray:
    """Generate dubbed audio track."""
    total_samples = int(total_duration * sample_rate)
    combined = np.zeros(total_samples, dtype=np.float32)
    last_end = 0

    iterator = tqdm(chunks, desc="Dubbing") if progress_callback is None else chunks

    for i, chunk in enumerate(iterator):
        if progress_callback:
            progress_callback(i, len(chunks), chunk.text[:50])

        try:
            audio, sr = generate_speech(
                model=model,
                text=chunk.text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                language=language,
                sample_rate=sample_rate,
            )
            sample_rate = sr

            if len(audio) == 0:
                continue

            # Place audio, avoiding overlap
            start_sample = max(int(chunk.start * sample_rate), last_end + int(0.1 * sample_rate))
            available = total_samples - start_sample
            audio_to_place = audio[:available] if len(audio) > available else audio

            end_sample = start_sample + len(audio_to_place)
            if start_sample < total_samples:
                combined[start_sample:end_sample] = audio_to_place
                last_end = end_sample

        except Exception as e:
            print(f"Warning: Failed chunk at {chunk.start:.1f}s: {e}")

    return combined


def mux_audio_video(
    video_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
) -> None:
    """Combine new audio with video."""
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    sf.write(audio_path, audio, sample_rate)

    try:
        video = ffmpeg.input(str(video_path))
        audio_input = ffmpeg.input(audio_path)

        (
            ffmpeg
            .output(
                video.video,
                audio_input.audio,
                str(output_path),
                vcodec="copy",
                acodec="aac",
                ar=44100,
                ac=2,
                audio_bitrate="192k",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    finally:
        os.unlink(audio_path)
