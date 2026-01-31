"""SRT file utilities."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SRTSegment:
    """A single subtitle segment."""
    index: int
    start: float  # seconds
    end: float    # seconds
    start_str: str  # Original timestamp string
    end_str: str
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_timestamp(ts: str) -> float:
    """Convert SRT timestamp to seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def parse_srt(srt_path: Path) -> list[SRTSegment]:
    """Parse an SRT file into segments."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

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

            match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
                timestamps
            )
            if not match:
                continue

            start_str = match.group(1)
            end_str = match.group(2)

            if text:
                segments.append(SRTSegment(
                    index=index,
                    start=parse_timestamp(start_str),
                    end=parse_timestamp(end_str),
                    start_str=start_str,
                    end_str=end_str,
                    text=text,
                ))
        except (ValueError, IndexError):
            continue

    return segments


def write_srt(segments: list[SRTSegment], output_path: Path) -> None:
    """Write segments to SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg.index}\n")
            f.write(f"{seg.start_str} --> {seg.end_str}\n")
            f.write(f"{seg.text}\n\n")


def find_good_reference_segment(
    segments: list[SRTSegment],
    min_duration: float = 15.0,
    max_start: float = 120.0,
) -> tuple[float, float, str] | None:
    """
    Find a good segment for voice reference.

    Looks for a segment or group of consecutive segments that:
    - Has enough duration (15-30 seconds)
    - Is early in the video (first 2 minutes)
    - Has substantial text content

    Returns:
        (start_time, duration, text) or None if not found
    """
    # Find consecutive segments that form a good reference
    for i, seg in enumerate(segments):
        if seg.start > max_start:
            break

        # Try to build a reference from consecutive segments
        ref_text_parts = [seg.text]
        ref_end = seg.end
        ref_duration = seg.duration

        # Add consecutive segments
        for j in range(i + 1, len(segments)):
            next_seg = segments[j]
            gap = next_seg.start - ref_end

            # Stop if gap too large or duration sufficient
            if gap > 2.0 or ref_duration >= min_duration:
                break

            ref_text_parts.append(next_seg.text)
            ref_end = next_seg.end
            ref_duration = ref_end - seg.start

        if ref_duration >= min_duration:
            return (seg.start, ref_duration, " ".join(ref_text_parts))

    return None
