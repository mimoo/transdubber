#!/usr/bin/env python3
"""
Improve English translations using Claude, with French original as reference.

Takes French SRT (original) and English SRT (machine translation),
and produces improved English SRT that is more natural and concise.
"""
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class SRTSegment:
    index: int
    start: str  # Keep as string for output
    end: str
    text: str


def parse_srt(srt_path: str) -> list[SRTSegment]:
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

            match = re.match(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", timestamps)
            if not match:
                continue

            start = match.group(1)
            end = match.group(2)

            segments.append(SRTSegment(index, start, end, text))
        except (ValueError, IndexError):
            continue

    return segments


def write_srt(segments: list[SRTSegment], output_path: str) -> None:
    """Write segments to SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg.index}\n")
            f.write(f"{seg.start} --> {seg.end}\n")
            f.write(f"{seg.text}\n\n")


def improve_translations_batch(
    client,
    french_segments: list[SRTSegment],
    english_segments: list[SRTSegment],
    model: str = "claude-sonnet-4-20250514",
) -> list[str]:
    """Improve a batch of translations using Claude."""

    # Build the prompt with paired segments
    pairs = []
    for fr, en in zip(french_segments, english_segments):
        pairs.append(f"[{fr.index}]\nFrench: {fr.text}\nCurrent English: {en.text}")

    pairs_text = "\n\n".join(pairs)

    prompt = f"""You are improving English subtitle translations. For each segment, you have:
- The original French text
- A machine-translated English version

Your task:
1. Improve the English translation to be more natural and idiomatic
2. Keep translations CONCISE - subtitles should be easy to read quickly
3. Preserve the original meaning and tone
4. Do NOT make translations longer than necessary
5. Fix any obvious translation errors

IMPORTANT: Output ONLY the improved English text for each segment, in this exact format:
[segment_number]
Improved English text here

Do not include the French or any explanations.

Here are the segments to improve:

{pairs_text}"""

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response
    response_text = response.content[0].text
    improved = {}

    # Parse [number]\ntext format
    pattern = r'\[(\d+)\]\s*\n([^\[]+?)(?=\n\[|\Z)'
    matches = re.findall(pattern, response_text, re.DOTALL)

    for idx_str, text in matches:
        improved[int(idx_str)] = text.strip()

    # Return in order, falling back to original if not found
    results = []
    for en in english_segments:
        if en.index in improved:
            results.append(improved[en.index])
        else:
            results.append(en.text)

    return results


@click.command()
@click.option("--french-srt", "-f", required=True, type=click.Path(exists=True),
              help="Original French SRT file")
@click.option("--english-srt", "-e", required=True, type=click.Path(exists=True),
              help="Machine-translated English SRT file")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output improved English SRT file")
@click.option("--batch-size", "-b", default=20, type=int,
              help="Number of segments to process per API call")
@click.option("--model", "-m", default="claude-sonnet-4-20250514",
              help="Claude model to use")
@click.option("--start-segment", type=int, default=1,
              help="Start from this segment number")
@click.option("--end-segment", type=int, default=None,
              help="End at this segment number")
@click.option("--dry-run", is_flag=True,
              help="Show what would be done without calling API")
def improve_translation(
    french_srt: str,
    english_srt: str,
    output: str,
    batch_size: int,
    model: str,
    start_segment: int,
    end_segment: Optional[int],
    dry_run: bool,
) -> None:
    """
    Improve English subtitle translations using Claude.

    Requires ANTHROPIC_API_KEY environment variable.

    EXAMPLE:

        # First, transcribe to French
        uv run translate_video.py video.mp4 --task transcribe --language fr

        # Then improve the English translation
        uv run improve_translation.py \\
            --french-srt video_transcribed.srt \\
            --english-srt video_translated.srt \\
            --output video_improved.srt
    """
    if anthropic is None:
        raise click.ClickException(
            "anthropic package not installed. Run: uv add anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not dry_run:
        raise click.ClickException(
            "ANTHROPIC_API_KEY environment variable not set"
        )

    # Parse both SRT files
    click.echo(f"Loading French SRT: {french_srt}")
    french_segments = parse_srt(french_srt)
    click.echo(f"  Found {len(french_segments)} segments")

    click.echo(f"Loading English SRT: {english_srt}")
    english_segments = parse_srt(english_srt)
    click.echo(f"  Found {len(english_segments)} segments")

    # Create lookup by index
    french_by_idx = {s.index: s for s in french_segments}
    english_by_idx = {s.index: s for s in english_segments}

    # Find common indices
    common_indices = sorted(set(french_by_idx.keys()) & set(english_by_idx.keys()))

    # Filter by range
    if end_segment:
        common_indices = [i for i in common_indices if start_segment <= i <= end_segment]
    else:
        common_indices = [i for i in common_indices if i >= start_segment]

    click.echo(f"Processing {len(common_indices)} segments")

    if dry_run:
        click.echo("\n[DRY RUN] Would process these segments:")
        for i in common_indices[:10]:
            click.echo(f"  [{i}] FR: {french_by_idx[i].text[:50]}...")
            click.echo(f"       EN: {english_by_idx[i].text[:50]}...")
        if len(common_indices) > 10:
            click.echo(f"  ... and {len(common_indices) - 10} more")
        return

    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)

    # Process in batches
    improved_texts = {}

    from tqdm import tqdm

    for i in tqdm(range(0, len(common_indices), batch_size), desc="Improving translations"):
        batch_indices = common_indices[i:i + batch_size]

        french_batch = [french_by_idx[idx] for idx in batch_indices]
        english_batch = [english_by_idx[idx] for idx in batch_indices]

        try:
            improved = improve_translations_batch(client, french_batch, english_batch, model)
            for idx, text in zip(batch_indices, improved):
                improved_texts[idx] = text
        except Exception as e:
            click.echo(f"\nWarning: Batch failed: {e}")
            # Keep original translations for failed batch
            for idx in batch_indices:
                improved_texts[idx] = english_by_idx[idx].text

    # Build output segments (keep all original English segments, update improved ones)
    output_segments = []
    for seg in english_segments:
        if seg.index in improved_texts:
            output_segments.append(SRTSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                text=improved_texts[seg.index],
            ))
        else:
            output_segments.append(seg)

    # Write output
    write_srt(output_segments, output)
    click.echo(f"Saved improved SRT: {output}")
    click.echo("Done!")


if __name__ == "__main__":
    improve_translation()
