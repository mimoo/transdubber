"""Translation improvement using Claude."""
import os
import re
from pathlib import Path

from tqdm import tqdm

from .srt_utils import SRTSegment, parse_srt, write_srt


def improve_translations(
    french_srt: Path,
    english_srt: Path,
    output_path: Path,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    batch_size: int = 20,
    progress_callback=None,
) -> None:
    """
    Improve English translations using Claude.

    Args:
        french_srt: Path to original French SRT
        english_srt: Path to machine-translated English SRT
        output_path: Path to output improved SRT
        api_key: Anthropic API key (uses env var if None)
        model: Claude model to use
        batch_size: Segments per API call
        progress_callback: Optional callback(current, total, text)
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: uv add anthropic")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    # Parse SRTs
    french_segments = parse_srt(french_srt)
    english_segments = parse_srt(english_srt)

    french_by_idx = {s.index: s for s in french_segments}
    english_by_idx = {s.index: s for s in english_segments}

    common_indices = sorted(set(french_by_idx.keys()) & set(english_by_idx.keys()))

    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)

    # Process in batches
    improved = {}

    iterator = range(0, len(common_indices), batch_size)
    if progress_callback is None:
        iterator = tqdm(iterator, desc="Improving translations")

    for i in iterator:
        batch_indices = common_indices[i:i + batch_size]

        if progress_callback:
            progress_callback(i // batch_size, len(common_indices) // batch_size + 1)

        french_batch = [french_by_idx[idx] for idx in batch_indices]
        english_batch = [english_by_idx[idx] for idx in batch_indices]

        try:
            results = _improve_batch(client, french_batch, english_batch, model)
            for idx, text in zip(batch_indices, results):
                improved[idx] = text
        except Exception as e:
            print(f"Warning: Batch failed: {e}")
            for idx in batch_indices:
                improved[idx] = english_by_idx[idx].text

    # Build output
    output_segments = []
    for seg in english_segments:
        new_text = improved.get(seg.index, seg.text)
        output_segments.append(SRTSegment(
            index=seg.index,
            start=seg.start,
            end=seg.end,
            start_str=seg.start_str,
            end_str=seg.end_str,
            text=new_text,
        ))

    write_srt(output_segments, output_path)


def _improve_batch(
    client,
    french_segments: list[SRTSegment],
    english_segments: list[SRTSegment],
    model: str,
) -> list[str]:
    """Improve a batch of translations."""
    pairs = []
    for fr, en in zip(french_segments, english_segments):
        pairs.append(f"[{fr.index}]\nFrench: {fr.text}\nCurrent English: {en.text}")

    prompt = f"""Improve these English subtitle translations. For each:
1. Make the English more natural and idiomatic
2. Keep translations CONCISE (subtitles should be quick to read)
3. Preserve meaning and tone
4. Fix translation errors

Output ONLY improved English in this format:
[segment_number]
Improved text

Segments:

{chr(10).join(pairs)}"""

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse response
    text = response.content[0].text
    improved = {}

    for match in re.finditer(r'\[(\d+)\]\s*\n([^\[]+?)(?=\n\[|\Z)', text, re.DOTALL):
        improved[int(match.group(1))] = match.group(2).strip()

    return [improved.get(en.index, en.text) for en in english_segments]
