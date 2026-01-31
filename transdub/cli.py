"""Command-line interface for transdub."""
import sys
from pathlib import Path
from typing import Optional

import click

from .config import TransdubConfig
from .pipeline import TransdubPipeline


def parse_skip_segments(skip_str: str) -> list[int]:
    """Parse skip string like '1-5,10,15-20' into list of ints."""
    result = []
    if not skip_str:
        return result

    for part in skip_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))

    return result


@click.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--language", "-l", default="English",
              help="Target language for dubbing")
@click.option("--source-language", "-s", default=None,
              help="Source language (auto-detect if not specified)")
@click.option("--output", "-o", default=None,
              type=click.Path(),
              help="Output video path")
@click.option("--whisper-model", "-w", default="medium",
              type=click.Choice(["tiny", "base", "small", "medium", "large"]),
              help="Whisper model for transcription")
@click.option("--improve/--no-improve", default=False,
              help="Improve translations with Claude (requires ANTHROPIC_API_KEY)")
@click.option("--skip", default=None,
              help="Segments to skip, e.g., '1-5,10' (different speakers)")
@click.option("--ref-start", type=float, default=None,
              help="Reference audio start time (auto-detect if not set)")
@click.option("--ref-duration", type=float, default=20.0,
              help="Reference audio duration in seconds")
@click.option("--ref-text", default=None,
              help="Reference audio transcript (auto-detect if not set)")
@click.option("--chunk-duration", type=float, default=12.0,
              help="Max TTS chunk duration (shorter = more faithful voice)")
@click.option("--dry-run", is_flag=True,
              help="Show what would be done without processing")
def main(
    video: str,
    language: str,
    source_language: Optional[str],
    output: Optional[str],
    whisper_model: str,
    improve: bool,
    skip: Optional[str],
    ref_start: Optional[float],
    ref_duration: float,
    ref_text: Optional[str],
    chunk_duration: float,
    dry_run: bool,
):
    """
    Translate and dub a video with AI voice cloning.

    Automatically transcribes, translates, and dubs the video using the
    speaker's cloned voice.

    \b
    EXAMPLES:
        transdub video.mp4
        transdub video.mp4 --language French
        transdub video.mp4 --improve --skip "1-5"
    """
    video_path = Path(video)

    # Parse skip segments
    skip_segments = parse_skip_segments(skip) if skip else []

    # Build config
    config = TransdubConfig(
        video_path=video_path,
        target_language=language,
        source_language=source_language,
        whisper_model=whisper_model,
        improve_translation=improve,
        skip_segments=skip_segments,
        ref_start=ref_start,
        ref_duration=ref_duration,
        ref_text=ref_text,
        max_chunk_duration=chunk_duration,
    )

    if output:
        config.output_dir = Path(output).parent

    if dry_run:
        click.echo("=== Dry Run ===")
        click.echo(f"Video: {config.video_path}")
        click.echo(f"Target language: {config.target_language}")
        click.echo(f"Whisper model: {config.whisper_model}")
        click.echo(f"Improve translation: {config.improve_translation}")
        click.echo(f"Skip segments: {config.skip_segments}")
        click.echo(f"Max chunk duration: {config.max_chunk_duration}s")
        click.echo(f"\nOutputs:")
        click.echo(f"  Transcribed: {config.transcribed_srt}")
        click.echo(f"  Translated: {config.translated_srt}")
        if config.improve_translation:
            click.echo(f"  Improved: {config.improved_srt}")
        click.echo(f"  Video: {config.output_video}")
        return

    # Run pipeline
    pipeline = TransdubPipeline(config)

    try:
        output_path = pipeline.run()
        click.echo(f"\nSuccess! Output: {output_path}")
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
