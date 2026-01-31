#!/usr/bin/env python3
"""Cut long silences from dubbed videos."""
import click
from pathlib import Path

from .postprocess import cut_silences, get_silence_stats


@click.command()
@click.argument("video", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output video path (default: input_cut.mp4)")
@click.option("--max-silence", "-m", type=float, default=0.3,
              help="Maximum silence duration to keep (seconds)")
@click.option("--threshold", "-t", type=float, default=-40.0,
              help="Silence threshold in dB")
@click.option("--min-duration", "-d", type=float, default=0.5,
              help="Minimum silence duration to consider cutting")
@click.option("--stats-only", is_flag=True,
              help="Only show silence statistics, don't cut")
def main(
    video: str,
    output: str,
    max_silence: float,
    threshold: float,
    min_duration: float,
    stats_only: bool,
):
    """
    Cut long silences from a dubbed video.

    This improves the flow of dubbed videos by removing awkward gaps
    where the TTS audio doesn't fill the original timing.

    \b
    EXAMPLES:
        transdub-cut video.mp4
        transdub-cut video.mp4 --max-silence 0.5
        transdub-cut video.mp4 --stats-only
    """
    video_path = Path(video)

    if stats_only:
        click.echo(f"Analyzing: {video_path.name}")
        stats = get_silence_stats(video_path, threshold, min_duration)

        click.echo(f"\nDuration: {stats['duration']:.1f}s")
        click.echo(f"Silence regions: {stats['silence_count']}")
        click.echo(f"Total silence: {stats['total_silence']:.1f}s ({stats['silence_percent']:.1f}%)")

        if stats['silences']:
            click.echo(f"\nLongest silences:")
            sorted_silences = sorted(stats['silences'], key=lambda x: x[1] - x[0], reverse=True)
            for start, end in sorted_silences[:10]:
                click.echo(f"  {start:.1f}s - {end:.1f}s ({end - start:.1f}s)")
        return

    if output is None:
        output = video_path.parent / f"{video_path.stem}_cut{video_path.suffix}"

    click.echo(f"Input: {video_path.name}")
    click.echo(f"Output: {output}")
    click.echo(f"Max silence: {max_silence}s")

    # Get stats first
    stats = get_silence_stats(video_path, threshold, min_duration)
    click.echo(f"\nFound {stats['silence_count']} silence regions ({stats['total_silence']:.1f}s total)")

    # Cut silences
    result = cut_silences(
        video_path,
        Path(output),
        max_silence=max_silence,
        threshold_db=threshold,
        min_silence_duration=min_duration,
    )

    # Get new duration
    import ffmpeg
    probe = ffmpeg.probe(str(result))
    new_duration = float(probe["format"]["duration"])

    saved = stats['duration'] - new_duration
    click.echo(f"\nDone! Saved {saved:.1f}s ({saved / stats['duration'] * 100:.1f}%)")
    click.echo(f"Output: {result}")


if __name__ == "__main__":
    main()
