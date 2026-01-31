#!/usr/bin/env python3
import os
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any
import click
import whisper
import ffmpeg


def extract_audio(video_path: str, output_path: str) -> None:
    """Extract audio from video and convert to 16kHz WAV format."""
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise click.ClickException(f"Failed to extract audio: {e.stderr.decode()}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')


def write_srt(segments: list, output_path: str) -> None:
    """Write segments to SRT format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def write_vtt(segments: list, output_path: str) -> None:
    """Write segments to WebVTT format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            start = format_timestamp(segment['start']).replace(',', '.')
            end = format_timestamp(segment['end']).replace(',', '.')
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def write_text(segments: list, output_path: str) -> None:
    """Write segments as plain text."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(f"{segment['text'].strip()}\n")


def write_json(result: Dict[str, Any], output_path: str) -> None:
    """Write full result as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--model', default='base',
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']),
              help='Whisper model size to use')
@click.option('--task', default='transcribe',
              type=click.Choice(['transcribe', 'translate']),
              help='Task to perform: transcribe (keep original language) or translate (to English)')
@click.option('--language', default=None,
              help='Source language code (e.g., fr, es, de). Auto-detect if not specified')
@click.option('--output-format', default='srt',
              type=click.Choice(['srt', 'vtt', 'text', 'json', 'all']),
              help='Output format for the transcription/translation')
@click.option('--output-dir', default=None,
              type=click.Path(file_okay=False, dir_okay=True),
              help='Output directory (default: same as video)')
@click.option('--device', default=None,
              type=click.Choice(['cpu', 'cuda']),
              help='Device to use for inference (auto-detect if not specified)')
@click.option('--verbose', is_flag=True,
              help='Show detailed progress information')
def translate_video(video_path: str, model: str, task: str, language: Optional[str],
                    output_format: str, output_dir: Optional[str], device: Optional[str],
                    verbose: bool) -> None:
    """
    Extract and translate/transcribe speech from video files using OpenAI Whisper.

    EXAMPLE:
        uv run translate_video.py video.mp4 --task translate --model medium
    """
    video_path = Path(video_path)

    # Set output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = video_path.parent

    # Base name for output files
    base_name = video_path.stem
    task_suffix = "translated" if task == "translate" else "transcribed"

    click.echo(f"Processing: {video_path.name}")
    click.echo(f"Task: {task}")
    click.echo(f"Model: {model}")

    # Extract audio to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        # Step 1: Extract audio
        click.echo("Extracting audio from video...")
        extract_audio(str(video_path), tmp_audio_path)

        # Step 2: Load Whisper model
        click.echo(f"Loading Whisper {model} model...")
        if device == 'cpu':
            whisper_model = whisper.load_model(model, device='cpu')
        else:
            whisper_model = whisper.load_model(model)

        # Step 3: Transcribe/Translate
        click.echo(f"{'Translating' if task == 'translate' else 'Transcribing'} audio...")

        transcribe_options = {
            'task': task,
            'verbose': verbose,
        }

        if language:
            transcribe_options['language'] = language

        result = whisper_model.transcribe(tmp_audio_path, **transcribe_options)

        # Display detected language if auto-detected
        if not language and 'language' in result:
            click.echo(f"Detected language: {result['language']}")

        # Step 4: Save output
        segments = result['segments']

        if output_format == 'all':
            formats = ['srt', 'vtt', 'text', 'json']
        else:
            formats = [output_format]

        for fmt in formats:
            output_file = output_dir / f"{base_name}_{task_suffix}.{fmt}"

            if fmt == 'srt':
                write_srt(segments, str(output_file))
            elif fmt == 'vtt':
                write_vtt(segments, str(output_file))
            elif fmt == 'text':
                write_text(segments, str(output_file))
            elif fmt == 'json':
                write_json(result, str(output_file))

            click.echo(f"Saved {fmt.upper()}: {output_file}")

        click.echo("✓ Processing complete!")

    finally:
        # Clean up temporary audio file
        if os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)


if __name__ == '__main__':
    translate_video()