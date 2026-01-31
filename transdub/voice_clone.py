#!/usr/bin/env python3
"""
Voice cloning using Qwen3-TTS with MLX on Apple Silicon.

Clone a voice from a reference audio sample (~3 seconds is enough).
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

# Suppress noisy warnings from transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*model of type qwen3_tts.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

import click
import soundfile as sf


DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"


def extract_audio_segment(
    input_path: str,
    output_path: str,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> None:
    """Extract audio segment from video/audio file."""
    import ffmpeg

    input_stream = ffmpeg.input(input_path)

    if start_time is not None:
        input_stream = ffmpeg.input(input_path, ss=start_time)

    output_kwargs = {
        "acodec": "pcm_s16le",
        "ac": 1,
        "ar": "24000",  # Qwen3-TTS uses 24kHz
    }

    if duration is not None:
        output_kwargs["t"] = duration

    try:
        (
            input_stream
            .output(output_path, **output_kwargs)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise click.ClickException(f"Failed to extract audio: {e.stderr.decode()}")


def get_model(model_name: str):
    """Load the TTS model."""
    from mlx_audio.tts.utils import load_model
    click.echo(f"Loading model: {model_name}")
    click.echo("(First run will download ~3GB)")
    return load_model(model_name)


@click.command()
@click.option("--ref-audio", "-r", required=True,
              type=click.Path(exists=True),
              help="Reference audio/video file to clone voice from")
@click.option("--ref-text", "-rt", required=True,
              help="Transcript of what is said in the reference audio")
@click.option("--text", "-t", required=False,
              help="Text to synthesize with cloned voice")
@click.option("--file", "-f", type=click.Path(exists=True),
              help="Read synthesis text from a file")
@click.option("--start", "-s", type=float, default=None,
              help="Start time in seconds for reference audio extraction")
@click.option("--duration", "-d", type=float, default=10.0,
              help="Duration in seconds for reference audio (default: 10)")
@click.option("--language", "-l", default="English",
              type=click.Choice(["English", "Chinese", "Japanese", "Korean",
                                "German", "French", "Russian", "Portuguese",
                                "Spanish", "Italian"]),
              help="Language for synthesis")
@click.option("--output", "-o", default="cloned_voice.wav",
              type=click.Path(),
              help="Output audio file path")
@click.option("--model", "-m", default=DEFAULT_MODEL,
              help="Model to use")
def voice_clone(ref_audio: str, ref_text: str, text: Optional[str],
                file: Optional[str], start: Optional[float], duration: float,
                language: str, output: str, model: str) -> None:
    """
    Clone a voice from reference audio and synthesize new speech.

    Requires:
    - Reference audio (~3-10 seconds of clear speech)
    - Transcript of what is said in the reference

    EXAMPLES:

        # Clone from a video file
        uv run voice_clone.py \\
            --ref-audio jancovici.mp4 \\
            --ref-text "The exact words spoken in the reference clip" \\
            --text "New text to speak with the cloned voice"

        # Clone from specific segment (start at 30s, use 5s)
        uv run voice_clone.py \\
            -r podcast.mp3 -s 30 -d 5 \\
            --ref-text "Words from that segment" \\
            --text "Hello, this is my cloned voice!"

        # Read text from file
        uv run voice_clone.py \\
            -r reference.wav \\
            --ref-text "Reference transcript" \\
            --file article.txt \\
            -o article_spoken.wav
    """
    # Get text to synthesize
    if file:
        text = Path(file).read_text(encoding="utf-8").strip()
    elif text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            raise click.ClickException(
                "No text provided. Use --text 'your text' or --file FILE"
            )

    if not text:
        raise click.ClickException("Text to synthesize is empty")

    ref_audio_path = Path(ref_audio)
    click.echo(f"Reference audio: {ref_audio_path.name}")
    click.echo(f"Reference text: {ref_text[:50]}{'...' if len(ref_text) > 50 else ''}")
    click.echo(f"Language: {language}")
    click.echo(f"Text to synthesize: {len(text)} characters")

    # Extract reference audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_ref_path = tmp.name

    try:
        click.echo("Extracting reference audio...")
        extract_audio_segment(
            str(ref_audio_path),
            tmp_ref_path,
            start_time=start,
            duration=duration,
        )

        # Load model
        tts_model = get_model(model)

        # Generate with cloned voice
        click.echo("Generating speech with cloned voice...")

        results = list(tts_model.generate(
            text=text,
            ref_audio=tmp_ref_path,
            ref_text=ref_text,
            language=language,
        ))

        if not results:
            raise click.ClickException("No audio generated")

        # Get audio and sample rate
        audio = results[0].audio
        sample_rate = results[0].sample_rate if hasattr(results[0], 'sample_rate') else 24000

        # Convert MLX array to numpy for soundfile
        import mlx.core as mx
        if isinstance(audio, mx.array):
            audio = audio.tolist()

        # Save audio
        output_path = Path(output)
        sf.write(str(output_path), audio, sample_rate)
        click.echo(f"Saved: {output_path}")
        click.echo("Done!")

    finally:
        # Cleanup temp file
        import os
        if os.path.exists(tmp_ref_path):
            os.unlink(tmp_ref_path)


if __name__ == "__main__":
    voice_clone()
