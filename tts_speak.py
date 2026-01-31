#!/usr/bin/env python3
"""
Text-to-Speech using Qwen3-TTS with MLX on Apple Silicon.

Uses the CustomVoice model with 9 premium built-in voices.
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Suppress noisy warnings from transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*model of type qwen3_tts.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

import click
import soundfile as sf


# Available voices in CustomVoice model
# Keys are user-friendly, values are the actual speaker IDs used by the model
VOICES = {
    "serena": "serena",      # Chinese female
    "ryan": "ryan",          # English male
    "aiden": "aiden",        # English male
    "vivian": "vivian",      # Chinese female
    "uncle_fu": "uncle_fu",  # Chinese male (older)
    "anna": "ono_anna",      # Japanese female
    "sohee": "sohee",        # Korean female
    "dylan": "dylan",        # Chinese male (Beijing dialect)
    "eric": "eric",          # Chinese male (Sichuan dialect)
}

VOICE_DESCRIPTIONS = {
    "serena": "苏瑶 Serena - Chinese female",
    "ryan": "甜茶 Ryan - English male",
    "aiden": "艾登 Aiden - English male",
    "vivian": "十三 Vivian - Chinese female",
    "uncle_fu": "福伯 Uncle Fu - Chinese male (older)",
    "anna": "小野杏 Ono Anna - Japanese female",
    "sohee": "素熙 Sohee - Korean female",
    "dylan": "晓东 Dylan - Beijing dialect male",
    "eric": "程川 Eric - Sichuan dialect male",
}

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"


def get_model(model_name: str):
    """Load the TTS model."""
    from mlx_audio.tts.utils import load_model
    click.echo(f"Loading model: {model_name}")
    click.echo("(First run will download ~2GB)")
    return load_model(model_name)


@click.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True),
              help="Read text from a file instead of command line")
@click.option("--voice", "-v", default="ryan",
              type=click.Choice(list(VOICES.keys()), case_sensitive=False),
              help="Voice to use (default: ryan)")
@click.option("--language", "-l", default="English",
              type=click.Choice(["English", "Chinese", "Japanese", "Korean",
                                "German", "French", "Russian", "Portuguese",
                                "Spanish", "Italian"]),
              help="Language for synthesis")
@click.option("--emotion", "-e", default=None,
              help="Emotion/style instruction (e.g., 'happy', 'sad', 'whisper')")
@click.option("--output", "-o", default="output.wav",
              type=click.Path(),
              help="Output audio file path")
@click.option("--model", "-m", default=DEFAULT_MODEL,
              help="Model to use")
@click.option("--list-voices", is_flag=True,
              help="List available voices and exit")
def tts_speak(text: Optional[str], file: Optional[str], voice: str,
              language: str, emotion: Optional[str], output: str,
              model: str, list_voices: bool) -> None:
    """
    Convert text to speech using Qwen3-TTS.

    EXAMPLES:

        uv run tts_speak.py "Hello, how are you today?"

        uv run tts_speak.py --voice serena --language Chinese "你好世界"

        uv run tts_speak.py -f article.txt -o article_audio.wav

        uv run tts_speak.py "I'm so excited!" --emotion "very happy and energetic"
    """
    if list_voices:
        click.echo("Available voices:")
        for key, desc in VOICE_DESCRIPTIONS.items():
            click.echo(f"  {key:12} - {desc}")
        return

    # Get text from file or argument
    if file:
        text = Path(file).read_text(encoding="utf-8").strip()
    elif text is None:
        # Read from stdin if no text provided
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            raise click.ClickException(
                "No text provided. Use: tts_speak.py 'your text' or --file FILE"
            )

    if not text:
        raise click.ClickException("Text is empty")

    click.echo(f"Voice: {VOICE_DESCRIPTIONS[voice]}")
    click.echo(f"Language: {language}")
    if emotion:
        click.echo(f"Style: {emotion}")
    click.echo(f"Text length: {len(text)} characters")

    # Load model
    tts_model = get_model(model)

    # Generate audio
    click.echo("Generating speech...")

    # Build instruction if emotion is provided
    instruct = emotion if emotion else None

    results = list(tts_model.generate_custom_voice(
        text=text,
        speaker=VOICES[voice],
        language=language,
        instruct=instruct,
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


if __name__ == "__main__":
    tts_speak()
