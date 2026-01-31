"""Text-to-speech and voice cloning utilities."""
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

# Suppress noisy warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*model of type qwen3_tts.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")


def load_tts_model(model_name: str):
    """Load TTS model."""
    from mlx_audio.tts.utils import load_model
    return load_model(model_name)


def extract_reference_audio(
    video_path: Path,
    start: float,
    duration: float,
    sample_rate: int = 24000,
) -> str:
    """Extract reference audio segment, return path to temp file."""
    import ffmpeg

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    (
        ffmpeg
        .input(str(video_path), ss=start)
        .output(tmp.name, acodec="pcm_s16le", ac=1, ar=sample_rate, t=duration)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    return tmp.name


def generate_speech(
    model,
    text: str,
    ref_audio_path: str,
    ref_text: str,
    language: str = "English",
    sample_rate: int = 24000,
) -> tuple[np.ndarray, int]:
    """Generate speech with cloned voice."""
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

    if isinstance(audio, mx.array):
        audio = np.array(audio.tolist(), dtype=np.float32)
    else:
        audio = np.array(audio, dtype=np.float32)

    return audio, sample_rate
