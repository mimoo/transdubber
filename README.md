# transdub

Translate and dub videos using AI voice cloning. Uses Qwen3-TTS for voice cloning on Apple Silicon (MLX) and Whisper for transcription.

## Quick Start

```bash
# Install
uv sync

# Dub a video (automatic transcription + translation + voice cloning)
uv run transdub video.mp4

# With improved translations (requires ANTHROPIC_API_KEY)
uv run transdub video.mp4 --improve

# Skip intro segments (different speaker)
uv run transdub video.mp4 --skip "1-5"

# Full example
uv run transdub video.mp4 --improve --skip "1-5" --language English
```

The pipeline automatically:
1. Transcribes the video to the original language
2. Translates to the target language
3. (Optional) Improves translations with Claude
4. Finds a good voice reference segment
5. Dubs the entire video with the cloned voice

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

## Installation

```bash
git clone <repo-url>
cd transdub
uv sync

# With translation improvement support
uv sync --extra improve
```

## CLI Options

```
transdub VIDEO [OPTIONS]

Options:
  -l, --language TEXT       Target language (default: English)
  -s, --source-language     Source language (auto-detect if not set)
  -o, --output PATH         Output video path
  -w, --whisper-model       tiny|base|small|medium|large (default: medium)
  --improve/--no-improve    Improve translations with Claude
  --skip TEXT               Segments to skip, e.g., "1-5,10"
  --ref-start FLOAT         Reference audio start time (auto-detect)
  --ref-duration FLOAT      Reference audio duration (default: 20s)
  --ref-text TEXT           Reference transcript (auto-detect)
  --chunk-duration FLOAT    Max TTS chunk duration (default: 12s)
  --dry-run                 Show what would be done
```

## Python API

```python
from transdub import TransdubConfig, TransdubPipeline

config = TransdubConfig(
    video_path="video.mp4",
    target_language="English",
    improve_translation=True,  # requires ANTHROPIC_API_KEY
    skip_segments=[1, 2, 3, 4, 5],
)

pipeline = TransdubPipeline(config)
output_path = pipeline.run()
```

## Standalone Scripts

For more control, use the individual scripts:

```bash
# Transcribe (keep original language)
uv run translate_video.py video.mp4 --task transcribe --model medium

# Translate to English
uv run translate_video.py video.mp4 --task translate

# Text-to-speech with preset voices
uv run tts_speak.py "Hello, world"
uv run tts_speak.py --list-voices

# Voice cloning
uv run voice_clone.py \
    --ref-audio speaker.mp4 \
    --ref-text "Words spoken in the reference" \
    --text "New text to speak" \
    --language English

# Dub with existing SRT
uv run dub_video.py \
    --video video.mp4 \
    --srt subtitles.srt \
    --ref-start 38 --ref-duration 20 \
    --ref-text "Reference transcript" \
    --language English

# Improve translations with Claude
uv run improve_translation.py \
    -f original.srt \
    -e translated.srt \
    -o improved.srt
```

## Justfile Commands

```bash
just dub video.mp4              # Full dubbing pipeline
just dub-improve video.mp4      # With translation improvement
just transcribe video.mp4       # Transcribe only
just translate video.mp4        # Translate only
just speak "Hello world"        # Quick TTS
just voices                     # List TTS voices
```

## Tips

- **Voice cloning quality**: Use 10-20 seconds of clear speech without background noise
- **Chunk duration**: Shorter chunks (8-12s) preserve voice better but may have more "cuts"
- **Skip segments**: Use `--skip` to exclude intro/outro or different speakers
- **Testing**: Use `--dry-run` to preview what would be processed

## Project Structure

```
transdub/
├── transdub/              # Library
│   ├── __init__.py
│   ├── cli.py             # CLI entry point
│   ├── config.py          # Configuration
│   ├── pipeline.py        # Main orchestration
│   ├── transcribe.py      # Whisper transcription
│   ├── tts.py             # Voice cloning
│   ├── dubbing.py         # Audio generation
│   ├── improve.py         # Claude translation improvement
│   └── srt_utils.py       # SRT parsing
├── transdub_cli.py        # Standalone CLI
├── translate_video.py     # Standalone transcription
├── tts_speak.py           # Standalone TTS
├── voice_clone.py         # Standalone voice cloning
├── dub_video.py           # Standalone dubbing
└── improve_translation.py # Standalone improvement
```

## License

MIT
