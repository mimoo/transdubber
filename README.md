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

# Cut awkward silences from dubbed video
uv run transdub-cut dubbed_video.mp4
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

## Commands

After `uv sync`, these commands are available:

| Command | Description |
|---------|-------------|
| `transdub` | Full dubbing pipeline |
| `transdub-transcribe` | Transcribe/translate video |
| `transdub-tts` | Text-to-speech with preset voices |
| `transdub-clone` | Clone a voice from audio |
| `transdub-dub` | Dub video with existing SRT |
| `transdub-improve` | Improve translations with Claude |
| `transdub-cut` | Cut long silences from video |

### Main Pipeline

```bash
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

### Post-Processing

```bash
# Cut long silences to improve flow
transdub-cut dubbed_video.mp4

# Customize silence threshold
transdub-cut video.mp4 --max-silence 0.5

# Just analyze silences
transdub-cut video.mp4 --stats-only
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

# Post-process to cut silences
from transdub.postprocess import cut_silences
cut_silences(output_path, "video_final.mp4", max_silence=0.3)
```

## Justfile Commands

```bash
just dub video.mp4              # Full dubbing pipeline
just dub-improve video.mp4      # With translation improvement
just cut video.mp4              # Cut silences
just transcribe video.mp4       # Transcribe only
just translate video.mp4        # Translate only
just speak "Hello world"        # Quick TTS
just voices                     # List TTS voices
```

## Tips

- **Voice cloning quality**: Use 10-20 seconds of clear speech without background noise
- **Chunk duration**: Shorter chunks (8-12s) preserve voice better but may have more "cuts"
- **Skip segments**: Use `--skip` to exclude intro/outro or different speakers
- **Post-processing**: Use `transdub-cut` to remove awkward gaps in dubbed audio
- **Testing**: Use `--dry-run` to preview what would be processed

## Shell Function for Quick TTS

Add a `speak` function to your shell for quick text-to-speech from anywhere:

```bash
# Clone the repo
git clone https://github.com/your-username/transdub.git ~/transdub
cd ~/transdub && uv sync

# Add to ~/.zshrc (or ~/.bashrc)
speak() { (cd ~/transdub && uv run transdub-tts "$1" --voice "${2:-serena}" -o /tmp/speech.wav) && afplay /tmp/speech.wav; }
```

Then reload your shell (`source ~/.zshrc`) and use it:

```bash
speak "Hello world"              # Uses default voice (serena)
speak "Hello world" ryan         # Uses specific voice
```

### Available Voices

| Voice | Description |
|-------|-------------|
| `serena` | Chinese female |
| `vivian` | Chinese female |
| `anna` | Japanese female |
| `sohee` | Korean female |
| `ryan` | English male |
| `aiden` | English male |
| `dylan` | Beijing dialect male |
| `eric` | Sichuan dialect male |
| `uncle_fu` | Chinese male (older) |

## Using with AI Coding Agents

You can configure AI coding agents (Claude Code, Cursor, etc.) to announce task completion or request attention using the `speak` function. Add this to your `CLAUDE.md` or `AGENTS.md`:

```markdown
## Voice Notifications

When you complete a task or need user attention, use the `speak` shell function:

- Task complete: `speak "Done with the task" serena`
- Need follow-up: `speak "I have a question" serena`
- Error occurred: `speak "Something went wrong" serena`

Rules:
- Keep messages under 50 characters (TTS works best with short text)
- Use voice: serena (or assign different voices per agent/project)
- Only announce significant milestones, not every small step
```

### Multi-Agent Voice Assignment

For projects with multiple agents, assign different voices to distinguish them:

```markdown
## Agent Voices
- Main agent: serena
- Code review agent: vivian
- Test runner agent: anna
- Documentation agent: sohee
```

This way you'll know which agent is speaking without looking at the screen.

## Project Structure

```
transdub/
├── transdub/
│   ├── __init__.py        # Package exports
│   ├── cli.py             # Main CLI
│   ├── config.py          # Configuration
│   ├── pipeline.py        # Orchestration
│   ├── transcribe.py      # Whisper integration
│   ├── tts.py             # Voice cloning core
│   ├── dubbing.py         # Audio generation
│   ├── improve.py         # Claude improvement
│   ├── srt_utils.py       # SRT parsing
│   ├── postprocess.py     # Silence cutting
│   ├── translate_video.py # Transcription CLI
│   ├── tts_speak.py       # TTS CLI
│   ├── voice_clone.py     # Voice clone CLI
│   ├── dub_video.py       # Dubbing CLI
│   ├── improve_translation.py  # Improvement CLI
│   └── cut_silences.py    # Silence cutting CLI
├── pyproject.toml
├── justfile
└── README.md
```

## License

MIT
