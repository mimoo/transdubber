# Transcribe a video (keep original language)
transcribe VIDEO MODEL="base":
    uv run translate_video.py {{VIDEO}} --task transcribe --model {{MODEL}} --output-format srt

# Translate a video to English
translate VIDEO MODEL="base":
    uv run translate_video.py {{VIDEO}} --task translate --model {{MODEL}} --output-format srt

# Transcribe with all output formats
transcribe-all VIDEO MODEL="base":
    uv run translate_video.py {{VIDEO}} --task transcribe --model {{MODEL}} --output-format all

# Translate with all output formats
translate-all VIDEO MODEL="base":
    uv run translate_video.py {{VIDEO}} --task translate --model {{MODEL}} --output-format all

# Text-to-speech with preset voice
speak TEXT VOICE="ryan":
    uv run tts_speak.py "{{TEXT}}" --voice {{VOICE}}

# TTS from a text file
speak-file FILE VOICE="ryan" OUTPUT="output.wav":
    uv run tts_speak.py --file {{FILE}} --voice {{VOICE}} --output {{OUTPUT}}

# List available TTS voices
voices:
    uv run tts_speak.py --list-voices

# Clone voice from video/audio and speak text
clone REF_AUDIO REF_TEXT TEXT:
    uv run voice_clone.py --ref-audio {{REF_AUDIO}} --ref-text "{{REF_TEXT}}" --text "{{TEXT}}"

# Clone voice from video segment (with start time and duration)
clone-segment REF_AUDIO REF_TEXT TEXT START="0" DURATION="10":
    uv run voice_clone.py --ref-audio {{REF_AUDIO}} --ref-text "{{REF_TEXT}}" --text "{{TEXT}}" --start {{START}} --duration {{DURATION}}

# === TRANSDUB (one-command dubbing) ===

# Full automatic dubbing pipeline
dub VIDEO *ARGS:
    uv run transdub_cli.py {{VIDEO}} {{ARGS}}

# Dub with improved translations (requires ANTHROPIC_API_KEY)
dub-improve VIDEO *ARGS:
    uv run transdub_cli.py {{VIDEO}} --improve {{ARGS}}

# Show what would be done without processing
dub-dry VIDEO *ARGS:
    uv run transdub_cli.py {{VIDEO}} --dry-run {{ARGS}}

# Show available recipes
help:
    @just --list
