# === MAIN PIPELINE ===

# Full automatic dubbing pipeline
dub VIDEO *ARGS:
    uv run transdub {{VIDEO}} {{ARGS}}

# Dub with improved translations (requires ANTHROPIC_API_KEY)
dub-improve VIDEO *ARGS:
    uv run transdub {{VIDEO}} --improve {{ARGS}}

# Show what would be done without processing
dub-dry VIDEO *ARGS:
    uv run transdub {{VIDEO}} --dry-run {{ARGS}}

# === POST-PROCESSING ===

# Cut long silences from a video
cut VIDEO *ARGS:
    uv run transdub-cut {{VIDEO}} {{ARGS}}

# Show silence statistics
silence-stats VIDEO:
    uv run transdub-cut {{VIDEO}} --stats-only

# === TRANSCRIPTION ===

# Transcribe a video (keep original language)
transcribe VIDEO MODEL="medium":
    uv run transdub-transcribe {{VIDEO}} --task transcribe --model {{MODEL}}

# Translate a video to English
translate VIDEO MODEL="medium":
    uv run transdub-transcribe {{VIDEO}} --task translate --model {{MODEL}}

# === TEXT-TO-SPEECH ===

# Text-to-speech with preset voice
speak TEXT VOICE="ryan":
    uv run transdub-tts "{{TEXT}}" --voice {{VOICE}}

# List available TTS voices
voices:
    uv run transdub-tts --list-voices

# === VOICE CLONING ===

# Clone voice from video/audio and speak text
clone REF_AUDIO REF_TEXT TEXT:
    uv run transdub-clone --ref-audio {{REF_AUDIO}} --ref-text "{{REF_TEXT}}" --text "{{TEXT}}"

# Clone voice from video segment (with start time and duration)
clone-segment REF_AUDIO REF_TEXT TEXT START="0" DURATION="10":
    uv run transdub-clone --ref-audio {{REF_AUDIO}} --ref-text "{{REF_TEXT}}" --text "{{TEXT}}" --start {{START}} --duration {{DURATION}}

# === HELP ===

# Show available recipes
help:
    @just --list
