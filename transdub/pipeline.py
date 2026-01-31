"""Main pipeline orchestration for transdub."""
import os
from pathlib import Path
from typing import Callable

from .config import TransdubConfig
from .srt_utils import parse_srt, find_good_reference_segment
from .transcribe import transcribe_video
from .tts import load_tts_model, extract_reference_audio
from .dubbing import group_into_chunks, get_video_duration, create_dubbed_audio, mux_audio_video
from .improve import improve_translations


class TransdubPipeline:
    """
    Automated video translation and dubbing pipeline.

    Usage:
        config = TransdubConfig(video_path="video.mp4")
        pipeline = TransdubPipeline(config)
        pipeline.run()
    """

    def __init__(self, config: TransdubConfig, log: Callable[[str], None] | None = None):
        self.config = config
        self.log = log or print
        self._tts_model = None
        self._ref_audio_path = None

    def run(self) -> Path:
        """
        Run the full pipeline.

        Returns:
            Path to the output dubbed video
        """
        self.log(f"=== Transdub Pipeline ===")
        self.log(f"Video: {self.config.video_path}")
        self.log(f"Target language: {self.config.target_language}")

        # Step 1: Transcribe to original language
        self._step_transcribe()

        # Step 2: Translate to target language
        self._step_translate()

        # Step 3: Optionally improve translations
        if self.config.improve_translation:
            self._step_improve()

        # Step 4: Find/validate voice reference
        self._step_prepare_reference()

        # Step 5: Dub the video
        self._step_dub()

        # Cleanup
        self._cleanup()

        self.log(f"\n=== Done! ===")
        self.log(f"Output: {self.config.output_video}")

        return self.config.output_video

    def _step_transcribe(self):
        """Transcribe video to original language."""
        if self.config.transcribed_srt.exists():
            self.log(f"\n[1/5] Transcription exists: {self.config.transcribed_srt}")
            return

        self.log(f"\n[1/5] Transcribing to {self.config.source_language or 'auto-detect'}...")

        result = transcribe_video(
            video_path=self.config.video_path,
            output_path=self.config.transcribed_srt,
            model_name=self.config.whisper_model,
            task="transcribe",
            language=self.config.source_language,
        )

        detected_lang = result.get("language", "unknown")
        self.log(f"    Detected language: {detected_lang}")
        self.log(f"    Saved: {self.config.transcribed_srt}")

    def _step_translate(self):
        """Translate to target language."""
        if self.config.translated_srt.exists():
            self.log(f"\n[2/5] Translation exists: {self.config.translated_srt}")
            return

        self.log(f"\n[2/5] Translating to {self.config.target_language}...")

        transcribe_video(
            video_path=self.config.video_path,
            output_path=self.config.translated_srt,
            model_name=self.config.whisper_model,
            task="translate",
            language=self.config.source_language,
        )

        self.log(f"    Saved: {self.config.translated_srt}")

    def _step_improve(self):
        """Improve translations with Claude."""
        if self.config.improved_srt.exists():
            self.log(f"\n[3/5] Improved translation exists: {self.config.improved_srt}")
            return

        api_key = self.config.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self.log(f"\n[3/5] Skipping improvement (no ANTHROPIC_API_KEY)")
            return

        self.log(f"\n[3/5] Improving translations with Claude...")

        improve_translations(
            french_srt=self.config.transcribed_srt,
            english_srt=self.config.translated_srt,
            output_path=self.config.improved_srt,
            api_key=api_key,
        )

        self.log(f"    Saved: {self.config.improved_srt}")

    def _step_prepare_reference(self):
        """Prepare voice reference audio."""
        self.log(f"\n[4/5] Preparing voice reference...")

        # Auto-detect reference if not provided
        if self.config.ref_start is None or self.config.ref_text is None:
            self.log("    Auto-detecting reference segment...")

            segments = parse_srt(self.config.transcribed_srt)

            # Skip specified segments
            if self.config.skip_segments:
                skip_set = set(self.config.skip_segments)
                segments = [s for s in segments if s.index not in skip_set]

            result = find_good_reference_segment(segments)
            if result is None:
                raise ValueError("Could not find suitable reference segment. Please provide --ref-start and --ref-text")

            ref_start, ref_duration, ref_text = result
            self.config.ref_start = ref_start
            self.config.ref_duration = ref_duration
            self.config.ref_text = ref_text

            self.log(f"    Found reference at {ref_start:.1f}s ({ref_duration:.1f}s)")
            self.log(f"    Text: {ref_text[:80]}...")

        # Extract reference audio
        self._ref_audio_path = extract_reference_audio(
            self.config.video_path,
            self.config.ref_start,
            self.config.ref_duration,
        )
        self.log(f"    Extracted reference audio")

    def _step_dub(self):
        """Dub the video with cloned voice."""
        self.log(f"\n[5/5] Dubbing video...")

        # Determine which SRT to use
        if self.config.improve_translation and self.config.improved_srt.exists():
            srt_path = self.config.improved_srt
        else:
            srt_path = self.config.translated_srt

        self.log(f"    Using subtitles: {srt_path.name}")

        # Parse and filter segments
        segments = parse_srt(srt_path)
        if self.config.skip_segments:
            skip_set = set(self.config.skip_segments)
            segments = [s for s in segments if s.index not in skip_set]
            self.log(f"    Skipping {len(skip_set)} segments")

        self.log(f"    Processing {len(segments)} segments")

        # Group into chunks
        chunks = group_into_chunks(segments, max_duration=self.config.max_chunk_duration)
        self.log(f"    Grouped into {len(chunks)} chunks")

        # Load TTS model
        self.log(f"    Loading TTS model...")
        self._tts_model = load_tts_model(self.config.tts_model)

        # Get video duration
        video_duration = get_video_duration(self.config.video_path)

        # Generate dubbed audio
        self.log(f"    Generating dubbed audio...")
        audio = create_dubbed_audio(
            chunks=chunks,
            model=self._tts_model,
            ref_audio_path=self._ref_audio_path,
            ref_text=self.config.ref_text,
            language=self.config.target_language,
            total_duration=video_duration,
        )

        # Mux with video
        self.log(f"    Creating final video...")
        mux_audio_video(
            self.config.video_path,
            audio,
            sample_rate=24000,
            output_path=self.config.output_video,
        )

    def _cleanup(self):
        """Clean up temporary files."""
        if self._ref_audio_path and os.path.exists(self._ref_audio_path):
            os.unlink(self._ref_audio_path)
