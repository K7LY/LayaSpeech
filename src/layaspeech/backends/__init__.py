"""Optional synthesizer backends."""

from .voicevox import VoicevoxSynthesizer, apply_voicevox_controls

__all__ = ["VoicevoxSynthesizer", "apply_voicevox_controls"]
