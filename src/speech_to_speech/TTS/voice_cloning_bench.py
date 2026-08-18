"""
Voice cloning benchmarking and testing module for Qwen3-TTS Base and other TTS providers.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from speech_to_speech.backend_registry import LLM_BACKENDS, STT_BACKENDS, TTS_BACKENDS
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

logger = logging.getLogger(__name__)

OFFICIAL_QWEN3_LANGUAGES = {
    "auto": "Auto Detect",
    "en": "English (Official)",
    "zh": "Chinese (Official)",
    "ja": "Japanese (Official)",
    "ko": "Korean (Official)",
    "de": "German (Official)",
    "fr": "French (Official)",
    "ru": "Russian (Official)",
    "pt": "Portuguese (Official)",
    "es": "Spanish (Official)",
    "it": "Italian (Official)",
}

EXPERIMENTAL_LANGUAGES = {
    "km": "Khmer (Experimental / Unsupported)",
    "vi": "Vietnamese (Experimental)",
    "th": "Thai (Experimental)",
    "id": "Indonesian (Experimental)",
    "ar": "Arabic (Experimental)",
    "hi": "Hindi (Experimental)",
}

ALL_TEST_LANGUAGES = {**OFFICIAL_QWEN3_LANGUAGES, **EXPERIMENTAL_LANGUAGES}


@dataclass
class VoiceCloningBenchmarkResult:
    model_name: str
    reference_audio_name: str
    reference_duration_seconds: float
    language: str
    is_official_language: bool
    cloning_mode: str  # "x_vector_only" or "reference_transcript"
    input_text: str
    generation_time_ms: float
    generated_audio_duration_seconds: float
    rtf: float
    sample_rate: int
    gpu_memory_allocated_mb: float
    gpu_memory_reserved_mb: float
    gpu_name: Optional[str] = None
    cuda_available: bool = False
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VoiceCloningService:
    """
    Singleton-friendly service that lazily loads and keeps Qwen3-TTS Base model
    in memory for repeated benchmark tests without per-request reload latency.
    """

    _instance: Optional[VoiceCloningService] = None
    _lock: Lock = Lock()

    def __init__(self, default_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        self.default_model = default_model
        self.active_handler: Optional[Qwen3TTSHandler] = None
        self.current_model_name: Optional[str] = None
        self.current_device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._handler_lock = Lock()
        self.sample_rate = 16000

    @classmethod
    def get_instance(cls, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base") -> VoiceCloningService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(default_model=model_name)
            return cls._instance

    def get_gpu_stats(self) -> Dict[str, Any]:
        cuda_avail = torch.cuda.is_available()
        stats = {
            "cuda_available": cuda_avail,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else None,
            "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2) if cuda_avail else 0.0,
            "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2) if cuda_avail else 0.0,
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2) if cuda_avail else 0.0,
        }
        return stats

    def get_providers_info(self) -> Dict[str, Any]:
        return {
            "stt_providers": list(STT_BACKENDS.keys()),
            "llm_providers": list(LLM_BACKENDS.keys()),
            "tts_providers": list(TTS_BACKENDS.keys()),
            "default_qwen_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "supported_languages": [
                {"code": code, "name": name, "is_official": code in OFFICIAL_QWEN3_LANGUAGES}
                for code, name in ALL_TEST_LANGUAGES.items()
            ],
            "gpu_info": self.get_gpu_stats(),
        }

    def validate_reference_audio(self, audio_path: str | Path) -> Tuple[bool, float, str]:
        path = Path(audio_path)
        if not path.is_file():
            return False, 0.0, f"Reference audio file not found: {path}"

        try:
            info = sf.info(str(path))
            duration = info.duration
            if duration < 0.2:
                return False, duration, "Reference audio is too short (less than 0.2s)."
            if duration > 120.0:
                logger.warning("Reference audio is very long (%0.1fs); recommend <= 30s.", duration)
            return True, duration, ""
        except Exception as e:
            return False, 0.0, f"Invalid audio file format: {e}"

    def ensure_model_loaded(
        self,
        model_name: str,
        ref_audio_path: Optional[str | Path] = None,
        ref_text: Optional[str] = None,
        xvec_only: bool = True,
        language: str = "auto",
        device: Optional[str] = None,
    ) -> Qwen3TTSHandler:
        target_device = device or self.current_device
        with self._handler_lock:
            # Check if existing handler matches model and device
            if (
                self.active_handler is not None
                and self.current_model_name == model_name
                and getattr(self.active_handler, "requested_device", None) == target_device
            ):
                # Update runtime fields on existing handler
                self.active_handler.ref_audio = ref_audio_path
                self.active_handler.ref_text = ref_text or ""
                self.active_handler.xvec_only = xvec_only
                self.active_handler.language = language
                return self.active_handler

            # If switching model or first load, clean up previous
            if self.active_handler is not None:
                try:
                    self.active_handler.cleanup()
                except Exception as exc:
                    logger.warning("Error cleaning up previous TTS handler: %s", exc)
                self.active_handler = None

            logger.info("Initializing Qwen3TTSHandler with model=%s on device=%s", model_name, target_device)
            stop_event = Event()
            should_listen = Event()
            queue_in: Queue[Any] = Queue()
            queue_out: Queue[Any] = Queue()

            handler = Qwen3TTSHandler(
                stop_event,
                queue_in=queue_in,
                queue_out=queue_out,
                setup_args=(should_listen,),
                setup_kwargs={
                    "model_name": model_name,
                    "device": target_device,
                    "backend": "torch",
                    "ref_audio": ref_audio_path,
                    "ref_text": ref_text or "",
                    "xvec_only": xvec_only,
                    "language": language,
                },
            )
            self.active_handler = handler
            self.current_model_name = model_name
            return handler

    def synthesize_voice_clone(
        self,
        text: str,
        ref_audio_path: str | Path,
        ref_text: Optional[str] = None,
        xvec_only: bool = True,
        language: str = "auto",
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ) -> VoiceCloningBenchmarkResult:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Input text cannot be empty.")

        is_valid, ref_duration, err_msg = self.validate_reference_audio(ref_audio_path)
        if not is_valid:
            raise ValueError(err_msg)

        if not xvec_only and not (ref_text and ref_text.strip()):
            raise ValueError("Reference transcript is required when x_vector_only mode is disabled.")

        is_official = language in OFFICIAL_QWEN3_LANGUAGES or language == "auto"

        # Track GPU memory prior to run
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        handler = self.ensure_model_loaded(
            model_name=model_name,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            xvec_only=xvec_only,
            language=language,
        )

        tts_input = TTSInput(text=clean_text, language_code=language)

        start_time = time.perf_counter()
        audio_chunks: List[np.ndarray] = []

        try:
            for chunk in handler.process(tts_input):
                if chunk is None:
                    continue
                if isinstance(chunk, (bytes, bytearray)):
                    arr = np.frombuffer(chunk, dtype=np.int16)
                    audio_chunks.append(arr)
                elif isinstance(chunk, np.ndarray):
                    audio_chunks.append(chunk)
        except Exception as e:
            logger.error("Synthesis failed: %s", e, exc_info=True)
            raise RuntimeError(f"Voice cloning generation error: {e}") from e

        generation_time_s = time.perf_counter() - start_time
        generation_time_ms = round(generation_time_s * 1000.0, 2)

        if not audio_chunks:
            raise RuntimeError("TTS handler produced no audio output.")

        combined_audio = np.concatenate(audio_chunks)
        sample_rate = self.sample_rate
        total_samples = len(combined_audio)
        audio_duration_s = round(total_samples / sample_rate, 3)

        # Real-time factor = generation_time / audio_duration
        rtf = round(generation_time_s / audio_duration_s, 4) if audio_duration_s > 0 else 0.0

        gpu_stats = self.get_gpu_stats()

        # Encode to WAV in memory
        import base64

        wav_io = io.BytesIO()
        sf.write(wav_io, combined_audio, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        return VoiceCloningBenchmarkResult(
            model_name=model_name,
            reference_audio_name=Path(ref_audio_path).name,
            reference_duration_seconds=round(ref_duration, 2),
            language=language,
            is_official_language=is_official,
            cloning_mode="x_vector_only" if xvec_only else "reference_transcript",
            input_text=clean_text,
            generation_time_ms=generation_time_ms,
            generated_audio_duration_seconds=audio_duration_s,
            rtf=rtf,
            sample_rate=sample_rate,
            gpu_memory_allocated_mb=gpu_stats["allocated_mb"],
            gpu_memory_reserved_mb=gpu_stats["reserved_mb"],
            gpu_name=gpu_stats["gpu_name"],
            cuda_available=gpu_stats["cuda_available"],
            audio_base64=f"data:audio/wav;base64,{audio_b64}",
        )
