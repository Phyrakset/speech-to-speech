import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import server as demo_server
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.voice_cloning_bench import (
    ALL_TEST_LANGUAGES,
    EXPERIMENTAL_LANGUAGES,
    OFFICIAL_QWEN3_LANGUAGES,
    VoiceCloningBenchmarkResult,
    VoiceCloningService,
)


@pytest.fixture
def sample_wav_path(tmp_path: Path) -> Path:
    """Creates a temporary valid 1-second 16kHz mono WAV file for testing."""
    wav_path = tmp_path / "test_ref.wav"
    sr = 16000
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr, endpoint=False)).astype(np.float32)
    sf.write(str(wav_path), samples, sr, format="WAV", subtype="PCM_16")
    return wav_path


def test_language_classification():
    """Verify that official vs experimental languages are correctly separated."""
    assert "en" in OFFICIAL_QWEN3_LANGUAGES
    assert "km" in EXPERIMENTAL_LANGUAGES
    assert "km" in ALL_TEST_LANGUAGES
    assert "en" in ALL_TEST_LANGUAGES


def test_validate_reference_audio(sample_wav_path: Path, tmp_path: Path):
    """Test reference audio validation for valid files, missing files, and corrupted files."""
    cloner = VoiceCloningService()
    
    # Valid file
    is_valid, duration, err = cloner.validate_reference_audio(sample_wav_path)
    assert is_valid is True
    assert 0.9 <= duration <= 1.1
    assert err == ""

    # Non-existent file
    is_valid, duration, err = cloner.validate_reference_audio(tmp_path / "non_existent.wav")
    assert is_valid is False
    assert "not found" in err

    # Corrupted / empty file
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_text("not audio data")
    is_valid, duration, err = cloner.validate_reference_audio(corrupt_path)
    assert is_valid is False
    assert "Invalid audio file" in err


def test_voice_cloning_xvec_mode_synthesize(sample_wav_path: Path):
    """Test voice cloning synthesis in x_vector_only mode with mocked TTS handler."""
    cloner = VoiceCloningService(default_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    mock_handler = MagicMock()
    # Mock generation of 16000 samples (1 second at 16kHz)
    mock_audio_chunk = np.zeros(16000, dtype=np.int16)
    mock_handler.process.return_value = iter([mock_audio_chunk])
    mock_handler.device = "cuda"

    with patch.object(cloner, "ensure_model_loaded", return_value=mock_handler):
        result = cloner.synthesize_voice_clone(
            text="Hello, this is a test.",
            ref_audio_path=sample_wav_path,
            xvec_only=True,
            language="en",
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        )

        assert isinstance(result, VoiceCloningBenchmarkResult)
        assert result.model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        assert result.language == "en"
        assert result.is_official_language is True
        assert result.cloning_mode == "x_vector_only"
        assert result.generated_audio_duration_seconds == 1.0
        assert result.generation_time_ms > 0
        assert result.rtf >= 0
        assert result.audio_base64 is not None
        assert result.audio_base64.startswith("data:audio/wav;base64,")


def test_voice_cloning_khmer_experimental_mode(sample_wav_path: Path):
    """Test voice cloning with Khmer (km) marked as experimental."""
    cloner = VoiceCloningService()

    mock_handler = MagicMock()
    mock_audio_chunk = np.zeros(16000, dtype=np.int16)
    mock_handler.process.return_value = iter([mock_audio_chunk])

    with patch.object(cloner, "ensure_model_loaded", return_value=mock_handler):
        result = cloner.synthesize_voice_clone(
            text="សួស្ដី សូមស្វាគមន៍",
            ref_audio_path=sample_wav_path,
            xvec_only=True,
            language="km",
        )

        assert result.language == "km"
        assert result.is_official_language is False  # Khmer is experimental


def test_voice_cloning_transcript_mode_requires_text(sample_wav_path: Path):
    """Test that reference transcript mode enforces non-empty reference transcript."""
    cloner = VoiceCloningService()

    with pytest.raises(ValueError, match="Reference transcript is required"):
        cloner.synthesize_voice_clone(
            text="Hello",
            ref_audio_path=sample_wav_path,
            ref_text="",
            xvec_only=False,
        )


def test_api_providers_endpoint():
    """Test /api/providers returns STT, LLM, TTS backends and language list."""
    client = TestClient(demo_server.app)
    response = client.get("/api/providers")
    assert response.status_code == 200
    data = response.json()

    assert "stt_providers" in data
    assert "llm_providers" in data
    assert "tts_providers" in data
    assert "supported_languages" in data
    assert "default_qwen_model" in data

    # Verify qwen3 is in tts_providers
    assert "qwen3" in data["tts_providers"]

    # Verify language entries have is_official property
    en_entry = next(item for item in data["supported_languages"] if item["code"] == "en")
    assert en_entry["is_official"] is True
    km_entry = next(item for item in data["supported_languages"] if item["code"] == "km")
    assert km_entry["is_official"] is False


def test_api_upload_reference_and_clone(sample_wav_path: Path):
    """Test FastAPI upload reference endpoint and clone endpoint."""
    client = TestClient(demo_server.app)

    # 1. Upload reference audio
    with sample_wav_path.open("rb") as f:
        upload_resp = client.post(
            "/api/tts/upload-reference",
            files={"file": ("test_ref.wav", f, "audio/wav")},
        )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["ok"] is True
    filename = upload_data["filename"]

    # 2. Test clone endpoint with mocked synthesis
    mock_result = VoiceCloningBenchmarkResult(
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        reference_audio_name=filename,
        reference_duration_seconds=1.0,
        language="en",
        is_official_language=True,
        cloning_mode="x_vector_only",
        input_text="Test synthesis",
        generation_time_ms=250.0,
        generated_audio_duration_seconds=1.5,
        rtf=0.1667,
        sample_rate=16000,
        gpu_memory_allocated_mb=3500.0,
        gpu_memory_reserved_mb=4000.0,
        gpu_name="NVIDIA Tesla T4",
        cuda_available=True,
        audio_base64="data:audio/wav;base64,UklGRgAAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
    )

    with patch.object(VoiceCloningService, "synthesize_voice_clone", return_value=mock_result):
        clone_resp = client.post(
            "/api/tts/clone",
            data={
                "text": "Test synthesis",
                "reference_filename": filename,
                "cloning_mode": "x_vector_only",
                "language": "en",
                "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            },
        )

        assert clone_resp.status_code == 200
        data = clone_resp.json()
        assert data["model_name"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        assert data["generation_time_ms"] == 250.0
        assert data["rtf"] == 0.1667
        assert data["audio_base64"].startswith("data:audio/wav;base64,")
