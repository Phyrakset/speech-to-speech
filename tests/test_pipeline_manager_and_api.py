import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import server as demo_server
from pipeline_manager import PipelineConfig, PipelineManager


@pytest.fixture
def client():
    return TestClient(demo_server.app)


def test_pipeline_config_build_command():
    manager = PipelineManager(workspace_root=Path("/fake/path"))
    cfg = PipelineConfig(
        stt_provider="parakeet-tdt",
        llm_provider="chat-completions",
        tts_provider="qwen3",
        tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        tts_backend="torch",
        ref_audio_path="/fake/voice.wav",
        xvec_only=True,
        language="en",
        port=8080,
    )

    with patch("os.path.isfile", return_value=True):
        cmd = manager.build_command(cfg)

    assert "serve" in cmd
    assert "--stt" in cmd and "parakeet-tdt" in cmd
    assert "--llm_backend" in cmd and "chat-completions" in cmd
    assert "--tts" in cmd and "qwen3" in cmd
    assert "--qwen3_tts_model_name" in cmd and "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in cmd
    assert "--qwen3_tts_backend" in cmd and "torch" in cmd
    assert "--qwen3_tts_ref_audio" in cmd and "/fake/voice.wav" in cmd
    assert "--qwen3_tts_xvec_only" in cmd
    assert "--qwen3_tts_language" in cmd and "en" in cmd

    # Test Gemini selection
    gemini_cfg = PipelineConfig(
        stt_provider="parakeet-tdt",
        llm_provider="gemini-flash",
        tts_provider="qwen3",
        port=8081,
    )
    gemini_cmd = manager.build_command(gemini_cfg)
    assert "--llm_backend" in gemini_cmd and "chat-completions" in gemini_cmd
    assert "--model_name" in gemini_cmd and "gemini-2.5-flash" in gemini_cmd


def test_api_pipeline_status_and_control(client):
    manager = PipelineManager.get_instance()

    # Status when stopped
    res = client.get("/api/pipeline/status")
    assert res.status_code == 200
    data = res.json()
    assert "running" in data
    assert "port" in data

    # Stop endpoint
    res_stop = client.post("/api/pipeline/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["ok"] is True


def test_api_config_includes_pipeline_info(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "pipeline" in data
    assert "s2sUrl" in data
