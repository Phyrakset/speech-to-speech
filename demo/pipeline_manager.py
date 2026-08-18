"""
Pipeline Process Manager for Speech-to-Speech.
Allows the Web UI to configure and spawn the speech-to-speech serve subprocess
directly without requiring manual terminal commands.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from speech_to_speech.utils.utils import load_dotenv_if_present

# Ensure .env is loaded on startup
load_dotenv_if_present()

logger = logging.getLogger("s2s.pipeline_manager")


@dataclass
class PipelineConfig:
    stt_provider: str = field(default_factory=lambda: os.getenv("DEFAULT_STT_PROVIDER", "parakeet-tdt"))
    llm_provider: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "gemini-flash"))
    llm_model_name: Optional[str] = field(
        default_factory=lambda: os.getenv("MODEL_NAME", os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash"))
    )
    tts_provider: str = field(default_factory=lambda: os.getenv("DEFAULT_TTS_PROVIDER", "qwen3"))
    tts_model_name: str = field(
        default_factory=lambda: os.getenv("DEFAULT_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    )
    tts_backend: str = field(default_factory=lambda: os.getenv("DEFAULT_TTS_BACKEND", "torch"))
    ref_audio_path: Optional[str] = field(
        default_factory=lambda: os.getenv("DEFAULT_REFERENCE_VOICE_PATH", None)
    )
    ref_transcript: Optional[str] = None
    xvec_only: bool = True
    language: str = field(default_factory=lambda: os.getenv("DEFAULT_TTS_LANGUAGE", "auto"))
    port: int = field(default_factory=lambda: int(os.getenv("PIPELINE_PORT", "8081")))
    host: str = field(default_factory=lambda: os.getenv("PIPELINE_HOST", "0.0.0.0"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineManager:
    """Manages the background subprocess for `speech-to-speech serve`."""

    _instance: Optional[PipelineManager] = None
    _lock = threading.Lock()

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path(__file__).resolve().parent.parent
        self.process: Optional[subprocess.Popen] = None
        self.current_config: Optional[PipelineConfig] = None
        self.logs: deque[str] = deque(maxlen=200)
        self.started_at: Optional[float] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._status_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> PipelineManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_status(self) -> Dict[str, Any]:
        with self._status_lock:
            running = self.is_running
            return {
                "running": running,
                "pid": self.process.pid if running and self.process else None,
                "started_at": self.started_at,
                "uptime_seconds": round(time.time() - self.started_at, 1) if running and self.started_at else 0,
                "config": self.current_config.to_dict() if self.current_config else None,
                "port": self.current_config.port if self.current_config else 8080,
                "ws_url": f"ws://127.0.0.1:{self.current_config.port}/v1/realtime" if self.current_config else "ws://127.0.0.1:8080/v1/realtime",
                "recent_logs": list(self.logs)[-20:],
            }

    def _log_reader(self, pipe: Any):
        """Consume subprocess stdout/stderr and store in log buffer."""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                stripped = line.strip()
                if stripped:
                    with self._status_lock:
                        self.logs.append(stripped)
                    logger.info("[s2s-pipeline] %s", stripped)
        except Exception as e:
            logger.debug("Log reader stopped: %s", e)
        finally:
            pipe.close()

    def build_command(self, config: PipelineConfig) -> List[str]:
        """Convert PipelineConfig into the exact CLI arguments for speech-to-speech serve."""
        venv_bin = self.workspace_root / ".venv" / "bin" / "speech-to-speech"
        executable = str(venv_bin) if venv_bin.is_file() else "speech-to-speech"

        # Resolve LLM backend and model name
        llm_backend = "chat-completions"
        model_name = None

        if config.llm_provider == "gemini-flash":
            llm_backend = "chat-completions"
            model_name = "gemini-2.5-flash"
        elif config.llm_provider == "gemini-pro":
            llm_backend = "chat-completions"
            model_name = "gemini-2.0-flash"
        elif config.llm_provider == "openai-mini":
            llm_backend = "chat-completions"
            model_name = "gpt-4o-mini"
        elif config.llm_provider == "openai-gpt4o":
            llm_backend = "chat-completions"
            model_name = "gpt-4o"
        elif config.llm_provider == "groq-llama":
            llm_backend = "chat-completions"
            model_name = "llama-3.3-70b-versatile"
        elif config.llm_provider == "deepseek-chat":
            llm_backend = "chat-completions"
            model_name = "deepseek-chat"
        elif config.llm_provider == "transformers":
            llm_backend = "transformers"
        elif config.llm_provider == "responses-api":
            llm_backend = "responses-api"
        elif config.llm_provider == "chat-completions":
            llm_backend = "chat-completions"
            if config.llm_model_name:
                model_name = config.llm_model_name

        cmd = [
            executable,
            "serve",
            "--port", str(config.port),
            "--host", config.host,
            "--stt", config.stt_provider,
            "--llm_backend", llm_backend,
            "--tts", config.tts_provider,
        ]

        if model_name:
            cmd.extend(["--model_name", model_name])

        # Qwen3-specific arguments
        if config.tts_provider == "qwen3":
            cmd.extend(["--qwen3_tts_model_name", config.tts_model_name])
            cmd.extend(["--qwen3_tts_backend", config.tts_backend])
            
            if config.ref_audio_path and os.path.isfile(config.ref_audio_path):
                cmd.extend(["--qwen3_tts_ref_audio", str(config.ref_audio_path)])
                if config.xvec_only:
                    cmd.append("--qwen3_tts_xvec_only")
                elif config.ref_transcript:
                    cmd.extend(["--qwen3_tts_ref_text", config.ref_transcript])

            if config.language and config.language != "auto":
                cmd.extend(["--qwen3_tts_language", config.language])

        return cmd

    def _free_port(self, port: int):
        """Ensure no dangling process is holding the server port."""
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=2)
            time.sleep(0.5)
        except Exception:
            pass

    def start(self, config: PipelineConfig) -> Dict[str, Any]:
        """Start or restart the speech-to-speech pipeline with the specified config."""
        with self._status_lock:
            if self.is_running:
                logger.info("Stopping existing running pipeline before start...")
                self._stop_process()

            self._free_port(config.port)
            self.logs.clear()
            self.current_config = config
            cmd = self.build_command(config)
            logger.info("Starting speech-to-speech pipeline with command: %s", " ".join(cmd))

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(self.workspace_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )
                self.started_at = time.time()
                self.logs.append(f"Started pipeline PID {self.process.pid}: {' '.join(cmd)}")

                # Launch background thread to read logs
                self._reader_thread = threading.Thread(
                    target=self._log_reader,
                    args=(self.process.stdout,),
                    daemon=True,
                )
                self._reader_thread.start()

                return {
                    "ok": True,
                    "message": f"Pipeline started on port {config.port}",
                    "pid": self.process.pid,
                    "ws_url": f"ws://127.0.0.1:{config.port}/v1/realtime",
                }
            except Exception as e:
                logger.error("Failed to start speech-to-speech pipeline: %s", e, exc_info=True)
                self.logs.append(f"Failed to start: {e}")
                return {"ok": False, "error": str(e)}

    def _stop_process(self):
        """Internal helper to terminate the running process group."""
        if self.process is None:
            return

        try:
            pid = self.process.pid
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except Exception:
                    self.process.terminate()
            else:
                self.process.terminate()

            # Wait up to 5s for graceful shutdown
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except Exception:
                        self.process.kill()
                else:
                    self.process.kill()
                self.process.wait(timeout=2)
        except Exception as e:
            logger.warning("Error while terminating pipeline process: %s", e)
        finally:
            self.process = None
            self.started_at = None

    def stop(self) -> Dict[str, Any]:
        """Stop the running pipeline."""
        with self._status_lock:
            if not self.is_running:
                return {"ok": True, "message": "Pipeline was not running"}

            self._stop_process()
            self.logs.append("Pipeline stopped by user request.")
            return {"ok": True, "message": "Pipeline stopped"}
