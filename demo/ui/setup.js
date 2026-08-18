// @ts-check
/**
 * Setup & Voice Studio Controller.
 * Allows user to configure STT, LLM, TTS, reference voice, and language,
 * and start the speech-to-speech background pipeline with a single click.
 */

import { $ } from "./dom.js";

export class SetupController {
  /**
   * @param {Object} options
   * @param {() => void} options.onPipelineReady - Callback when pipeline is ready and orb should connect
   * @param {() => void} options.onPipelineStopped - Callback when pipeline is stopped
   */
  constructor({ onPipelineReady, onPipelineStopped }) {
    this.onPipelineReady = onPipelineReady;
    this.onPipelineStopped = onPipelineStopped;

    // Modal dialog
    /** @type {HTMLDialogElement} */
    this.modal = $("#setup-modal");
    /** @type {HTMLButtonElement} */
    this.btnOpenModal = $("#open-setup-modal-btn");
    /** @type {HTMLButtonElement} */
    this.btnCloseModal = $("#setup-modal-close");

    // Active voice bar elements on main screen
    /** @type {HTMLElement} */
    this.activeVoiceBar = $("#active-voice-bar");
    /** @type {HTMLElement} */
    this.activeVoiceName = $("#active-voice-name");
    /** @type {HTMLElement} */
    this.activeModelName = $("#active-model-name");

    // Topbar pipeline badge
    /** @type {HTMLElement} */
    this.topbarPipelineBadge = $("#topbar-pipeline-badge");
    /** @type {HTMLButtonElement} */
    this.btnReconfigure = $("#reconfigure-btn");

    // Setup fields
    /** @type {HTMLSelectElement} */
    this.sttSelect = $("#setup-stt-provider");
    /** @type {HTMLSelectElement} */
    this.llmSelect = $("#setup-llm-provider");
    /** @type {HTMLSelectElement} */
    this.ttsSelect = $("#setup-tts-provider");

    // Qwen3 & Voice options
    /** @type {HTMLElement} */
    this.qwenOptionsCard = $("#setup-qwen-options");
    /** @type {HTMLSelectElement} */
    this.qwenModelSelect = $("#setup-qwen-model");
    /** @type {HTMLSelectElement} */
    this.refVoiceSelect = $("#setup-ref-voice");
    /** @type {HTMLInputElement} */
    this.fileUploadInput = $("#setup-file-upload");
    /** @type {HTMLAudioElement} */
    this.voicePreviewPlayer = $("#setup-voice-preview");
    /** @type {HTMLElement} */
    this.voiceStatus = $("#setup-voice-status");

    /** @type {NodeListOf<HTMLInputElement>} */
    this.modeRadios = document.querySelectorAll('input[name="setup-cloning-mode"]');
    /** @type {HTMLElement} */
    this.transcriptContainer = $("#setup-transcript-container");
    /** @type {HTMLTextAreaElement} */
    this.transcriptInput = $("#setup-transcript-input");
    /** @type {HTMLSelectElement} */
    this.langSelect = $("#setup-language-select");
    /** @type {HTMLElement} */
    this.langBadge = $("#setup-lang-badge");

    // Start / Launch button & status
    /** @type {HTMLButtonElement} */
    this.btnStart = $("#start-pipeline-btn");
    /** @type {HTMLElement} */
    this.btnStartText = this.btnStart.querySelector(".btn-text");
    /** @type {HTMLElement} */
    this.btnStartSpinner = this.btnStart.querySelector(".btn-spinner");
    /** @type {HTMLElement} */
    this.statusCard = $("#setup-status-card");
    /** @type {HTMLElement} */
    this.statusMsg = $("#setup-status-msg");
    /** @type {HTMLElement} */
    this.logConsole = $("#setup-log-console");

    /** @type {number | null} */
    this.pollTimer = null;

    this.init();
  }

  init() {
    this.loadProviders();

    // Open setup modal
    if (this.btnOpenModal) {
      this.btnOpenModal.addEventListener("click", () => {
        if (this.modal) this.modal.showModal();
      });
    }

    // Close setup modal
    if (this.btnCloseModal) {
      this.btnCloseModal.addEventListener("click", () => {
        if (this.modal) this.modal.close();
      });
    }

    if (this.modal) {
      this.modal.addEventListener("click", (e) => {
        if (e.target === this.modal) this.modal.close();
      });
    }

    // Reconfigure button in topbar
    if (this.btnReconfigure) {
      this.btnReconfigure.addEventListener("click", () => {
        this.stopPipeline();
        if (this.modal) this.modal.showModal();
      });
    }

    // TTS provider change
    if (this.ttsSelect) {
      this.ttsSelect.addEventListener("change", () => {
        const isQwen = this.ttsSelect.value === "qwen3";
        if (this.qwenOptionsCard) {
          this.qwenOptionsCard.style.display = isQwen ? "flex" : "none";
        }
      });
    }

    // Qwen model change
    if (this.qwenModelSelect) {
      this.qwenModelSelect.addEventListener("change", () => {
        const isBase = this.qwenModelSelect.value.includes("Base");
        const voiceSection = $("#setup-voice-selection-group");
        if (voiceSection) {
          voiceSection.style.display = isBase ? "block" : "none";
        }
      });
    }

    // Reference voice dropdown change
    if (this.refVoiceSelect) {
      this.refVoiceSelect.addEventListener("change", () => {
        const val = this.refVoiceSelect.value;
        if (val) {
          if (this.voicePreviewPlayer) this.voicePreviewPlayer.src = `/api/tts/reference-audio/${val}`;
          if (this.voiceStatus) this.voiceStatus.textContent = val;
          if (this.activeVoiceName) this.activeVoiceName.textContent = val;
        }
      });
    }

    // Upload voice file
    if (this.fileUploadInput) {
      this.fileUploadInput.addEventListener("change", async () => {
        if (this.fileUploadInput.files && this.fileUploadInput.files.length > 0) {
          await this.handleFileUpload(this.fileUploadInput.files[0]);
        }
      });
    }

    // Cloning mode radio change
    this.modeRadios.forEach((radio) => {
      radio.addEventListener("change", () => {
        const isXvec = radio.value === "x_vector_only";
        if (this.transcriptContainer) {
          this.transcriptContainer.style.display = isXvec ? "none" : "block";
        }
      });
    });

    // Language change
    if (this.langSelect) {
      this.langSelect.addEventListener("change", () => {
        this.updateLanguageBadge();
      });
    }

    // Start conversation button
    if (this.btnStart) {
      this.btnStart.addEventListener("click", () => {
        this.startPipeline();
      });
    }
  }

  updateLanguageBadge() {
    if (!this.langSelect || !this.langBadge) return;
    const lang = this.langSelect.value;
    const isOfficial = !["km", "vi", "th", "id", "ar", "hi"].includes(lang);
    if (isOfficial) {
      this.langBadge.className = "badge badge-success";
      this.langBadge.textContent = "Official";
    } else {
      this.langBadge.className = "badge badge-experimental";
      this.langBadge.textContent = "Experimental";
    }
  }

  async loadProviders() {
    try {
      const res = await fetch("/api/providers");
      if (!res.ok) return;
      const data = await res.json();

      // Populate reference voice dropdown
      if (data.reference_files && data.reference_files.length > 0 && this.refVoiceSelect) {
        const current = this.refVoiceSelect.value;
        this.refVoiceSelect.innerHTML = "";
        data.reference_files.forEach((/** @type {{ name: string, duration: number }} */ f) => {
          const opt = document.createElement("option");
          opt.value = f.name;
          opt.textContent = `${f.name} (${f.duration}s)`;
          // Select khmer_bong_nika_sound.wav by default if present
          if (f.name === "khmer_bong_nika_sound.wav" && !current) {
            opt.selected = true;
          }
          this.refVoiceSelect.appendChild(opt);
        });

        if (current && Array.from(this.refVoiceSelect.options).some((o) => o.value === current)) {
          this.refVoiceSelect.value = current;
        }

        const activeName = this.refVoiceSelect.value;
        if (activeName) {
          if (this.voicePreviewPlayer) this.voicePreviewPlayer.src = `/api/tts/reference-audio/${activeName}`;
          if (this.voiceStatus) this.voiceStatus.textContent = activeName;
          if (this.activeVoiceName) this.activeVoiceName.textContent = activeName;
        }
      }

      // Check initial pipeline status
      if (data.pipeline_status && data.pipeline_status.running) {
        this.updateActivePill(data.pipeline_status.config, true);
      } else {
        this.updateActivePill(null, false);
      }
    } catch (e) {
      console.warn("Failed to load providers:", e);
    }
  }

  async handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    try {
      if (this.voiceStatus) this.voiceStatus.textContent = "Uploading...";
      const res = await fetch("/api/tts/upload-reference", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      // Add to select
      if (this.refVoiceSelect) {
        const opt = document.createElement("option");
        opt.value = data.filename;
        opt.textContent = `${data.filename} (${data.duration}s)`;
        this.refVoiceSelect.insertBefore(opt, this.refVoiceSelect.firstChild);
        this.refVoiceSelect.selectedIndex = 0;
      }

      if (this.voicePreviewPlayer) {
        this.voicePreviewPlayer.src = data.url;
        this.voicePreviewPlayer.load();
      }
      if (this.voiceStatus) this.voiceStatus.textContent = `${data.filename} (${data.duration}s)`;
      if (this.activeVoiceName) this.activeVoiceName.textContent = data.filename;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      alert(`Voice upload error: ${msg}`);
      if (this.voiceStatus) this.voiceStatus.textContent = "Upload failed";
    }
  }

  async startPipeline() {
    const stt = this.sttSelect?.value || "parakeet-tdt";
    const llm = this.llmSelect?.value || "chat-completions";
    const tts = this.ttsSelect?.value || "qwen3";
    const model = this.qwenModelSelect?.value || "Qwen/Qwen3-TTS-12Hz-1.7B-Base";
    const refVoice = this.refVoiceSelect?.value || "khmer_bong_nika_sound.wav";
    const mode = Array.from(this.modeRadios).find((r) => r.checked)?.value || "x_vector_only";
    const transcript = this.transcriptInput?.value.trim() || "";
    const language = this.langSelect?.value || "auto";

    this.setStarting(true);
    if (this.statusCard) this.statusCard.hidden = false;
    if (this.statusMsg) this.statusMsg.textContent = "Starting speech-to-speech engine (loading CUDA models)...";
    if (this.logConsole) this.logConsole.textContent = "";

    const payload = {
      stt_provider: stt,
      llm_provider: llm,
      tts_provider: tts,
      tts_model_name: model,
      tts_backend: "torch",
      ref_audio_name: refVoice,
      ref_transcript: transcript || null,
      xvec_only: mode === "x_vector_only",
      language: language,
      port: 8081,
    };

    try {
      const res = await fetch("/api/pipeline/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to start pipeline");
      }

      // Poll until the pipeline server is ready
      this.pollPipelineReady(payload);
    } catch (err) {
      this.setStarting(false);
      const msg = err instanceof Error ? err.message : String(err);
      if (this.statusMsg) this.statusMsg.textContent = `Error: ${msg}`;
    }
  }

  /** @param {any} config */
  pollPipelineReady(config) {
    let attempts = 0;
    const maxAttempts = 40;

    const check = async () => {
      attempts++;
      try {
        const res = await fetch("/api/pipeline/status");
        if (res.ok) {
          const status = await res.json();
          if (status.recent_logs && status.recent_logs.length > 0 && this.logConsole) {
            this.logConsole.textContent = status.recent_logs.slice(-8).join("\n");
          }

          if (status.running) {
            const logsStr = (status.recent_logs || []).join(" ");
            const isReady =
              logsStr.includes("OpenAI Realtime API starting") ||
              logsStr.includes("Uvicorn running on") ||
              logsStr.includes("Application startup complete");

            if (isReady) {
              this.setStarting(false);
              this.updateActivePill(config, true);
              if (this.modal) this.modal.close();
              if (typeof this.onPipelineReady === "function") {
                this.onPipelineReady();
              }
              return;
            }
          } else if (attempts >= 3 && !status.running) {
            const lastLog = (status.recent_logs || []).slice(-3).join("\n") || "Process terminated.";
            throw new Error(`Pipeline exited unexpectedly:\n${lastLog}`);
          }
        }
      } catch (err) {
        this.setStarting(false);
        const msg = err instanceof Error ? err.message : String(err);
        if (this.statusMsg) this.statusMsg.textContent = msg;
        return;
      }

      if (attempts < maxAttempts) {
        this.pollTimer = window.setTimeout(check, 1500);
      } else {
        this.setStarting(false);
        if (this.statusMsg) this.statusMsg.textContent = "Pipeline startup timed out. Check GPU memory and logs.";
      }
    };

    this.pollTimer = window.setTimeout(check, 1500);
  }

  async stopPipeline() {
    if (this.pollTimer) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }

    try {
      await fetch("/api/pipeline/stop", { method: "POST" });
    } catch (e) {
      console.warn("Stop pipeline request error:", e);
    }

    this.updateActivePill(null, false);
    if (typeof this.onPipelineStopped === "function") {
      this.onPipelineStopped();
    }
  }

  /**
   * @param {any} config
   * @param {boolean} running
   */
  updateActivePill(config, running) {
    if (this.topbarPipelineBadge) {
      this.topbarPipelineBadge.hidden = !running;
      if (running) {
        const voiceName = config?.ref_audio_name || this.refVoiceSelect?.value || "khmer_bong_nika_sound.wav";
        const badgeText = this.topbarPipelineBadge.querySelector(".pipeline-badge-text");
        if (badgeText) badgeText.textContent = `Pipeline: Online (${voiceName})`;
      }
    }

    if (this.activeVoiceName) {
      this.activeVoiceName.textContent = config?.ref_audio_name || this.refVoiceSelect?.value || "khmer_bong_nika_sound.wav";
    }
    if (this.activeModelName) {
      const isBase = (config?.tts_model_name || this.qwenModelSelect?.value || "").includes("Base");
      this.activeModelName.textContent = isBase ? "Qwen3-TTS 1.7B Base (Cloned)" : "Qwen3-TTS CustomVoice";
    }
  }

  /** @param {boolean} starting */
  setStarting(starting) {
    if (this.btnStart) this.btnStart.disabled = starting;
    if (this.btnStartText) {
      this.btnStartText.textContent = starting ? "Launching CUDA Pipeline..." : "Apply & Start Pipeline";
    }
    if (this.btnStartSpinner) {
      this.btnStartSpinner.hidden = !starting;
    }
  }
}
