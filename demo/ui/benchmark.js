// @ts-check
/**
 * Voice Cloning & Speech-to-Speech Benchmarking UI Controller.
 * Implements a 3-step wizard: Model & Voice → Configure → Results.
 */

import { $ } from "./dom.js";

export class BenchmarkController {
  constructor() {
    /** @type {HTMLButtonElement} */
    this.btnOpen = $("#benchmark-btn");
    /** @type {HTMLDialogElement} */
    this.modal = $("#benchmark-modal");
    /** @type {HTMLButtonElement} */
    this.btnClose = $("#benchmark-close");

    // ── Wizard navigation ──
    /** @type {number} */
    this.currentStep = 1;
    /** @type {HTMLElement[]} */
    this.panels = [
      $("#wizard-panel-1"),
      $("#wizard-panel-2"),
      $("#wizard-panel-3"),
    ];
    /** @type {HTMLElement[]} */
    this.stepIndicators = Array.from(
      document.querySelectorAll(".wizard-step[data-step]")
    );

    // Step 1 controls
    /** @type {NodeListOf<HTMLInputElement>} */
    this.modelRadios = document.querySelectorAll('input[name="tts-model"]');
    /** @type {NodeListOf<HTMLLabelElement>} */
    this.modelCards = document.querySelectorAll(".model-card");
    /** @type {HTMLInputElement} */
    this.fileInput = $("#ref-file-input");
    /** @type {HTMLSelectElement} */
    this.audioSelect = $("#ref-audio-select");
    /** @type {HTMLAudioElement} */
    this.refPlayer = $("#ref-audio-player");
    /** @type {HTMLElement} */
    this.refStatus = $("#ref-audio-status");
    /** @type {HTMLButtonElement} */
    this.btnNext1 = $("#wizard-next-1");

    // Step 2 controls
    /** @type {NodeListOf<HTMLInputElement>} */
    this.modeRadios = document.querySelectorAll('input[name="cloning-mode"]');
    /** @type {HTMLElement} */
    this.transcriptContainer = $("#ref-transcript-container");
    /** @type {HTMLTextAreaElement} */
    this.transcriptInput = $("#ref-transcript-input");
    /** @type {HTMLSelectElement} */
    this.langSelect = $("#target-language-select");
    /** @type {HTMLElement} */
    this.langBadge = $("#lang-status-badge");
    /** @type {HTMLElement} */
    this.langExplanation = $("#lang-explanation");
    /** @type {HTMLTextAreaElement} */
    this.textInput = $("#tts-input-text");
    /** @type {HTMLButtonElement} */
    this.btnGenerate = $("#generate-speech-btn");
    /** @type {HTMLElement} */
    this.btnText = this.btnGenerate.querySelector(".btn-text");
    /** @type {HTMLElement} */
    this.btnSpinner = this.btnGenerate.querySelector(".btn-spinner");
    /** @type {HTMLButtonElement} */
    this.btnBack2 = $("#wizard-back-2");

    // Step 3 controls
    /** @type {HTMLElement} */
    this.errorCard = $("#bench-error-card");
    /** @type {HTMLElement} */
    this.errorMsg = $("#bench-error-msg");
    /** @type {HTMLElement} */
    this.resultsCard = $("#benchmark-results-card");
    /** @type {HTMLElement} */
    this.rtfBadge = $("#rtf-badge");
    /** @type {HTMLAudioElement} */
    this.resRefPlayer = $("#result-ref-player");
    /** @type {HTMLElement} */
    this.resRefMeta = $("#result-ref-meta");
    /** @type {HTMLAudioElement} */
    this.resGenPlayer = $("#result-gen-player");
    /** @type {HTMLElement} */
    this.resGenMeta = $("#result-gen-meta");
    /** @type {HTMLElement} */
    this.metricModel = $("#metric-model");
    /** @type {HTMLElement} */
    this.metricLatency = $("#metric-latency");
    /** @type {HTMLElement} */
    this.metricDuration = $("#metric-duration");
    /** @type {HTMLElement} */
    this.metricRtf = $("#metric-rtf");
    /** @type {HTMLElement} */
    this.metricVram = $("#metric-vram");
    /** @type {HTMLElement} */
    this.metricDevice = $("#metric-device");
    /** @type {HTMLButtonElement} */
    this.btnBack3 = $("#wizard-back-3");
    /** @type {HTMLButtonElement} */
    this.btnNewTest = $("#wizard-new-test");

    /** @type {File | null} */
    this.selectedFile = null;

    /** @type {boolean} */
    this.hasValidVoice = true; // built-in ref_audio.wav is pre-selected

    this.init();
  }

  init() {
    // ── Modal open / close ──
    if (this.btnOpen) {
      this.btnOpen.addEventListener("click", () => {
        this.modal.showModal();
        this.goToStep(1);
        this.fetchProviders();
      });
    }
    if (this.btnClose) {
      this.btnClose.addEventListener("click", () => this.modal.close());
    }
    this.modal.addEventListener("click", (e) => {
      if (e.target === this.modal) this.modal.close();
    });

    // ── Step 1: Model selection ──
    this.modelRadios.forEach((radio) => {
      radio.addEventListener("change", () => {
        this.modelCards.forEach((card) => card.classList.remove("selected"));
        const selected = radio.closest(".model-card");
        if (selected) selected.classList.add("selected");
        this.validateStep1();
      });
    });

    // ── Step 1: File upload ──
    this.fileInput.addEventListener("change", async () => {
      if (this.fileInput.files && this.fileInput.files.length > 0) {
        await this.handleFileUpload(this.fileInput.files[0]);
      }
    });

    // ── Step 1: Audio select dropdown ──
    this.audioSelect.addEventListener("change", () => {
      const filename = this.audioSelect.value;
      this.selectedFile = null;
      this.refStatus.textContent = filename;
      this.refPlayer.src = `/api/tts/reference-audio/${filename}`;
      this.refPlayer.load();
      this.hasValidVoice = true;
      this.validateStep1();
    });

    // Enable the Next button since built-in ref is pre-selected
    this.validateStep1();

    // ── Step 1 → Step 2 ──
    this.btnNext1.addEventListener("click", () => this.goToStep(2));

    // ── Step 2: Cloning mode toggle ──
    this.modeRadios.forEach((radio) => {
      radio.addEventListener("change", () => {
        const isXvec = radio.value === "x_vector_only";
        this.transcriptContainer.style.display = isXvec ? "none" : "flex";
      });
    });

    // ── Step 2: Language select ──
    this.langSelect.addEventListener("change", () =>
      this.updateLanguageStatus()
    );

    // ── Step 2: Quick sample chips ──
    document.querySelectorAll(".sample-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const text = chip.getAttribute("data-text");
        const lang = chip.getAttribute("data-lang");
        if (text) this.textInput.value = text;
        if (lang) {
          this.langSelect.value = lang;
          this.updateLanguageStatus();
        }
      });
    });

    // ── Step 2: Generate ──
    this.btnGenerate.addEventListener("click", () => this.generateSpeech());

    // ── Step 2 ← Back ──
    this.btnBack2.addEventListener("click", () => this.goToStep(1));

    // ── Step 3 ← Back ──
    this.btnBack3.addEventListener("click", () => this.goToStep(2));

    // ── Step 3: New Test → back to step 1 ──
    this.btnNewTest.addEventListener("click", () => this.goToStep(1));
  }

  // ── Wizard Navigation ──────────────────────────────────────────────────

  /** @param {number} step */
  goToStep(step) {
    this.currentStep = step;

    // Update panels visibility
    this.panels.forEach((panel, i) => {
      panel.classList.toggle("active", i + 1 === step);
    });

    // Update step indicator
    this.stepIndicators.forEach((el) => {
      const s = parseInt(el.dataset.step, 10);
      el.classList.toggle("active", s === step);
      el.classList.toggle("completed", s < step);
    });

    // Reset error when navigating
    if (step !== 3) {
      this.hideError();
    }
  }

  validateStep1() {
    const modelSelected = Array.from(this.modelRadios).some((r) => r.checked);
    this.btnNext1.disabled = !(modelSelected && this.hasValidVoice);
  }

  // ── Language Status ────────────────────────────────────────────────────

  updateLanguageStatus() {
    const lang = this.langSelect.value;
    const isOfficial = !["km", "vi", "th", "id", "ar", "hi"].includes(lang);
    if (isOfficial) {
      this.langBadge.className = "badge badge-success";
      this.langBadge.textContent = "Officially Supported";
      this.langExplanation.textContent =
        "This language is officially supported by Qwen3-TTS 1.7B Base.";
    } else {
      this.langBadge.className = "badge badge-experimental";
      this.langBadge.textContent = "Experimental / Unsupported";
      this.langExplanation.textContent =
        "Experimental language: evaluates cross-lingual speaker voice preservation quality.";
    }
  }

  // ── Providers / Reference Files ────────────────────────────────────────

  async fetchProviders() {
    try {
      const res = await fetch("/api/providers");
      if (!res.ok) return;
      const data = await res.json();

      // Populate reference files
      if (data.reference_files && data.reference_files.length > 0) {
        const currentVal = this.audioSelect.value;
        this.audioSelect.innerHTML = "";
        data.reference_files.forEach((f) => {
          const opt = document.createElement("option");
          opt.value = f.name;
          opt.textContent = `${f.name} (${f.duration}s)`;
          this.audioSelect.appendChild(opt);
        });
        if (
          currentVal &&
          Array.from(this.audioSelect.options).some(
            (o) => o.value === currentVal
          )
        ) {
          this.audioSelect.value = currentVal;
        } else if (this.audioSelect.options.length > 0) {
          this.audioSelect.selectedIndex = 0;
        }
        const activeName = this.audioSelect.value;
        this.refStatus.textContent = activeName;
        this.refPlayer.src = `/api/tts/reference-audio/${activeName}`;
        this.hasValidVoice = true;
        this.validateStep1();
      }

      // GPU info
      if (data.gpu_info && data.gpu_info.gpu_name) {
        this.metricDevice.textContent = `${data.gpu_info.gpu_name} (CUDA)`;
      }
    } catch (e) {
      console.warn("Failed to fetch provider info:", e);
    }
  }

  // ── File Upload ────────────────────────────────────────────────────────

  async handleFileUpload(file) {
    this.hideError();
    const formData = new FormData();
    formData.append("file", file);

    try {
      this.refStatus.textContent = "Uploading...";
      const res = await fetch("/api/tts/upload-reference", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      this.selectedFile = file;
      this.refStatus.textContent = `${data.filename} (${data.duration}s)`;
      this.hasValidVoice = true;

      // Add to select if not present
      let exists = false;
      for (let i = 0; i < this.audioSelect.options.length; i++) {
        if (this.audioSelect.options[i].value === data.filename) {
          this.audioSelect.selectedIndex = i;
          exists = true;
          break;
        }
      }
      if (!exists) {
        const opt = document.createElement("option");
        opt.value = data.filename;
        opt.textContent = `${data.filename} (${data.duration}s)`;
        this.audioSelect.insertBefore(opt, this.audioSelect.firstChild);
        this.audioSelect.selectedIndex = 0;
      }

      this.refPlayer.src = data.url;
      this.refPlayer.load();
      this.validateStep1();
    } catch (err) {
      this.showError(`Audio upload error: ${err.message}`);
      this.refStatus.textContent = "Upload failed";
      this.hasValidVoice = false;
      this.validateStep1();
    }
  }

  // ── Generate Speech ────────────────────────────────────────────────────

  async generateSpeech() {
    this.hideError();
    const text = this.textInput.value.trim();
    if (!text) {
      this.showError("Please enter some text to synthesize.");
      return;
    }

    const mode =
      Array.from(this.modeRadios).find((r) => r.checked)?.value ||
      "x_vector_only";
    const transcript = this.transcriptInput.value.trim();
    const language = this.langSelect.value;
    const refFilename = this.audioSelect.value;

    if (mode === "reference_transcript" && !transcript) {
      this.showError(
        "Reference Transcript is required when Reference Transcript mode is selected."
      );
      return;
    }

    this.setLoading(true);

    const formData = new FormData();
    formData.append("text", text);
    formData.append("cloning_mode", mode);
    formData.append("language", language);
    formData.append("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base");

    if (this.selectedFile) {
      formData.append("reference_file", this.selectedFile);
    } else if (refFilename) {
      formData.append("reference_filename", refFilename);
    }

    if (transcript) {
      formData.append("reference_transcript", transcript);
    }

    try {
      const res = await fetch("/api/tts/clone", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Voice cloning generation failed.");
      }

      // Move to Step 3 and display results
      this.goToStep(3);
      this.displayResults(data, refFilename);
    } catch (err) {
      // Move to Step 3 to show error
      this.goToStep(3);
      this.showError(
        err.message || "An unexpected error occurred during synthesis."
      );
    } finally {
      this.setLoading(false);
    }
  }

  // ── Display Results ────────────────────────────────────────────────────

  displayResults(data, refFilename) {
    this.resultsCard.hidden = false;

    // Reference Voice
    const refSrc = `/api/tts/reference-audio/${data.reference_audio_name || refFilename}`;
    this.resRefPlayer.src = refSrc;
    this.resRefMeta.textContent = `Ref Duration: ${data.reference_duration_seconds}s · ${data.cloning_mode}`;

    // Generated Voice
    if (data.audio_base64) {
      this.resGenPlayer.src = data.audio_base64;
      this.resGenMeta.textContent = `Generated Duration: ${data.generated_audio_duration_seconds}s`;
      try {
        this.resGenPlayer.play().catch(() => {});
      } catch {}
    }

    // RTF Badge
    const rtf = Number(data.rtf);
    this.rtfBadge.textContent = `RTF: ${rtf.toFixed(2)}`;
    this.rtfBadge.className =
      rtf < 1.0 ? "badge badge-success" : "badge badge-experimental";

    // Telemetry Grid
    this.metricModel.textContent =
      data.model_name || "Qwen3-TTS-12Hz-1.7B-Base";
    this.metricLatency.textContent = `${data.generation_time_ms} ms (${(data.generation_time_ms / 1000).toFixed(2)}s)`;
    this.metricDuration.textContent = `${data.generated_audio_duration_seconds}s`;
    this.metricRtf.textContent = `${rtf.toFixed(2)} (Time / Duration)`;
    this.metricVram.textContent = `${data.gpu_memory_allocated_mb || 0} MB / ${data.gpu_memory_reserved_mb || 0} MB`;
    if (data.gpu_name) {
      this.metricDevice.textContent = `${data.gpu_name} (CUDA)`;
    }

    this.resultsCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ── UI Helpers ─────────────────────────────────────────────────────────

  setLoading(loading) {
    this.btnGenerate.disabled = loading;
    if (loading) {
      this.btnText.textContent = "Synthesizing Cloned Voice...";
      this.btnSpinner.hidden = false;
    } else {
      this.btnText.textContent = "Generate Cloned Speech";
      this.btnSpinner.hidden = true;
    }
  }

  showError(msg) {
    this.errorCard.hidden = false;
    this.errorMsg.textContent = msg;
  }

  hideError() {
    this.errorCard.hidden = true;
    this.errorMsg.textContent = "";
  }
}
