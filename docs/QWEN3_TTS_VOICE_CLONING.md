# Qwen3-TTS 1.7B Base Voice Cloning Guide & Benchmark

This document provides a guide for evaluating and benchmarking arbitrary speaker voice cloning using **`Qwen/Qwen3-TTS-12Hz-1.7B-Base`** inside the `speech-to-speech` framework.

---

## 1. Overview & Architecture

Unlike `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (which only synthesizes fixed preset speaker identities like Aiden, Ryan, etc.), **`Qwen/Qwen3-TTS-12Hz-1.7B-Base`** allows zero-shot **arbitrary speaker voice cloning** from any short reference audio clip:

```text
Reference Voice Audio (.wav / .mp3)
                ↓
Speaker Embedding / Acoustic Codes (x-vector extraction / ICL prompt)
                ↓
    Qwen3-TTS 1.7B Base Model
                ↓
    Target Text (e.g. English, Khmer [Experimental])
                ↓
    Generated Voice Preserving Reference Speaker Characteristics
```

---

## 2. Installation & Hardware Requirements

### Dependencies
The voice cloning module uses the repository's existing virtual environment and dependencies:
```bash
# In the repository root:
uv sync
```
Key packages utilized:
* `faster-qwen3-tts>=0.3.2` (on Linux / Windows with CUDA or GGML)
* `torch>=2.4.0` / `torchaudio>=2.4.0`
* `soundfile>=0.13.0` & `scipy>=1.10.0`
* `fastapi>=0.115.0` & `python-multipart`

### Hardware
* **NVIDIA GPU**: Tesla T4 (or any CUDA GPU with >= 8GB VRAM). In BF16/FP16, the 1.7B model consumes ~3.5–4.5 GB of GPU VRAM.
* **Apple Silicon**: Automatically uses the `mlx-audio` backend with mapped 6-bit/BF16 weights.
* **CPU**: Fallback supported if CUDA is unavailable.

---

## 3. Voice Cloning Modes

### A. Speaker / X-Vector Only Mode (`x_vector_only_mode=True`)
* **How it works**: The acoustic speaker embedding (x-vector) is extracted directly from the reference audio waveform.
* **Transcript Required**: **No**.
* **Advantages**:
  * Clean audio start without needing manual reference transcriptions.
  * More robust when synthesizing cross-lingual speech (e.g., English voice speaking Khmer).
* **Usage**:
  ```python
  handler = Qwen3TTSHandler(
      stop_event,
      queue_in=queue_in,
      queue_out=queue_out,
      setup_args=(should_listen,),
      setup_kwargs={
          "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
          "ref_audio": "path/to/reference_voice.wav",
          "xvec_only": True,
          "language": "en",
      },
  )
  ```

### B. Reference Transcript Mode (In-Context Learning / ICL)
* **How it works**: Uses both the reference audio acoustic tokens and its transcript as an in-context learning prefix for generation.
* **Transcript Required**: **Yes** (must match what is spoken in the reference audio).
* **Characteristics**: Can capture fine-grained cadence and speaking style when the language matches, but sensitive to transcript accuracy.

---

## 4. Supported vs. Experimental Languages

| Category | Languages | Status / Notes |
| :--- | :--- | :--- |
| **Officially Supported** | English (`en`), Chinese (`zh`), Japanese (`ja`), Korean (`ko`), German (`de`), French (`fr`), Russian (`ru`), Portuguese (`pt`), Spanish (`es`), Italian (`it`) | Full tokenizer & acoustic support in Qwen3-TTS base pretraining. |
| **Experimental / Unsupported** | **Khmer (`km`)**, Vietnamese (`vi`), Thai (`th`), Indonesian (`id`), Arabic (`ar`), Hindi (`hi`) | Unsupported by official tokenizers; evaluated experimentally to test cross-lingual speaker characteristic preservation. |

> [!NOTE]
> When testing Khmer (`km`), use `x_vector_only_mode=True` with an English reference voice to observe whether tone and timbre carry over.

---

## 5. Running the Voice Cloning & Benchmark UI

Start the web application:
```bash
# From the repository root:
.venv/bin/python demo/server.py
```
or
```bash
speech-to-speech serve
```

Then open `http://localhost:8080` in your browser.

### Test Workflow:
1. Click the **Voice Cloning & Benchmark** button (microphone icon) in the top-right header.
2. In the modal:
   * Select **STT Provider** (e.g. Parakeet-TDT).
   * Select **LLM Provider** (e.g. Chat Completions).
   * Select **TTS Provider**: `Qwen3-TTS Base (1.7B Arbitrary Voice Cloning)`.
3. In the **Voice Cloning** section:
   * Upload a `.wav` or `.mp3` reference voice file (or pick `ref_audio.wav`).
   * Listen to the **Reference Audio** preview.
   * Choose **Cloning Mode**: `Speaker / X-Vector Only` (recommended) or `Reference Transcript`.
   * Select **Language**: `English (Official)` or `Khmer (Experimental)`.
   * Enter input text (or click the quick sample buttons: *English Sample* or *Khmer Sample*).
   * Click **Generate Cloned Speech**.
4. Evaluate:
   * Listen to **Reference Voice** vs **Generated Cloned Voice** side-by-side.
   * Review telemetry metrics:
     * **Generation Latency (ms)**
     * **Audio Duration (s)**
     * **Real-Time Factor (RTF = Time / Duration)**
     * **GPU VRAM Usage (MB)**
