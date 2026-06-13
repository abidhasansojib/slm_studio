# SLM Studio: Small Language Model Engine

SLM Studio is a professional-grade suite for training, converting, and serving Small Language Models (SLMs) directly on Android devices via Termux. I have optimized it for ARM64 architecture and provided a dual-engine approach: PyTorch for training/experimentation and llama.cpp (GGUF) for high-performance inference.

---

## ⚠️ CRITICAL HARDWARE WARNING
**RAM Management & Stability:**
*   **PyTorch/Safetensors Models:** Running inference or training using raw `.pt` or `.safetensors` files is **extremely RAM-intensive**. On most Android devices, this can lead to system-level crashes or the Android OOM (Out-of-Memory) killer terminating the app.
*   **Recommendation:** For stable chat interactions, I highly recommend converting your trained models to **GGUF format**. GGUF supports quantization and memory mapping, which drastically reduces RAM usage and prevents crashes.

---

## 📚 Chapter 1: Environment Setup

Before starting, ensure you have Termux installed. Open Termux and run the setup script:

```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

**What this does:**
1. Updates your system and installs the `clang` compiler toolchain.
2. Adds the **TUR (Termux User Repository)** for `llama-cpp` support.
3. Installs pre-compiled **Numpy** and **Torch** (much faster than pip installs).
4. Installs **Flask** (for the Web UI) and **GGUF** tools.

---

## 📂 Chapter 2: Data Acquisition

To train a model, you need a corpus. I have provided a script to download a high-quality "Tiny Stories" and encyclopedic dataset (approx. 500MB+).

```bash
python download_data.py
```
*   Data is stored in `data_corpus/`.
*   You can add your own `.txt` files to this folder; the trainer will automatically ingest them.

---

## 🚀 Chapter 3: The Web Dashboard

The heart of SLM Studio is the Web UI. It provides real-time training metrics, a configuration editor, and a dual-engine chat interface.

**Start the server:**
```bash
python server.py
```
**Accessing the UI:**
*   Open your mobile browser and go to: `http://localhost:5000`
*   If you are on the same Wi-Fi, you can access it from a PC using the Android device's IP address.

---

## 🧠 Chapter 4: Training Your Model

You can train the **Native SLM** (a custom Transformer architecture) directly from the "Train" tab in the Web UI.

1.  **Configure Architecture:** Set `n_layer`, `n_embd`, and `n_head`. (Default: 50M parameters).
2.  **Set Hyperparameters:** Adjust Learning Rate, Batch Size, and Max Iterations.
3.  **Start Training:** Click "Start Training".
4.  **Monitoring:** The UI shows real-time Loss Graphs and Step Logs.
5.  **Checkpoints:** I have ensured the system auto-saves `checkpoint.tar` and `model.pt` in `models/native_slm/`.

---

## 💬 Chapter 5: Chat & Inference

The "Chat" tab allows you to talk to your models.

### Dual-Engine Support:
1.  **PyTorch Engine:** Uses the trained `model.pt`.
    *   *Pros:* Instant testing of new weights.
    *   *Cons:* High RAM usage, slower on mobile.
2.  **GGUF Engine (Recommended):** Uses `llama-server` in the background.
    *   *Pros:* Extremely fast, low RAM, hardware accelerated.
    *   *Cons:* Requires conversion first.

---

## 🛠 Chapter 6: Model Conversion (The GGUF Workflow)

To get the best performance, you should convert your trained PyTorch model to GGUF. This process reduces the model size and makes it much faster.

### Example Walkthrough:

**1. Convert your PyTorch Checkpoint to Safetensors:**
Safetensors is a faster, safer format for model weights. I have provided a script that aligns the data for maximum efficiency.
```bash
# This will look for models/native_slm/model.pt and create model.safetensors
python convert_to_safetensors.py
```

**2. Convert to GGUF format:**
Once you have your weights (either `.pt` or `.safetensors`), you can convert them to GGUF, which is the gold standard for mobile inference.
```bash
# This generates models/native_slm/model.gguf
python convert_to_gguf.py
```

**3. Use in the Web UI:**
After running these commands, go to the **Models** tab in the Web UI. You will see `model.gguf` in the list. Select it, and the "Chat" tab will now use the high-performance GGUF engine!

---

## 📥 Chapter 7: Downloading External Models

You can download GGUF or Safetensors models from Hugging Face:

1.  Go to the **Models** tab in the Web UI.
2.  Enter a Hugging Face Repo ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct-GGUF`).
3.  The system will scan for `.gguf` files and let you download them directly to your device.

---

## 📝 Feature Highlights

*   **Robust Tokenization:** I have included a custom `ByteTokenizer` for native training and a `QwenTokenizer` for external models.
*   **Prefix Stripping:** I have implemented a system that automatically handles varying state-dict prefixes (`model.`, `transformer.`, etc.) when loading external weights.
*   **Incremental Decoding:** I added logic to prevent text corruption in GGUF streams by handling partial UTF-8 characters.
*   **Checkpoint Resumption:** You can stop training anytime and resume exactly where you left off.

---

## 🛠 Troubleshooting

*   **Server won't start:** Ensure no other app is using port 5000.If running then close termux completely and then reopen and try runing again
*   **Training is slow:** Reduce `batch_size` or `n_embd`. Mobile CPUs have thermal limits.
*   **GGUF Chat Error:** Ensure `llama-cpp` is installed via `pkg`. If the package is missing, the server will log a warning.

---

## 💎 Credits

### Made by
<a href="https://t.me/abidhasansojib">
  <img src="https://img.shields.io/badge/Telegram-Abid_Hasan_Sojib-blue?style=for-the-badge&logo=telegram&logoColor=white" alt="Abid Hasan Sojib Telegram">
</a>
<a href="https://deepmind.google/technologies/gemini/">
  <img src="https://img.shields.io/badge/Powered_by-Gemini-orange?style=for-the-badge&logo=google-gemini&logoColor=white" alt="Gemini Logo">
</a>
