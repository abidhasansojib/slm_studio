# 🧠 Mobile SLM Studio: On-Device Small Language Model

An advanced, light-weight, and fully-featured Small Language Model (SLM) platform designed to train and run inference directly on resource-constrained mobile hardware (CPUs/GPUs) via **Termux** on Android (or local macOS/Linux systems).

This studio provides both a standalone command-line interface and a premium, responsive Web-based Chatbot and Training Dashboard.

---

## 🚀 Key Features

* **Modern LLM Architecture**: Built on a causal transformer decoder structure identical to state-of-the-art models like **Llama 3** and **Gemma**.
  * **Rotary Position Embeddings (RoPE)**: Encodes relative token positions dynamically for better long-range context mapping.
  * **RMSNorm**: Pre-normalization layers for fast, stable gradient convergence.
  * **SwiGLU Activation**: Leverages gated linear units with SiLU activation in the feed-forward blocks for improved representation learning.
  * **KV-Caching**: Maintains key-value caches during generation, resulting in ultra-fast, step-by-step local generation on mobile CPUs.
* **100% Robust Byte-Level Tokenizer**: Uses raw UTF-8 bytes (vocabulary size 259) to guarantee zero out-of-vocabulary (OOV) errors. Can tokenize and generate text in any language, symbols, or code.
* **Premium Web Dashboard**: A futuristic, single-page application built with glassmorphism aesthetics, responsive styling, and dynamic animations.
  * **Real-time SSE Token Streaming**: Chat responses stream byte-by-byte in real time.
  * **Live Learning Curves**: Interactive graphs (via Chart.js) mapping training loss and validation loss progress.
  * **Dataset Manager**: In-browser text editor to modify and save the `training_data.txt` corpus.
  * **Interactive Configurator**: Reconfigure network hyper-parameters (dimensions, heads, layers, block size) and reset weights directly from the web interface.

---

## 🛠️ Termux Setup & Installation

To run this model on Android via Termux, run the following commands to install Python, compilers, and PyTorch:

```bash
# 1. Update system packages
apt update && apt upgrade -y

# 2. Install Python and compilation tools
apt install python clang make ndk-sysroot build-essential -y

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch & Flask
# (Note: PyTorch installation in Termux can sometimes take a few minutes)
pip install torch flask
```

---

## 🖥️ How to Run

### Option 1: Launch the Web Chat & Training Dashboard (Recommended)
Launch the Flask-based server to access the premium graphical dashboard:
```bash
python server.py
```
After launching, open your browser and navigate to:
* **Local (on phone)**: [http://localhost:5000](http://localhost:5000)
* **Local Network (LAN)**: Locate your phone's IP address (e.g., `192.168.1.15`) and visit `http://192.168.1.15:5000` from any device connected to the same Wi-Fi.

### Option 2: Train via Command Line Interface (CLI)
You can train and test the model directly inside the terminal:
```bash
python model.py
```

---

## 📖 Architecture & Configuration

The model architecture hyper-parameters can be customized in the Web UI or edited inside the auto-generated `model_config.json` file:

| Hyperparameter | Default Value | Description |
| :--- | :--- | :--- |
| `n_embd` | `128` | Embedding dimensions ($d_{model}$) |
| `n_head` | `4` | Number of Attention Heads (must divide `n_embd`) |
| `n_layer` | `4` | Number of sequential Transformer Blocks |
| `block_size` | `128` | Context window size (maximum input sequence length) |
| `learning_rate`| `2e-3` | Step rate for AdamW Optimizer |
| `batch_size` | `32` | Number of sequences processed in parallel |
| `num_threads` | `4` | Hardware threads to pin (prevents phone lag) |

### Memory & Performance Recommendation for Mobile
For a smooth experience on midrange mobile CPUs (e.g., Snapdragon 600/700 series), we recommend:
* Parameter Size: **0.8M to 5M parameters** (Default: ~830K parameters).
* This provides the full architectural capabilities of large-scale models, but optimizes training speed to less than 1 second per step, and yields instantaneous inference generation.

---

## 📁 File Structure

* [model.py](file:///data/data/com.termux/files/home/termux_slm/model.py): Core mathematical modules (RMSNorm, RoPE, SwiGLU, Attention, KV Cache), Byte Tokenizer, and CLI training code.
* [server.py](file:///data/data/com.termux/files/home/termux_slm/server.py): Flask backend API, background thread-safe training manager, SSE token generator, and single-page web studio.
* [training_data.txt](file:///data/data/com.termux/files/home/termux_slm/training_data.txt): The dataset corpus (defaults to Shakespeare’s plays) used to train the model.
* `model_config.json`: Persistent model architecture settings (auto-generated).
* `native_slm.pt`: Saved binary model weights checkpoint.
