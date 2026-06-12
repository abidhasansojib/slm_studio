import os
import time
import json
import threading
import subprocess
from queue import Queue
import torch
from flask import Flask, request, jsonify, Response, render_template

# Import core components exclusively from model.py
from model import (
    TermuxSLM, 
    ByteTokenizer, 
    get_device, 
    load_config, 
    save_config,
    DEFAULT_CONFIG
)

# Graceful conditional import for GGUF handling
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Initialize Flask
app = Flask(__name__, template_folder='.')

# --- 1. SUBPROCESS & MODEL MANAGER ---
class ExternalTrainingManager:
    def __init__(self):
        self.is_training = False
        self.train_process = None
        
        # Web UI metrics tracking
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.val_loss = 0.0
        self.losses_history = []
        self.val_losses_history = []
        self.logs = []
        
        self.device = get_device()
        self.tokenizer = ByteTokenizer()
        self.config = load_config()
        
        # Dual-engine state parameters
        self.model = None          # PyTorch Model Instance
        self.gguf_model = None     # Llama.cpp GGUF Model Instance
        self.model_type = None     # 'gguf' or 'pytorch'
        
        # Automatically establish suggested model storage directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.init_model()

    def init_model(self):
        """Scans for GGUF compiled models first, falling back to PyTorch .pt/.tar configurations if empty."""
        self.config = load_config()
        self.gguf_model = None
        self.model = None
        
        # 1. SCAN FOR GGUF MODELS FIRST
        gguf_files = [f for f in os.listdir(self.model_dir) if f.endswith('.gguf')]
        
        if gguf_files and LLAMA_CPP_AVAILABLE:
            selected_gguf = os.path.join(self.model_dir, gguf_files[0])
            self.log(f"Found GGUF model candidate inside '{self.model_dir}/': {gguf_files[0]}")
            try:
                self.gguf_model = Llama(
                    model_path=selected_gguf,
                    n_ctx=self.config.get("block_size", 128),
                    n_threads=self.config.get("num_threads", 2),
                    verbose=False
                )
                self.model_type = "gguf"
                self.log(f"[ENGINE SWITCH] Running active sessions via Llama.cpp GGUF Core Engine Engine.")
                return
            except Exception as e:
                self.log(f"Warning: Failed to load GGUF compilation layer ({e}). Attempting PyTorch paths...")

        elif gguf_files and not LLAMA_CPP_AVAILABLE:
            self.log("Warning: GGUF models detected, but 'llama-cpp-python' is not installed. Run 'pip install llama-cpp-python'.")

        # 2. PYTORCH FALLBACK ENGINE (.pt or .tar)
        self.model_type = "pytorch"
        self.model = TermuxSLM(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=self.config["n_embd"],
            n_head=self.config["n_head"],
            n_layer=self.config["n_layer"],
            block_size=self.config["block_size"]
        ).to(self.device)
        
        # Check standard deployment weights file
        if os.path.exists('native_slm.pt'):
            try:
                self.model.load_state_dict(torch.load('native_slm.pt', map_location=self.device))
                self.log("Loaded PyTorch model weights from 'native_slm.pt'")
                self.model.eval()
            except Exception as e:
                self.log(f"Warning: Could not load raw weights file: {e}. Checking backup structures...")
                self._load_tar_fallback()
        # Check training snapshot archive file
        elif os.path.exists('native_slm_checkpoint.tar'):
            self._load_tar_fallback()
        else:
            self.log("No compiled GGUF, 'native_slm.pt', or 'native_slm_checkpoint.tar' found. Ready for fresh training initialization.")

    def _load_tar_fallback(self):
        """Helper to parse state parameters straight out of continuous training snapshot objects."""
        if os.path.exists('native_slm_checkpoint.tar'):
            try:
                checkpoint = torch.load('native_slm_checkpoint.tar', map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.log("Loaded PyTorch backup parameters successfully out of 'native_slm_checkpoint.tar'.")
                self.model.eval()
            except Exception as tar_err:
                self.log(f"Warning: Initialization error loading snapshot object: {tar_err}. Running random state weights.")
        else:
            self.log("No backup architecture maps accessible. Running random initialization layout.")

    def log(self, message):
        """Appends a log line to the server console and web logger dashboard."""
        timestamp = time.strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message.strip()}"
        self.logs.append(log_line)
        print(log_line)
        if len(self.logs) > 300:
            self.logs.pop(0)

    def start_training(self, new_config=None):
        """Updates configurations and executes model.py as an isolated pipeline."""
        if self.is_training:
            return False, "Training is already in progress."
            
        if new_config:
            # Check if network layout scaling rules changed
            arch_keys = ["n_embd", "n_head", "n_layer", "block_size"]
            arch_changed = False
            for k in arch_keys:
                if k in new_config and int(new_config[k]) != self.config[k]:
                    arch_changed = True
            
            # Write adjustments directly to model_config.json
            for k, v in new_config.items():
                if k in self.config:
                    if isinstance(DEFAULT_CONFIG[k], int):
                        self.config[k] = int(v)
                    elif isinstance(DEFAULT_CONFIG[k], float):
                        self.config[k] = float(v)
            save_config(self.config)
            
            if arch_changed:
                self.log("Network dimensions modified. Resetting checkpoint layers to prevent mismatch...")
                if os.path.exists('native_slm.pt'):
                    try: os.remove('native_slm.pt')
                    except: pass
                if os.path.exists('native_slm_checkpoint.tar'):
                    try: os.remove('native_slm_checkpoint.tar')
                    except: pass
                self.init_model()
        
        self.is_training = True
        self.current_step = 0
        self.total_steps = self.config["max_iters"]
        self.losses_history = []
        self.val_losses_history = []
        
        # Spawn execution tracking background worker
        threading.Thread(target=self._subprocess_monitor_worker, daemon=True).start()
        return True, "Subprocess pipeline initialized."

    def stop_training(self):
        """Terminates the running model.py background process gracefully."""
        if not self.is_training or not self.train_process:
            return False, "Training is not active."
        try:
            self.train_process.terminate()
            self.log("Termination signal dispatched to model.py pipeline process.")
            return True, "Process stop sequence executed."
        except Exception as e:
            return False, f"Failed to terminate process: {e}"

    def _subprocess_monitor_worker(self):
        """Executes model.py using unbuffered python streams and parses real-time output logs."""
        try:
            cmd = ["python3", "-u", "model.py"]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            self.train_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )
            self.log("Successfully spawned 'python3 -u model.py' background session.")

            while True:
                line = self.train_process.stdout.readline()
                if not line and self.train_process.poll() is not None:
                    break
                if line:
                    clean_line = line.strip()
                    self.log(clean_line)
                    self._parse_metrics(clean_line)

            rc = self.train_process.returncode
            self.log(f"External training session finished with exit code: {rc}")
            
            # Recheck and reload weight parameters immediately upon processing completion
            self.init_model()
            self.log("Web server memory layer matrices refreshed.")

        except Exception as e:
            self.log(f"Subprocess supervisor error: {e}")
        finally:
            self.is_training = False
            self.train_process = None

    def _parse_metrics(self, line):
        """Parses lines like: Step  100/1000 | Train Loss: 0.5041 | Val Loss: 1.6554"""
        if "Step" in line and "Train Loss:" in line:
            try:
                parts = [p.strip() for p in line.split("|")]
                step_str = parts[0].replace("Step", "").split("/")[0].strip()
                self.current_step = int(step_str)
                
                loss_str = parts[1].replace("Train Loss:", "").strip()
                self.current_loss = float(loss_str)
                self.losses_history.append({"step": self.current_step, "loss": self.current_loss})
                
                if len(parts) > 2 and "Val Loss:" in parts[2]:
                    val_str = parts[2].replace("Val Loss:", "").strip()
                    self.val_loss = float(val_str)
                    self.val_losses_history.append({"step": self.current_step, "loss": self.val_loss})
            except Exception:
                pass

# Instantiate global manager
training_manager = ExternalTrainingManager()

# --- 2. REST API ENDPOINTS ---

@app.route('/api/status', methods=['GET'])
def get_status():
    if training_manager.model_type == "gguf":
        param_count = "N/A (GGUF Binary Compiled)"
        device_info = "Llama.cpp Engine (CPU/Hardware-Bound)"
    else:
        param_count = sum(p.numel() for p in training_manager.model.parameters() if p.requires_grad) if training_manager.model else 0
        device_info = training_manager.device.upper()
    
    try:
        threads = torch.get_num_threads() if training_manager.model_type == "pytorch" else training_manager.config.get("num_threads", 4)
    except:
        threads = training_manager.config.get("num_threads", 2)
        
    return jsonify({
        "is_training": training_manager.is_training,
        "current_step": training_manager.current_step,
        "total_steps": training_manager.total_steps,
        "current_loss": training_manager.current_loss,
        "val_loss": training_manager.val_loss,
        "losses_history": training_manager.losses_history,
        "val_losses_history": training_manager.val_losses_history,
        "logs": training_manager.logs[-80:], 
        "device": device_info,
        "param_count": param_count,
        "threads": threads,
        "config": training_manager.config,
        "active_engine": training_manager.model_type
    })

@app.route('/api/train/start', methods=['POST'])
def start_train():
    data = request.json or {}
    success, msg = training_manager.start_training(data)
    if success:
        return jsonify({"message": msg, "status": "success"})
    return jsonify({"error": msg, "status": "error"}), 400

@app.route('/api/train/stop', methods=['POST'])
def stop_train():
    success, msg = training_manager.stop_training()
    if success:
        return jsonify({"message": msg, "status": "success"})
    return jsonify({"error": msg, "status": "error"}), 400

@app.route('/api/reset', methods=['POST'])
def reset_model_route():
    if training_manager.is_training:
        return jsonify({"error": "Cannot wipe weights while training loop is active.", "status": "error"}), 400
    if training_manager.model_type == "gguf":
        return jsonify({"error": "Cannot wipe weights while running an immutable GGUF binary model. Remove the file from the models directory instead.", "status": "error"}), 400
        
    training_manager.log("Resetting all parameters to random initial distribution values...")
    if os.path.exists('native_slm.pt'):
        try: os.remove('native_slm.pt')
        except: pass
    if os.path.exists('native_slm_checkpoint.tar'):
        try: os.remove('native_slm_checkpoint.tar')
        except: pass
    training_manager.init_model()
    return jsonify({"message": "Model reset completed. Weights cleared.", "status": "success"})

@app.route('/api/dataset', methods=['GET', 'POST'])
def handle_dataset():
    dataset_path = 'training_data.txt'
    if request.method == 'GET':
        if not os.path.exists(dataset_path):
            return jsonify({"text": ""})
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                content = f.read(100000)
            return jsonify({"text": content, "truncated": os.path.getsize(dataset_path) > 100000})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        data = request.json or {}
        text = data.get('text', '')
        try:
            with open(dataset_path, 'w', encoding='utf-8') as f:
                f.write(text)
            return jsonify({"message": "Dataset saved successfully.", "status": "success"})
        except Exception as e:
            return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if training_manager.is_training:
        return jsonify({"error": "Model training process is running. Please pause training to activate chat interactions."}), 400

    data = request.json or {}
    prompt = data.get('prompt', '')
    temperature = float(data.get('temperature', 0.8))
    top_k = data.get('top_k')
    if top_k is not None:
        top_k = int(top_k)
    max_tokens = int(data.get('max_tokens', 150))
    stream = data.get('stream', False)

    if not prompt:
        return jsonify({"error": "Empty prompt provided."}), 400

    # --- ROUTE A: INTERACTIVE SELECTION VIA GGUF CORE ENGINE ---
    if training_manager.model_type == "gguf":
        if not stream:
            start_time = time.time()
            output = training_manager.gguf_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k if top_k is not None else 40
            )
            elapsed = time.time() - start_time
            response_text = output['choices'][0]['text']
            tokens_gen = output['usage']['completion_tokens']
            speed = tokens_gen / (elapsed + 1e-5)
            
            return jsonify({
                "response": response_text,
                "tokens_generated": tokens_gen,
                "elapsed_seconds": elapsed,
                "speed_tokens_sec": speed
            })
        else:
            def generate_gguf_sse():
                start_time = time.time()
                response_stream = training_manager.gguf_model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k is not None else 40,
                    stream=True
                )
                tokens_streamed = 0
                for chunk in response_stream:
                    token_text = chunk['choices'][0]['text']
                    tokens_streamed += 1
                    yield f"data: {json.dumps({'token': token_text})}\n\n"
                    
                elapsed = time.time() - start_time
                yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_streamed, 'elapsed_seconds': elapsed, 'speed_tokens_sec': tokens_streamed / (elapsed + 1e-5)})}\n\n"
            
            return Response(generate_gguf_sse(), mimetype='text/event-stream')

    # --- ROUTE B: PYTORCH PARSING PATHWAY FALLBACK ---
    else:
        prompt_tokens = training_manager.tokenizer.encode(prompt, bos=True, eos=False)
        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=training_manager.device)

        if not stream:
            start_time = time.time()
            with torch.no_grad():
                generated_ids = training_manager.model.generate(
                    input_tensor,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
            elapsed = time.time() - start_time
            full_text = training_manager.tokenizer.decode(generated_ids)
            response_text = full_text[len(prompt):]
            tokens_gen = len(generated_ids) - len(prompt_tokens)
            speed = tokens_gen / (elapsed + 1e-5)
            
            return jsonify({
                "response": response_text,
                "tokens_generated": tokens_gen,
                "elapsed_seconds": elapsed,
                "speed_tokens_sec": speed
            })
        else:
            def generate_sse():
                q = Queue()
                def token_callback(token_id): q.put(token_id)
                    
                def worker():
                    try:
                        training_manager.model.generate(
                            input_tensor,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            callback=token_callback
                        )
                    except Exception as e:
                        print(f"Streaming error: {e}")
                    finally:
                        q.put(None)
                
                threading.Thread(target=worker, daemon=True).start()
                start_time = time.time()
                tokens_streamed = 0
                
                while True:
                    token_id = q.get()
                    if token_id is None:
                        break
                    tokens_streamed += 1
                    yield f"data: {json.dumps({'token': training_manager.tokenizer.decode([token_id])})}\n\n"
                    
                elapsed = time.time() - start_time
                yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_streamed, 'elapsed_seconds': elapsed, 'speed_tokens_sec': tokens_streamed / (elapsed + 1e-5)})}\n\n"

            return Response(generate_sse(), mimetype='text/event-stream')

@app.route('/')
def index_route():
    return render_template('index.html')

if __name__ == "__main__":
    print("--------------------------------------------------")
    print("   Native Termux SLM Server (Dual-Engine Monitor) ")
    print("--------------------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
