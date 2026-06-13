import os
import time
import json
import threading
import subprocess
import shutil
import signal
import sys
from queue import Queue
import torch
from flask import Flask, request, jsonify, Response, render_template

# Import core components exclusively from model.py
from model import (
    TermuxSLM, 
    Qwen3Model,
    ByteTokenizer, 
    QwenTokenizer,
    get_device, 
    load_config, 
    save_config,
    load_safetensors,
    DEFAULT_CONFIG
)

# Initialize Flask
app = Flask(__name__, template_folder='.')

def list_available_models():
    model_dir = "models"
    if not os.path.exists(model_dir):
        return []
    
    models = []
    
    # 1. Search for direct GGUF files in models/
    for entry in os.scandir(model_dir):
        if entry.is_file() and entry.name.endswith(".gguf"):
            models.append({
                "name": entry.name,
                "path": "models",
                "weights_file": entry.name,
                "type": "gguf",
                "config": {"model_type": "gguf", "parameters": "Quantized GGUF"}
            })
            
    # 2. Search for directories inside models/
    for entry in os.scandir(model_dir):
        if entry.is_dir():
            try:
                gguf_files = [f.name for f in os.scandir(entry.path) if f.is_file() and f.name.endswith(".gguf")]
            except:
                gguf_files = []
                
            if gguf_files:
                for gf in gguf_files:
                    name = f"{entry.name}/{gf}" if len(gguf_files) > 1 else entry.name
                    models.append({
                        "name": name,
                        "path": entry.path,
                        "weights_file": gf,
                        "type": "gguf",
                        "config": {"model_type": "gguf", "parameters": "Quantized GGUF"}
                    })
                continue
            
            # Check for standard configs and weights (PyTorch/Safetensors)
            config_path = os.path.join(entry.path, "config.json")
            if os.path.exists(config_path):
                weight_files = ["model.safetensors", "model.pt", "native_slm.pt", "pytorch_model.bin"]
                found_weights = None
                for wf in weight_files:
                    if os.path.exists(os.path.join(entry.path, wf)):
                        found_weights = wf
                        break
                
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                except:
                    cfg = {}
                
                is_qwen = "hidden_size" in cfg or "model_type" in cfg or "num_attention_heads" in cfg
                model_type = "qwen3" if is_qwen else "native_slm"
                
                models.append({
                    "name": entry.name,
                    "path": entry.path,
                    "weights_file": found_weights,
                    "type": model_type,
                    "config": cfg
                })
    return models

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
        self.tokenizer = None
        self.config = None
        self.model = None
        
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.is_gguf = False
        self.llama_process = None
        
        self.active_model_name = "native_slm"
        self.init_model()
        
        import atexit
        atexit.register(self.shutdown)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except: pass

    def _signal_handler(self, sig, frame):
        self.shutdown()
        sys.exit(0)

    def stop_llama_server(self):
        """Terminates the running llama-server background process gracefully."""
        if hasattr(self, 'llama_process') and self.llama_process:
            self.log("Stopping llama-server background process...")
            try:
                self.llama_process.terminate()
                # Wait up to 3 seconds for it to exit
                for _ in range(30):
                    if self.llama_process.poll() is not None:
                        break
                    time.sleep(0.1)
                else:
                    self.llama_process.kill()
                    self.llama_process.wait()
            except Exception as e:
                self.log(f"Error terminating llama-server: {e}")
            self.llama_process = None

        # Clean up any remaining processes on port 5001
        try:
            pids = subprocess.check_output(["lsof", "-t", "-i", ":5001"], stderr=subprocess.DEVNULL).decode().split()
            for p in pids:
                pid = int(p)
                if pid != os.getpid():
                    os.kill(pid, 9)
        except Exception:
            pass

    def shutdown(self):
        self.stop_training()
        self.stop_llama_server()
        os.system("pkill -9 -f llama-cli >/dev/null 2>&1 || true")
        os.system("pkill -9 -f llama-server >/dev/null 2>&1 || true")

    def init_model(self, model_name=None):
        """Initializes the PyTorch model or GGUF llama-server subprocess."""
        if not model_name:
            if os.path.exists("server_config.json"):
                try:
                    with open("server_config.json", "r") as f:
                        sc = json.load(f)
                        model_name = sc.get("active_model", "native_slm")
                except:
                    model_name = "native_slm"
            else:
                model_name = "native_slm"
        
        self.active_model_name = model_name
        self.log(f"Initializing model '{model_name}'...")
        
        models = list_available_models()
        model_info = next((m for m in models if m["name"] == model_name), None)
        
        if not model_info:
            if model_name == "native_slm":
                model_dir = os.path.join(self.model_dir, "native_slm")
                os.makedirs(model_dir, exist_ok=True)
                model_info = {
                    "name": "native_slm",
                    "path": model_dir,
                    "weights_file": "model.pt" if os.path.exists(os.path.join(model_dir, "model.pt")) else None,
                    "type": "native_slm",
                    "config": load_config(os.path.join(model_dir, "config.json"))
                }
            else:
                self.log(f"Model '{model_name}' not found. Falling back to native_slm...")
                self.init_model("native_slm")
                return

        # Always stop any running llama-server first
        self.stop_llama_server()
        self.is_gguf = False

        try:
            model_dir = model_info["path"]
            
            if model_info["type"] == "gguf":
                self.is_gguf = True
                self.config = load_config()
                if model_info.get("config"):
                    self.config.update(model_info["config"])
                self.tokenizer = None
                self.model = None
                
                # Unload PyTorch weights from Python memory to free RAM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                weights_path = os.path.join(model_dir, model_info["weights_file"]) if model_dir != "models" else os.path.join("models", model_info["weights_file"])
                self.log(f"Active model is a quantized GGUF. Starting C++ engine llama-server...")
                
                # Check and clean port 5001 if already bound
                try:
                    import socket
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(("127.0.0.1", 5001))
                    s.close()
                    self.log("[!] Port 5001 is already in use. Cleaning up...")
                    os.system("fuser -k 5001/tcp >/dev/null 2>&1 || true")
                    time.sleep(0.5)
                except Exception:
                    pass
                
                # Command to launch llama-server
                # Limit threads to 4 to prevent UI starvation, specify standard context size 2048
                cmd = [
                    "llama-server",
                    "-m", weights_path,
                    "-c", "2048",
                    "-t", "4",
                    "--port", "5001",
                    "--host", "127.0.0.1"
                ]
                
                self.log(f"Running command: {' '.join(cmd)}")
                self.llama_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Spawn non-blocking background reader thread to prevent pipe buffer from freezing
                def log_reader(process, manager):
                    try:
                        for line in iter(process.stdout.readline, ''):
                            if not line:
                                break
                            manager.log(f"[llama-server] {line.strip()}")
                    except Exception as e:
                        manager.log(f"[llama-server Reader Error] {e}")
                    finally:
                        process.stdout.close()
                
                threading.Thread(target=log_reader, args=(self.llama_process, self), daemon=True).start()
                
                # Verify server startup and wait for model loading to complete (preventing HTTP 503 Service Unavailable errors)
                self.log("Waiting for llama-server to initialize and load model weights...")
                ready = False
                for i in range(120):  # Try for up to 60 seconds (useful for slower mobile CPUs/storage)
                    time.sleep(0.5)
                    if self.llama_process.poll() is not None:
                        raise RuntimeError("llama-server terminated unexpectedly during startup.")
                    
                    try:
                        import urllib.request
                        req = urllib.request.Request("http://127.0.0.1:5001/health")
                        with urllib.request.urlopen(req, timeout=1.0) as response:
                            if response.status == 200:
                                res_data = json.loads(response.read().decode('utf-8'))
                                if res_data.get("status") == "ok":
                                    ready = True
                                    break
                    except Exception:
                        pass
                
                if not ready:
                    raise RuntimeError("llama-server failed to become ready within 60 seconds.")
                
                self.log("llama-server successfully started and model is fully loaded on port 5001!")
                with open("server_config.json", "w") as f:
                    json.dump({"active_model": model_name}, f)
                return
                
            # --- Standard PyTorch/Safetensors Loading ---
            config_path = os.path.join(model_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            else:
                self.config = load_config()
                
            weights_file = model_info.get("weights_file")
            tar_path = os.path.join(model_dir, 'checkpoint.tar')
            
            # Load state_dict first to determine precision and avoid OOM by pre-casting model
            state_dict = None
            if weights_file:
                weights_path = os.path.join(model_dir, weights_file)
                self.log(f"Loading weights from '{weights_path}'...")
                try:
                    if weights_file.endswith(".safetensors"):
                        state_dict = load_safetensors(weights_path, device=self.device)
                    else:
                        state_dict = torch.load(weights_path, map_location=self.device)
                        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']
                except Exception as e:
                    self.log(f"[-] Error reading weights file: {e}")
            elif os.path.exists(tar_path):
                self.log(f"Loading native weights from backup checkpoint '{tar_path}'...")
                try:
                    state_dict = torch.load(tar_path, map_location=self.device)
                    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                except Exception as e:
                    self.log(f"[-] Error reading backup checkpoint: {e}")
            
            # Auto-detect precision from weights (FP32 vs BF16/FP16)
            model_dtype = torch.float32
            if state_dict:
                for tensor in state_dict.values():
                    if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.bfloat16, torch.float16]:
                        model_dtype = torch.bfloat16
                        break
            
            self.log(f"Configuring model layers with precision: {model_dtype}")
            
            if model_info["type"] == "qwen3":
                self.tokenizer = QwenTokenizer(model_dir)
                self.model = Qwen3Model(self.config).to(model_dtype).to(self.device)
            else:
                self.tokenizer = ByteTokenizer()
                self.model = TermuxSLM(
                    vocab_size=self.tokenizer.vocab_size,
                    n_embd=self.config.get("n_embd", 256),
                    n_head=self.config.get("n_head", 8),
                    n_layer=self.config.get("n_layer", 6),
                    block_size=self.config.get("block_size", 384)
                ).to(model_dtype).to(self.device)
                
            if state_dict:
                m_keys = set(self.model.state_dict().keys())
                for prefix in ["model.", "transformer.", "llm."]:
                    if not any(k in m_keys for k in state_dict.keys()):
                        state_dict = { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }
                
                # Cast state_dict tensors to model_dtype to ensure clean loading
                state_dict = {k: v.to(model_dtype) for k, v in state_dict.items() if k in m_keys}
                
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.log(f"Successfully loaded '{model_name}' weights!")
            else:
                self.log("Ready for fresh training initialization.")
            
            # Save selection to server_config.json
            with open("server_config.json", "w") as f:
                json.dump({"active_model": model_name}, f)
                
        except Exception as e:
            import traceback
            self.log(f"Error loading model '{model_name}': {e}")
            self.log(traceback.format_exc())
            if model_name != "native_slm":
                self.log("Falling back to native_slm...")
                self.init_model("native_slm")

    def log(self, message, **kwargs):
        """Appends a log line to the server console and web logger dashboard."""
        msg_str = str(message)
        
        # Handle progress updates (starting with \r) to prevent UI log spam
        if msg_str.startswith('\r'):
            clean_msg = msg_str.replace('\r', '').strip()
            if clean_msg and self.logs:
                timestamp = time.strftime("%H:%M:%S")
                self.logs[-1] = f"[{timestamp}] {clean_msg}"
            elif clean_msg:
                timestamp = time.strftime("%H:%M:%S")
                self.logs.append(f"[{timestamp}] {clean_msg}")
            
            print(msg_str, end=kwargs.get('end', '\n'), flush=True)
            return

        if not msg_str.strip():
            print(msg_str, end=kwargs.get('end', '\n'), flush=True)
            return

        timestamp = time.strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {msg_str.strip()}"
        self.logs.append(log_line)
        print(log_line, end=kwargs.get('end', '\n'))
        
        if len(self.logs) > 300:
            self.logs.pop(0)

    def update_config(self, new_config):
        """Helper to update internal config and save to disk."""
        restart_needed = False
        if self.is_gguf:
            old_block_size = self.config.get("block_size")
            old_threads = self.config.get("num_threads")
            
        for k, v in new_config.items():
            if isinstance(v, bool):
                self.config[k] = bool(v)
            elif isinstance(v, (int, float)):
                self.config[k] = v
            elif isinstance(v, str) and v.isdigit():
                self.config[k] = int(v)
            else:
                try:
                    self.config[k] = float(v)
                except ValueError:
                    self.config[k] = v
                    
        if self.is_gguf:
            new_block_size = self.config.get("block_size")
            new_threads = self.config.get("num_threads")
            if old_block_size != new_block_size or old_threads != new_threads:
                restart_needed = True

        # Save config to the active model's directory
        model_dir = os.path.join(self.model_dir, self.active_model_name)
        if os.path.exists(model_dir):
            config_path = os.path.join(model_dir, "config.json")
            save_config(self.config, config_path)
        save_config(self.config, "model_config.json")
        
        if restart_needed:
            self.log("Configuration updated. Restarting llama-server with new settings...")
            self.init_model(self.active_model_name)

    def start_training(self, new_config=None):
        """Updates configurations and executes model.py as an isolated pipeline."""
        if self.is_gguf:
            return False, "Training is not supported for quantized GGUF models. Select the native_slm model to start training."
            
        if self.is_training:
            return False, "Training is already in progress."
            
        if new_config:
            # Check if network layout scaling rules changed
            arch_keys = ["n_embd", "n_head", "n_layer", "block_size"]
            arch_changed = False
            for k in arch_keys:
                if k in new_config and int(new_config[k]) != self.config.get(k):
                    arch_changed = True
            
            self.update_config(new_config)
            
            if arch_changed:
                self.log("Network dimensions modified. Resetting checkpoint layers to prevent mismatch...")
                model_dir = os.path.join(self.model_dir, "native_slm")
                pt_path = os.path.join(model_dir, 'model.pt')
                tar_path = os.path.join(model_dir, 'checkpoint.tar')
                if os.path.exists(pt_path):
                    try: os.remove(pt_path)
                    except: pass
                if os.path.exists(tar_path):
                    try: os.remove(tar_path)
                    except: pass
                self.init_model("native_slm")
        
        self.is_training = True
        self.current_step = 0
        self.total_steps = self.config.get("max_iters", 1000)
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
            os.system("pkill -f 'python3 -u model.py' >/dev/null 2>&1 || true")
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
        if "Step" in line and "Loss:" in line:
            try:
                parts = [p.strip() for p in line.split("|")]
                step_str = parts[0].replace("Step", "").split("/")[0].strip()
                self.current_step = int(step_str)
                
                loss_str = parts[1].split(":")[1].strip()
                self.current_loss = float(loss_str)
                self.losses_history.append({"step": self.current_step, "loss": self.current_loss})
                
                if len(parts) > 2 and "Val Loss:" in parts[2]:
                    val_str = parts[2].split(":")[1].strip()
                    self.val_loss = float(val_str)
                    self.val_losses_history.append({"step": self.current_step, "loss": self.val_loss})
            except Exception:
                pass

# Instantiate global manager
manager = ExternalTrainingManager()

# --- 2. REST API ENDPOINTS ---

@app.route('/api/status', methods=['GET'])
def get_status():
    param_count = sum(p.numel() for p in manager.model.parameters() if p.requires_grad) if manager.model else 0
    device_info = manager.device.upper()
    
    try:
        threads = torch.get_num_threads()
    except:
        threads = manager.config.get("num_threads", 2)
        
    return jsonify({
        "is_training": manager.is_training,
        "current_step": manager.current_step,
        "total_steps": manager.total_steps,
        "current_loss": manager.current_loss,
        "val_loss": manager.val_loss,
        "losses_history": manager.losses_history,
        "val_losses_history": manager.val_losses_history,
        "logs": manager.logs[-80:], 
        "device": device_info,
        "param_count": param_count,
        "threads": threads,
        "config": manager.config,
        "active_model": manager.active_model_name
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    models = list_available_models()
    # Ensure native_slm is always in the list even if folder is empty
    if not any(m["name"] == "native_slm" for m in models):
        native_dir = os.path.join(manager.model_dir, "native_slm")
        models.append({
            "name": "native_slm",
            "path": native_dir,
            "weights_file": "model.pt" if os.path.exists(os.path.join(native_dir, "model.pt")) else None,
            "type": "native_slm",
            "config": load_config(os.path.join(native_dir, "config.json"))
        })
    return jsonify({
        "models": models,
        "active_model": manager.active_model_name
    })

@app.route('/api/models/select', methods=['POST'])
def select_model():
    if manager.is_training:
        return jsonify({"error": "Cannot change models while training is in progress.", "status": "error"}), 400
    data = request.json or {}
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"error": "No model name specified.", "status": "error"}), 400
    
    manager.init_model(model_name)
    return jsonify({"message": f"Model '{model_name}' selected and loaded.", "status": "success"})

@app.route('/api/models/delete', methods=['POST'])
def delete_model():
    if manager.is_training:
        return jsonify({"error": "Cannot delete models while training is in progress.", "status": "error"}), 400
    data = request.json or {}
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"error": "No model name specified.", "status": "error"}), 400
    if model_name == "native_slm":
        return jsonify({"error": "Cannot delete the native baseline model.", "status": "error"}), 400
    
    if manager.active_model_name == model_name:
        # Switch back to native first
        manager.init_model("native_slm")
        
    models = list_available_models()
    info = next((m for m in models if m["name"] == model_name), None)
    if not info:
        return jsonify({"error": "Model not found.", "status": "error"}), 404
        
    try:
        target_path = os.path.abspath(info["path"])
        models_dir = os.path.abspath("models")
        if not target_path.startswith(models_dir):
            return jsonify({"error": "Invalid model path.", "status": "error"}), 400
            
        if os.path.isfile(target_path):
            os.remove(target_path)
            manager.log(f"Deleted GGUF model file: {target_path}")
        elif os.path.isdir(target_path):
            shutil.rmtree(target_path)
            manager.log(f"Deleted model folder: {target_path}")
            
        return jsonify({"message": f"Model '{model_name}' deleted.", "status": "success"})
    except Exception as e:
        return jsonify({"error": f"Failed to delete model: {e}", "status": "error"}), 500

@app.route('/api/models/download', methods=['POST'])
def download_model_route():
    data = request.json or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL or Repo ID specified.", "status": "error"}), 400
        
    def run_download():
        manager.log(f"Starting background download for: {url}")
        try:
            from download_model import download_model as dl_model
            success = dl_model(url, log_callback=manager.log)
            if success:
                manager.log(f"Download complete! Scanning and refreshing available models.")
            else:
                manager.log(f"Download failed for: {url}")
        except Exception as e:
            manager.log(f"Download exception occurred: {e}")
            
    threading.Thread(target=run_download, daemon=True).start()
    return jsonify({"message": "Download process started in background.", "status": "success"})

@app.route('/api/train/start', methods=['POST'])
def start_train():
    data = request.json or {}
    success, msg = manager.start_training(data)
    if success:
        return jsonify({"message": msg, "status": "success"})
    return jsonify({"error": msg, "status": "error"}), 400

@app.route('/api/train/stop', methods=['POST'])
def stop_train():
    success, msg = manager.stop_training()
    if success:
        return jsonify({"message": msg, "status": "success"})
    return jsonify({"error": msg, "status": "error"}), 400

@app.route('/api/reset', methods=['POST'])
def reset_model_route():
    if manager.is_training:
        return jsonify({"error": "Cannot wipe weights while training loop is active.", "status": "error"}), 400
        
    manager.log("Resetting all parameters to random initial distribution values...")
    model_dir = os.path.join(manager.model_dir, "native_slm")
    pt_path = os.path.join(model_dir, 'model.pt')
    tar_path = os.path.join(model_dir, 'checkpoint.tar')
    if os.path.exists(pt_path):
        try: os.remove(pt_path)
        except: pass
    if os.path.exists(tar_path):
        try: os.remove(tar_path)
        except: pass
    manager.init_model("native_slm")
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

@app.route('/api/config/save', methods=['POST'])
def save_config_route():
    data = request.json or {}
    manager.update_config(data)
    return jsonify({"message": "Configuration saved successfully.", "status": "success"})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if manager.is_training:
        return jsonify({"error": "Model training process is running. Please pause training to activate chat interactions."}), 400

    data = request.json or {}
    messages = data.get('messages', [])
    prompt = data.get('prompt', '')
    
    # Construct full context from history if messages are provided
    if messages:
        context = ""
        for msg in messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context += f"{role}: {msg['content']}\n"
        context += "Assistant: "
    else:
        # Fallback to single prompt if no history (or if prompt is explicitly provided)
        context = f"User: {prompt}\nAssistant: "

    # Use defaults from config if not provided
    temperature = float(data.get('temperature', manager.config.get('temperature', 0.7)))
    
    top_k = data.get('top_k', manager.config.get('top_k', 40))
    if top_k is not None:
        try:
            top_k = int(top_k)
            if top_k == 0:
                top_k = None
        except (ValueError, TypeError):
            top_k = None
            
    max_tokens = int(data.get('max_tokens', manager.config.get('max_new_tokens', 150)))
    stream = data.get('stream', manager.config.get('stream', True))

    if not context.strip():
        return jsonify({"error": "Empty prompt provided."}), 400

    # --- GGUF C++ ENGINE PATHWAY ---
    if manager.is_gguf:
        if not manager.llama_process or manager.llama_process.poll() is not None:
            manager.log("llama-server process not running. Re-initializing model to start it...")
            manager.init_model(manager.active_model_name)
            
        if not manager.llama_process or manager.llama_process.poll() is not None:
            return jsonify({"error": "Failed to start llama-server. Check server logs."}), 500

        import urllib.request
        
        # Prepare the payload for llama-server OpenAI endpoint
        formatted_messages = messages if messages else [{"role": "user", "content": prompt}]
        
        payload = {
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        if top_k is not None:
            payload["top_k"] = top_k
        
        if not stream:
            start_time = time.time()
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:5001/v1/chat/completions",
                    data=json.dumps(payload).encode('utf-8'),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                
                elapsed = time.time() - start_time
                choice = res_data['choices'][0]
                response_text = choice['message']['content']
                tokens_gen = res_data.get('usage', {}).get('completion_tokens', 0)
                speed = tokens_gen / (elapsed + 1e-5)
                
                return jsonify({
                    "response": response_text,
                    "tokens_generated": tokens_gen,
                    "elapsed_seconds": elapsed,
                    "speed_tokens_sec": speed
                })
            except Exception as e:
                manager.log(f"Failed to communicate with llama-server: {e}")
                return jsonify({"error": f"Failed to communicate with llama-server: {e}"}), 500
        else:
            def generate_gguf_sse():
                import urllib.request
                import json
                
                req = urllib.request.Request(
                    "http://127.0.0.1:5001/v1/chat/completions",
                    data=json.dumps(payload).encode('utf-8'),
                    headers={"Content-Type": "application/json"}
                )
                
                start_time = time.time()
                tokens_streamed = 0
                
                try:
                    with urllib.request.urlopen(req) as response:
                        buffer = ""
                        while True:
                            chunk = response.read(1024)
                            if not chunk:
                                break
                            buffer += chunk.decode('utf-8', errors='ignore')
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        delta = data_json['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            tokens_streamed += 1
                                            yield f"data: {json.dumps({'token': content})}\n\n"
                                    except Exception:
                                        pass
                except Exception as e:
                    yield f"data: {json.dumps({'token': f'\n[Error streaming from llama-server: {e}]'})}\n\n"
                    
                elapsed = time.time() - start_time
                yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_streamed, 'elapsed_seconds': elapsed, 'speed_tokens_sec': tokens_streamed / (elapsed + 1e-5)})}\n\n"
                
            return Response(generate_gguf_sse(), mimetype='text/event-stream')

    # --- NATIVE PYTORCH INFERENCE PATHWAY ---
    prompt_tokens = manager.tokenizer.encode(context, bos=True, eos=False)
    
    # Truncate if context is longer than block_size - max_tokens
    model_max_len = manager.config.get('block_size') or manager.config.get('max_position_embeddings') or 2048
    max_context = model_max_len - max_tokens - 10
    if max_context < 32:
        max_context = 32
    if len(prompt_tokens) > max_context:
        prompt_tokens = prompt_tokens[-max_context:]
        
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=manager.device)

    if not stream:
        start_time = time.time()
        with torch.no_grad():
            generated_ids = manager.model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
        elapsed = time.time() - start_time
        response_text = manager.tokenizer.decode(generated_ids)
        tokens_gen = len(generated_ids)
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
                    manager.model.generate(
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
                yield f"data: {json.dumps({'token': manager.tokenizer.decode([token_id])})}\n\n"
                
            elapsed = time.time() - start_time
            yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_streamed, 'elapsed_seconds': elapsed, 'speed_tokens_sec': tokens_streamed / (elapsed + 1e-5)})}\n\n"

        return Response(generate_sse(), mimetype='text/event-stream')

@app.route('/api/models/hf_list', methods=['POST'])
def hf_list_models():
    data = request.json or {}
    repo_id = data.get("repo_id", "").strip()
    if not repo_id:
        return jsonify({"error": "No Repository ID specified.", "status": "error"}), 400
    
    # Clean up Repo ID in case user pasted a full URL
    if "huggingface.co/" in repo_id:
        repo_id = repo_id.split("huggingface.co/")[-1]
    # Remove branch details if they exist
    tokens = [t for t in repo_id.split('/') if t]
    if len(tokens) >= 2:
        repo_id = f"{tokens[0]}/{tokens[1]}"
    else:
        return jsonify({"error": "Invalid Hugging Face Repository ID format. Should be 'owner/repo'.", "status": "error"}), 400
        
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            tree_data = json.loads(response.read().decode('utf-8'))
            
        gguf_files = []
        for item in tree_data:
            if item.get("type") == "file" and item.get("path", "").endswith(".gguf"):
                gguf_files.append({
                    "name": item.get("path"),
                    "size_bytes": item.get("size", 0),
                    "size_mb": round(item.get("size", 0) / (1024 * 1024), 1)
                })
        if gguf_files:
            return jsonify({"files": gguf_files, "status": "success", "repo_id": repo_id})
    except Exception:
        pass
        
    # Fallback to general model info API
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            model_data = json.loads(response.read().decode('utf-8'))
            
        siblings = model_data.get("siblings", [])
        gguf_files = []
        for s in siblings:
            filename = s.get("rfilename", "")
            if filename.endswith(".gguf"):
                gguf_files.append({
                    "name": filename,
                    "size_bytes": 0,
                    "size_mb": -1
                })
        return jsonify({"files": gguf_files, "status": "success", "repo_id": repo_id})
    except Exception as e:
        return jsonify({"error": f"Failed to query Hugging Face repo '{repo_id}': {e}", "status": "error"}), 500

@app.route('/')
def index_route():
    return render_template('index.html')

if __name__ == "__main__":
    # Clean up port 5000 and 5001 bindings
    try:
        pids = subprocess.check_output(["lsof", "-t", "-i", ":5000"], stderr=subprocess.DEVNULL).decode().split()
        for p in pids:
            if int(p) != os.getpid(): 
                os.kill(int(p), 9)
    except: 
        pass
    try:
        pids = subprocess.check_output(["lsof", "-t", "-i", ":5001"], stderr=subprocess.DEVNULL).decode().split()
        for p in pids:
            if int(p) != os.getpid(): 
                os.kill(int(p), 9)
    except: 
        pass
    
    print("--------------------------------------------------")
    print("   Native Termux SLM Server (Dual-Engine Monitor) ")
    print("--------------------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
