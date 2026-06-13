import os
import time
import json
import threading
import subprocess
import shutil
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
            # Check if there is any .gguf file inside this subdirectory
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
                
                # FIX: Explicitly ensure native_slm is never parsed as a qwen3 model layout
                if entry.name == "native_slm":
                    model_type = "native_slm"
                else:
                    is_qwen = "hidden_size" in cfg or "model_type" in cfg or "num_attention_heads" in cfg
                    model_type = "qwen3" if (is_qwen or "gpt2" in entry.name.lower()) else "native_slm"
                
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
        self.log_lock = threading.Lock()
        
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
        atexit.register(self.stop_llama_server)

    def stop_llama_server(self):
        """Terminates the running llama-server background process gracefully."""
        if hasattr(self, 'llama_process') and self.llama_process:
            self.log("Stopping llama-server background process...")
            try:
                self.llama_process.terminate()
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

    def _translate_gpt2_to_qwen(self, state_dict, cfg):
        """Translates and reshapes incoming legacy GPT2 state_dicts into modern Qwen3 layouts."""
        self.log("[!] Legacy GPT-2 architecture detected. Running automatic state_dict translation layout adapter...")
        new_state = {}
        
        # 1. Map core embedding matrices
        if "wte.weight" in state_dict:
            new_state["embed_tokens.weight"] = state_dict["wte.weight"]
        
        # 2. Extract configuration rules
        n_head = cfg.get("n_head", cfg.get("num_attention_heads", 12))
        hidden_size = cfg.get("n_embd", cfg.get("hidden_size", 768))
        head_dim = hidden_size // n_head
        
        # 3. Translate block matrices sequentially
        for k, v in state_dict.items():
            if k.startswith("h."):
                parts = k.split(".")
                layer_idx = parts[1]
                sub_layer = parts[2]
                
                if sub_layer == "ln_1":
                    new_state[f"layers.{layer_idx}.input_layernorm.weight"] = v
                elif sub_layer == "ln_2":
                    new_state[f"layers.{layer_idx}.post_attention_layernorm.weight"] = v
                elif sub_layer == "mlp":
                    param_type = parts[3] # c_fc or c_proj
                    if param_type == "c_fc" and "weight" in k:
                        w_t = v.t()
                        chunks = w_t.chunk(2, dim=0) if w_t.size(0) == hidden_size * 8 else [w_t, w_t]
                        new_state[f"layers.{layer_idx}.mlp.gate_proj.weight"] = chunks[0]
                        new_state[f"layers.{layer_idx}.mlp.up_proj.weight"] = chunks[1]
                    elif param_type == "c_proj" and "weight" in k:
                        new_state[f"layers.{layer_idx}.mlp.down_proj.weight"] = v.t()
                elif sub_layer == "attn":
                    param_type = parts[3] # c_attn or c_proj
                    if param_type == "c_attn" and "weight" in k:
                        w_t = v.t()
                        q, k_t, v_t = w_t.chunk(3, dim=0)
                        new_state[f"layers.{layer_idx}.self_attn.q_proj.weight"] = q
                        new_state[f"layers.{layer_idx}.self_attn.k_proj.weight"] = k_t
                        new_state[f"layers.{layer_idx}.self_attn.v_proj.weight"] = v_t
                        
                        # Fix head allocation shape dimension mapping parameters (head_dim instead of hidden_size)
                        new_state[f"layers.{layer_idx}.self_attn.q_norm.weight"] = torch.ones(head_dim, dtype=v.dtype, device=v.device)
                        new_state[f"layers.{layer_idx}.self_attn.k_norm.weight"] = torch.ones(head_dim, dtype=v.dtype, device=v.device)
                    elif param_type == "c_proj" and "weight" in k:
                        new_state[f"layers.{layer_idx}.self_attn.o_proj.weight"] = v.t()

        # 4. Final layer normalization matrices mapping
        if "ln_f.weight" in state_dict:
            new_state["norm.weight"] = state_dict["ln_f.weight"]
        if "wte.weight" in state_dict:
            new_state["lm_head.weight"] = state_dict["wte.weight"]
            
        return new_state

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

        self.stop_llama_server()
        self.is_gguf = False

        try:
            model_dir = model_info["path"]
            
            if model_info["type"] == "gguf":
                self.is_gguf = True
                self.config = model_info["config"]
                self.tokenizer = None
                self.model = None
                
                weights_path = os.path.join(model_dir, model_info["weights_file"]) if model_dir != "models" else os.path.join("models", model_info["weights_file"])
                self.log(f"Active model is a quantized GGUF. Starting C++ engine llama-server...")
                
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
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                
                def log_reader(process, manager):
                    try:
                        for line in iter(process.stdout.readline, ''):
                            if not line: break
                            manager.log(f"[llama-server] {line.strip()}")
                    except Exception as e:
                        manager.log(f"[llama-server Reader Error] {e}")
                    finally:
                        process.stdout.close()
                
                threading.Thread(target=log_reader, args=(self.llama_process, self), daemon=True).start()
                
                time.sleep(2.5)
                if self.llama_process.poll() is not None:
                    raise RuntimeError("llama-server failed to launch.")
                
                self.log("llama-server successfully started in the background on port 5001!")
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
            
            # Auto-translate keys if GPT2 weight vectors are found
            if state_dict and ("wte.weight" in state_dict or "h.0.ln_1.weight" in state_dict):
                state_dict = self._translate_gpt2_to_qwen(state_dict, self.config)

            model_dtype = torch.float32
            if state_dict:
                for tensor in state_dict.values():
                    if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.bfloat16, torch.float16]:
                        model_dtype = torch.bfloat16
                        break
            
            self.log(f"Configuring model layers with precision: {model_dtype}")
            
            if model_info["type"] == "qwen3":
                if "n_layer" in self.config and "num_hidden_layers" not in self.config:
                    self.config["num_hidden_layers"] = self.config["n_layer"]
                if "n_embd" in self.config and "hidden_size" not in self.config:
                    self.config["hidden_size"] = self.config["n_embd"]
                    
                self.tokenizer = QwenTokenizer(model_dir) if os.path.exists(os.path.join(model_dir, "qwen.tiktoken")) else ByteTokenizer()
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
                if any(k.startswith("model.") for k in state_dict.keys()):
                    state_dict = { (k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items() }
                
                state_dict = {k: v.to(model_dtype) for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.log(f"Successfully loaded '{model_name}' weights!")
            else:
                self.log("Ready for fresh training initialization.")
            
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
        with self.log_lock:
            if msg_str.startswith('\r'):
                clean_msg = msg_str.replace('\r', '').strip()
                if clean_msg:
                    timestamp = time.strftime("%H:%M:%S")
                    if self.logs:
                        self.logs[-1] = f"[{timestamp}] {clean_msg}"
                    else:
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
        for k, v in new_config.items():
            if isinstance(v, bool):
                self.config[k] = bool(v)
            elif isinstance(v, (int, float)):
                self.config[k] = v
            elif isinstance(v, str) and v.isdigit():
                self.config[k] = int(v)
            else:
                try: self.config[k] = float(v)
                except ValueError: self.config[k] = v
                    
        model_dir = os.path.join(self.model_dir, self.active_model_name)
        if os.path.exists(model_dir):
            config_path = os.path.join(model_dir, "config.json")
            save_config(self.config, config_path)

    def start_training(self, new_config=None):
        if self.is_gguf:
            return False, "Training is not supported for quantized GGUF models."
        if self.is_training:
            return False, "Training is already in progress."
            
        if new_config:
            arch_keys = ["n_embd", "n_head", "n_layer", "block_size"]
            arch_changed = False
            for k in arch_keys:
                if k in new_config and int(new_config[k]) != self.config.get(k):
                    arch_changed = True
            
            self.update_config(new_config)
            
            if arch_changed:
                self.log("Network dimensions modified. Resetting checkpoint layers...")
                model_dir = os.path.join(self.model_dir, "native_slm")
                for path in ['model.pt', 'checkpoint.tar']:
                    p = os.path.join(model_dir, path)
                    if os.path.exists(p): os.remove(p)
                self.init_model("native_slm")
        
        self.is_training = True
        self.current_step = 0
        self.total_steps = self.config.get("max_iters", 1000)
        self.losses_history = []
        self.val_losses_history = []
        
        threading.Thread(target=self._subprocess_monitor_worker, daemon=True).start()
        return True, "Subprocess pipeline initialized."

    def stop_training(self):
        if not self.is_training or not self.train_process:
            return False, "Training is not active."
        try:
            self.train_process.terminate()
            self.log("Termination signal dispatched.")
            return True, "Process stop sequence executed."
        except Exception as e:
            return False, f"Failed to terminate process: {e}"

    def _subprocess_monitor_worker(self):
        try:
            cmd = ["python3", "-u", "model.py"]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            self.train_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )
            while True:
                line = self.train_process.stdout.readline()
                if not line and self.train_process.poll() is not None: break
                if line:
                    clean_line = line.strip()
                    self.log(clean_line)
                    self._parse_metrics(clean_line)

            self.init_model()
        except Exception as e:
            self.log(f"Subprocess supervisor error: {e}")
        finally:
            self.is_training = False
            self.train_process = None

    def _parse_metrics(self, line):
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

training_manager = ExternalTrainingManager()

# --- 2. REST API ENDPOINTS ---

@app.route('/api/status', methods=['GET'])
def get_status():
    param_count = sum(p.numel() for p in training_manager.model.parameters() if p.requires_grad) if training_manager.model else 0
    device_info = training_manager.device.upper()
    try: threads = torch.get_num_threads()
    except: threads = training_manager.config.get("num_threads", 2)
        
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
        "active_engine": "pytorch",
        "active_model": training_manager.active_model_name
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    models = list_available_models()
    if not any(m["name"] == "native_slm" for m in models):
        native_dir = os.path.join(training_manager.model_dir, "native_slm")
        models.append({
            "name": "native_slm",
            "path": native_dir,
            "weights_file": "model.pt" if os.path.exists(os.path.join(native_dir, "model.pt")) else None,
            "type": "native_slm",
            "config": load_config(os.path.join(native_dir, "config.json"))
        })
    return jsonify({"models": models, "active_model": training_manager.active_model_name})

@app.route('/api/models/select', methods=['POST'])
def select_model():
    if training_manager.is_training:
        return jsonify({"error": "Cannot change models during training."}), 400
    data = request.json or {}
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"error": "No model name specified."}), 400
    
    training_manager.init_model(model_name)
    return jsonify({"message": f"Model '{model_name}' loaded.", "status": "success"})

@app.route('/api/models/delete', methods=['POST'])
def delete_model():
    if training_manager.is_training:
        return jsonify({"error": "Cannot delete models during training."}), 400
    data = request.json or {}
    model_name = data.get("model_name")
    if model_name == "native_slm":
        return jsonify({"error": "Cannot delete native baseline."}), 400
    
    if training_manager.active_model_name == model_name:
        training_manager.init_model("native_slm")
        
    model_dir = os.path.join(training_manager.model_dir, model_name)
    if os.path.exists(model_dir):
        try:
            shutil.rmtree(model_dir)
            return jsonify({"message": f"Model '{model_name}' deleted.", "status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Model not found."}), 404

@app.route('/api/models/download', methods=['POST'])
def download_model_route():
    data = request.json or {}
    url = data.get("url")
    if not url: return jsonify({"error": "No URL specified."}), 400
        
    def run_download():
        try:
            from download_model import download_model as dl_model
            dl_model(url, log_callback=training_manager.log)
        except Exception as e:
            training_manager.log(f"Download exception: {e}")
            
    threading.Thread(target=run_download, daemon=True).start()
    return jsonify({"message": "Download process started in background.", "status": "success"})

@app.route('/api/train/start', methods=['POST'])
def start_train():
    data = request.json or {}
    success, msg = training_manager.start_training(data)
    if success: return jsonify({"message": msg, "status": "success"})
    return jsonify({"error": msg, "status": "error"}), 400

@app.route('/api/train/stop', methods=['POST'])
def stop_train():
    success, msg = training_manager.stop_training()
    if success: return jsonify({"message": msg, "status": "success"})
    return jsonify({"error": msg, "status": "error"}), 400

@app.route('/api/reset', methods=['POST'])
def reset_model_route():
    if training_manager.is_training: return jsonify({"error": "Training loop is active."}), 400
    model_dir = os.path.join(training_manager.model_dir, "native_slm")
    for filename in ['model.pt', 'checkpoint.tar']:
        p = os.path.join(model_dir, filename)
        if os.path.exists(p): os.remove(p)
    training_manager.init_model("native_slm")
    return jsonify({"message": "Weights cleared.", "status": "success"})

@app.route('/api/dataset', methods=['GET', 'POST'])
def handle_dataset():
    dataset_path = 'training_data.txt'
    if request.method == 'GET':
        if not os.path.exists(dataset_path): return jsonify({"text": ""})
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f: content = f.read(100000)
            return jsonify({"text": content, "truncated": os.path.getsize(dataset_path) > 100000})
        except Exception as e: return jsonify({"error": str(e)}), 500
    else:
        data = request.json or {}
        try:
            with open(dataset_path, 'w', encoding='utf-8') as f: f.write(data.get('text', ''))
            return jsonify({"message": "Dataset saved successfully.", "status": "success"})
        except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/config/save', methods=['POST'])
def save_config_route():
    data = request.json or {}
    training_manager.update_config(data)
    return jsonify({"message": "Configuration saved successfully.", "status": "success"})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if training_manager.is_training:
        return jsonify({"error": "Model training process is running. Pause training to chat."}), 400

    data = request.json or {}
    messages = data.get('messages', [])
    prompt = data.get('prompt', '')
    
    if messages:
        context = ""
        for msg in messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context += f"{role}: {msg['content']}\n"
        context += "Assistant: "
    else:
        context = f"User: {prompt}\nAssistant: "

    temperature = float(data.get('temperature', training_manager.config.get('temperature', 0.7)))
    top_k = data.get('top_k', training_manager.config.get('top_k', 40))
    if top_k is not None:
        try: top_k = int(top_k)
        except: top_k = None
            
    max_tokens = int(data.get('max_tokens', training_manager.config.get('max_new_tokens', 150)))
    stream = data.get('stream', training_manager.config.get('stream', True))

    if not context.strip(): return jsonify({"error": "Empty prompt."}), 400

    # --- GGUF C++ ENGINE PATHWAY ---
    if training_manager.is_gguf:
        import urllib.request
        formatted_messages = messages if messages else [{"role": "user", "content": prompt}]
        payload = {"messages": formatted_messages, "temperature": temperature, "max_tokens": max_tokens, "stream": stream}
        
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
                return jsonify({
                    "response": choice['message']['content'],
                    "tokens_generated": res_data.get('usage', {}).get('completion_tokens', 0),
                    "elapsed_seconds": elapsed,
                    "speed_tokens_sec": res_data.get('usage', {}).get('completion_tokens', 0) / (elapsed + 1e-5)
                })
            except Exception as e: return jsonify({"error": str(e)}), 500
        else:
            def generate_gguf_sse():
                import urllib.request
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
                            if not chunk: break
                            buffer += chunk.decode('utf-8', errors='ignore')
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str == "[DONE]": break
                                    try:
                                        content = json.loads(data_str)['choices'][0].get('delta', {}).get('content', '')
                                        if content:
                                            tokens_streamed += 1
                                            yield f"data: {json.dumps({'token': content})}\n\n"
                                    except: pass
                except Exception as e:
                    yield f"data: {json.dumps({'token': f'\n[Streaming error: {e}]'})}\n\n"
                elapsed = time.time() - start_time
                yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_streamed, 'elapsed_seconds': elapsed, 'speed_tokens_sec': tokens_streamed / (elapsed + 1e-5)})}\n\n"
            return Response(generate_gguf_sse(), mimetype='text/event-stream')

    # --- NATIVE PYTORCH INFERENCE PATHWAY ---
    prompt_tokens = training_manager.tokenizer.encode(context, bos=True, eos=False)
    model_max_len = training_manager.config.get('block_size') or training_manager.config.get('max_position_embeddings') or 2048
    max_context = model_max_len - max_tokens - 10
    if max_context < 32: max_context = 32
    if len(prompt_tokens) > max_context: prompt_tokens = prompt_tokens[-max_context:]
        
    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=training_manager.device)

    if not stream:
        start_time = time.time()
        with torch.no_grad():
            generated_ids = training_manager.model.generate(
                input_tensor, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k
            )
        elapsed = time.time() - start_time
        response_text = training_manager.tokenizer.decode(generated_ids)
        return jsonify({
            "response": response_text,
            "tokens_generated": len(generated_ids),
            "elapsed_seconds": elapsed,
            "speed_tokens_sec": len(generated_ids) / (elapsed + 1e-5)
        })
    else:
        def generate_sse():
            q = Queue()
            def token_callback(token_id): q.put(token_id)
            def worker():
                try:
                    training_manager.model.generate(
                        input_tensor, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, callback=token_callback
                    )
                except: pass
                finally: q.put(None)
            
            threading.Thread(target=worker, daemon=True).start()
            start_time = time.time()
            tokens_streamed = 0
            while True:
                token_id = q.get()
                if token_id is None: break
                tokens_streamed += 1
                yield f"data: {json.dumps({'token': training_manager.tokenizer.decode([token_id])})}\n\n"
            elapsed = time.time() - start_time
            yield f"data: {json.dumps({'done': True, 'tokens_generated': tokens_streamed, 'elapsed_seconds': elapsed, 'speed_tokens_sec': tokens_streamed / (elapsed + 1e-5)})}\n\n"
        return Response(generate_sse(), mimetype='text/event-stream')

@app.route('/api/models/hf_list', methods=['POST'])
def hf_list_models():
    data = request.json or {}
    repo_id = data.get("repo_id", "").strip()
    if not repo_id: return jsonify({"error": "No Repository ID specified."}), 400
    if "huggingface.co/" in repo_id: repo_id = repo_id.split("huggingface.co/")[-1]
    tokens = [t for t in repo_id.split('/') if t]
    if len(tokens) >= 2: repo_id = f"{tokens[0]}/{tokens[1]}"
    else: return jsonify({"error": "Invalid format. Should be 'owner/repo'."}), 400
        
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            tree_data = json.loads(response.read().decode('utf-8'))
        gguf_files = []
        for item in tree_data:
            if item.get("type") == "file" and item.get("path", "").endswith(".gguf"):
                gguf_files.append({"name": item.get("path"), "size_bytes": item.get("size", 0), "size_mb": round(item.get("size", 0) / (1024 * 1024), 1)})
        if gguf_files: return jsonify({"files": gguf_files, "status": "success", "repo_id": repo_id})
    except: pass

    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            model_data = json.loads(response.read().decode('utf-8'))
        gguf_files = [{"name": s.get("rfilename", ""), "size_bytes": 0, "size_mb": -1} for s in model_data.get("siblings", []) if s.get("rfilename", "").endswith(".gguf")]
        return jsonify({"files": gguf_files, "status": "success", "repo_id": repo_id})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/')
def index_route(): return render_template('index.html')

if __name__ == "__main__":
    print("--------------------------------------------------")
    print("   Native Termux SLM Server (Dual-Engine Monitor) ")
    print("--------------------------------------------------")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
