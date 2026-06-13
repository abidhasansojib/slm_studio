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
        
        # Model state parameters
        self.model = None          # PyTorch Model Instance
        
        # Automatically establish suggested model storage directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.init_model()

    def init_model(self):
        """Initializes the PyTorch model and loads weights from .pt or .tar configurations."""
        self.config = load_config()
        self.model = TermuxSLM(
            vocab_size=self.tokenizer.vocab_size,
            n_embd=self.config["n_embd"],
            n_head=self.config["n_head"],
            n_layer=self.config["n_layer"],
            block_size=self.config["block_size"]
        ).to(self.device)
        
        # Check standard deployment weights file
        pt_path = os.path.join(self.model_dir, 'native_slm.pt')
        tar_path = os.path.join(self.model_dir, 'native_slm_checkpoint.tar')
        
        if os.path.exists(pt_path):
            try:
                self.model.load_state_dict(torch.load(pt_path, map_location=self.device))
                self.log(f"Loaded PyTorch model weights from '{pt_path}'")
                self.model.eval()
            except Exception as e:
                self.log(f"Warning: Could not load raw weights file: {e}. Checking backup structures...")
                self._load_tar_fallback()
        # Check training snapshot archive file
        elif os.path.exists(tar_path):
            self._load_tar_fallback()
        else:
            self.log(f"No PyTorch '.pt' or '.tar' found. Ready for fresh training initialization.")

    def _load_tar_fallback(self):
        """Helper to parse state parameters straight out of continuous training snapshot objects."""
        tar_path = os.path.join(self.model_dir, 'native_slm_checkpoint.tar')
        if os.path.exists(tar_path):
            try:
                checkpoint = torch.load(tar_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.log(f"Loaded PyTorch backup parameters successfully out of '{tar_path}'.")
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

    def update_config(self, new_config):
        """Helper to update internal config and save to disk."""
        for k, v in new_config.items():
            if k in DEFAULT_CONFIG:
                if isinstance(DEFAULT_CONFIG[k], bool):
                    self.config[k] = bool(v)
                elif isinstance(DEFAULT_CONFIG[k], int):
                    self.config[k] = int(v)
                elif isinstance(DEFAULT_CONFIG[k], float):
                    self.config[k] = float(v)
                else:
                    self.config[k] = v
        save_config(self.config)

    def start_training(self, new_config=None):
        """Updates configurations and executes model.py as an isolated pipeline."""
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
                pt_path = os.path.join(self.model_dir, 'native_slm.pt')
                tar_path = os.path.join(self.model_dir, 'native_slm_checkpoint.tar')
                if os.path.exists(pt_path):
                    try: os.remove(pt_path)
                    except: pass
                if os.path.exists(tar_path):
                    try: os.remove(tar_path)
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
    param_count = sum(p.numel() for p in training_manager.model.parameters() if p.requires_grad) if training_manager.model else 0
    device_info = training_manager.device.upper()
    
    try:
        threads = torch.get_num_threads()
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
        "active_engine": "pytorch"
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
        
    training_manager.log("Resetting all parameters to random initial distribution values...")
    pt_path = os.path.join(training_manager.model_dir, 'native_slm.pt')
    tar_path = os.path.join(training_manager.model_dir, 'native_slm_checkpoint.tar')
    if os.path.exists(pt_path):
        try: os.remove(pt_path)
        except: pass
    if os.path.exists(tar_path):
        try: os.remove(tar_path)
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

@app.route('/api/config/save', methods=['POST'])
def save_config_route():
    data = request.json or {}
    training_manager.update_config(data)
    return jsonify({"message": "Configuration saved successfully.", "status": "success"})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if training_manager.is_training:
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
    temperature = float(data.get('temperature', training_manager.config.get('temperature', 0.7)))
    top_k = data.get('top_k', training_manager.config.get('top_k', 40))
    if top_k is not None:
        top_k = int(top_k)
    max_tokens = int(data.get('max_tokens', training_manager.config.get('max_new_tokens', 150)))
    stream = data.get('stream', training_manager.config.get('stream', True))

    if not context.strip():
        return jsonify({"error": "Empty prompt provided."}), 400

    # --- NATIVE PYTORCH INFERENCE PATHWAY ---
    prompt_tokens = training_manager.tokenizer.encode(context, bos=True, eos=False)
    
    # Truncate if context is longer than block_size - max_tokens
    max_context = training_manager.config.get('block_size', 128) - max_tokens - 10
    if len(prompt_tokens) > max_context:
        prompt_tokens = prompt_tokens[-max_context:]
        
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
        response_text = training_manager.tokenizer.decode(generated_ids)
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
