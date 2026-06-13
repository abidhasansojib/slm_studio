import os
import json
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. PROFESSIONAL CONFIGURATION SYSTEM ---
DEFAULT_CONFIG = {
    "n_embd": 256,            # Increased for better pattern representation
    "n_head": 8,              # More heads for diverse attention patterns
    "n_layer": 6,             # Deeper network for better logic
    "block_size": 384,        # Significantly larger context window (approx 100 words)
    "learning_rate": 8e-4,    
    "min_lr": 8e-5,
    "batch_size": 32,         
    "max_iters": 15000,       # More steps needed for character-level learning
    "eval_interval": 250,
    "save_interval": 1000,
    "num_threads": 4,         
    "dropout": 0.1,           
    "weight_decay": 0.1,      
    "temperature": 0.75,
    "top_k": 40,
    "max_new_tokens": 200,
    "stream": True
}

def load_config(path="model_config.json"):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in config:
                    config[k] = v
            return config
        except Exception as e:
            print(f"[-] Error reading config {path}: {e}. Initializing defaults.")
    return DEFAULT_CONFIG.copy()

def save_config(config, path="model_config.json"):
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"[-] Failed to update architecture configuration metadata: {e}")

# --- 2. ROBUST BYTE-LEVEL TOKENIZER ---
class ByteTokenizer:
    def __init__(self):
        self.pad_token = 256
        self.bos_token = 257
        self.eos_token = 258
        self.vocab_size = 259

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        raw_bytes = text.encode('utf-8', errors='ignore')
        tokens = []
        if bos:
            tokens.append(self.bos_token)
        tokens.extend(list(raw_bytes))
        if eos:
            tokens.append(self.eos_token)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        filtered_bytes = bytearray([t for t in tokens if t < 256])
        return filtered_bytes.decode('utf-8', errors='replace')

# --- 3. HARDWARE CAPABILITY MONITOR ---
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# --- 4. ADVANCED TRANSFORMER MODULES WITH REGULARIZATION ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=512, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x):
        half_dim = self.dim // 2
        return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)

    def forward(self, x, start_pos=0):
        T = x.shape[2]
        cos = self.cos_cached[start_pos : start_pos + T, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin_cached[start_pos : start_pos + T, :].unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

class Attention(nn.Module):
    def __init__(self, n_embd, n_head, rope, dropout=0.15):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.rope = rope
        
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        
        # Drops attention paths and projection weights during backward passes
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        start_pos = 0 if kv_cache is None else kv_cache[0].shape[2]
        
        q = self.rope(q, start_pos=start_pos)
        k = self.rope(k, start_pos=start_pos)
        
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_kv_cache = (k, v)
        
        total_T = k.shape[2]
        
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        if mask is not None:
            scores = scores.masked_fill(mask[:T, :total_T] == 0, float('-inf'))
            
        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        
        out = scores @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(out)), new_kv_cache

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.15):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2.7 * dim)
            hidden_dim = 32 * ((hidden_dim + 32 - 1) // 32)
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.resid_dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, rope, dropout=0.15):
        super().__init__()
        self.attention = Attention(n_embd, n_head, rope, dropout=dropout)
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.norm1 = RMSNorm(n_embd)
        self.norm2 = RMSNorm(n_embd)

    def forward(self, x, mask=None, kv_cache=None):
        attn_out, new_cache = self.attention(self.norm1(x), mask=mask, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffwd(self.norm2(x))
        return x, new_cache

# --- 5. THE CAUSAL AUTOREGRESSIVE DECODER ARCHITECTURE ---
class TermuxSLM(nn.Module):
    def __init__(self, vocab_size=259, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.15):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        self.rope = RoPE(dim=n_embd // n_head, max_seq_len=block_size)
        self.layers = nn.ModuleList([
            TransformerBlock(n_embd, n_head, self.rope, dropout=dropout) for _ in range(n_layer)
        ])
        self.final_norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.token_embedding_table.weight = self.lm_head.weight
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)), persistent=False)

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        
        mask = self.tril[:T, :T] if T > 1 else None
        
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, mask=mask, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, callback=None):
        self.eval()
        B, T = idx.shape
        kv_caches = [None] * len(self.layers)
        
        if T >= self.block_size:
            idx = idx[:, -max(1, self.block_size - 32):]
            T = idx.shape[1]
            
        max_new_tokens = min(max_new_tokens, self.block_size - T)
        if max_new_tokens <= 0:
            return []
            
        logits, _, kv_caches = self(idx, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :] / (temperature + 1e-5)
        
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, self.vocab_size))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated = [next_token.item()]
        if callback:
            callback(next_token.item())
            
        for _ in range(max_new_tokens - 1):
            logits, _, kv_caches = self(next_token, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :] / (temperature + 1e-5)
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, self.vocab_size))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token.item())
            if callback:
                callback(next_token.item())
                
            if next_token.item() == 258: # End-of-Stream boundary match
                break
                
        return generated

# --- 6. MULTI-SOURCE CORPUS COMPILATION PIPELINE ---
def compile_training_corpus(data_directory="data_corpus", fallback_file="training_data.txt"):
    """
    Scans for raw encyclopedic knowledge and live user interaction loops, 
    compiling them dynamically into a unified training vector.
    """
    compiled_text = ""
    
    # 1. Self-Learning Loop Ingestion
    if os.path.exists("self_learning_buffer.txt"):
        try:
            with open("self_learning_buffer.txt", "r", encoding="utf-8") as f:
                buffer_data = f.read().strip()
                if len(buffer_data) > 0:
                    compiled_text += buffer_data + "\n"
                    print("[Self-Learning Engine] Ingested user interface conversation histories.")
        except Exception as e:
            print(f"[-] Warning: Failed to read self-learning buffer: {e}")

    # 2. Folder Corpora Aggregation (Wikipedia/Books/Scraped Text)
    if os.path.exists(data_directory) and os.path.isdir(data_directory):
        target_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
        if target_files:
            print(f"[Data Pipeline] Bundling {len(target_files)} external text sources from '{data_directory}'...")
            for path in target_files:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        compiled_text += f.read() + "\n"
                except Exception as e:
                    print(f"[-] Warning: Failed to read {path}: {e}")

    # 3. Native Fallback Script Verification
    if os.path.exists(fallback_file):
        print(f"[Data Pipeline] Processing native sequence script from '{fallback_file}'.")
        try:
            with open(fallback_file, 'r', encoding='utf-8') as f:
                compiled_text += f.read()
        except Exception as e:
            print(f"[-] Warning: Failed to read fallback file: {e}")
            
    if not compiled_text.strip():
        raise FileNotFoundError(f"Data stream targets missing. Populate '{data_directory}/' or '{fallback_file}' to proceed.")
        
    return compiled_text

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters=5):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size)
        _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# --- 7. PRODUCTION CONTINUAL LEARNING LOOP ---
if __name__ == "__main__":
    print("================================================================")
    print("  Professional Continual Learning Engine (ARM Mobile Architecture) ")
    print("================================================================")
    
    config = load_config()
    torch.set_num_threads(config["num_threads"])
    torch.set_num_interop_threads(1)
    device = get_device()
    
    print(f"Hardware Worker Threads Assigned: {config['num_threads']}")
    print(f"Target Compute Layer Engine: {device.upper()}")
    
    # Instantiate internal layout structure directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data_corpus", exist_ok=True)
    
    try:
        raw_text_stream = compile_training_corpus()
    except Exception as e:
        print(f"[-] Processing Interrupted: {e}")
        exit(1)
        
    print(f"[Data Pipeline] Unified raw characters inside text stream: {len(raw_text_stream):,}")
    tokenizer = ByteTokenizer()
    
    print("[Data Pipeline] Transforming symbols into byte-level tensors...")
    encoded_stream = tokenizer.encode(raw_text_stream, bos=True, eos=True)
    data_tensor = torch.tensor(encoded_stream, dtype=torch.long, device=device)
    print(f"[Data Pipeline] Total token sequences built for memory assignment: {len(data_tensor):,}")
    
    # 90/10 Train and Evaluation data splitting
    split_idx = int(0.9 * len(data_tensor))
    train_data = data_tensor[:split_idx]
    val_data = data_tensor[split_idx:]
    
    # Build core network architecture
    model = TermuxSLM(
        vocab_size=tokenizer.vocab_size,
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
        dropout=config.get("dropout", 0.15)
    ).to(device)
    
    # Professional optimizer with weight decay scaling to prevent over-clustering
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.1))
    
    # --- NON-DESTRUCTIVE RUNTIME RESUMPTION GUARD ---
    checkpoint_tar_path = 'models/native_slm_checkpoint.tar' # Main session snapshot container
    active_runtime_weights = 'models/native_slm.pt'           # Target weight path consumed by Flask Server
    start_step = 0
    
    if os.path.exists(checkpoint_tar_path):
        try:
            print(f"[Checkpoint Manager] Located session cache layout file. Restoring context parameters...")
            checkpoint = torch.load(checkpoint_tar_path, map_location=device)
            
            # Safely recover weights, step matrices, and optimizer momentum
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step'] + 1
            
            print(f"--> Success! Session state restored. Progress resuming instantly at Step {start_step}.")
        except Exception as e:
            print(f"[-] Snapshot read warning ({e}). Regenerating a fresh session trace context.")
            
    total_target_iters = config["max_iters"]
    if start_step >= total_target_iters:
        print(f"[!] Target limit match detected ({start_step} >= {total_target_iters}). Expanding learning horizon automatically by 500 steps.")
        total_target_iters = start_step + 500
        config["max_iters"] = total_target_iters

    print(f"\nTraining execution engine engaged. Operational Scope: Steps {start_step} to {total_target_iters}")
    print("----------------------------------------------------------------")
    
    model.train()
    start_time = time.time()
    
    for step in range(start_step, total_target_iters + 1):
        # Continuous Cosine Learning Rate Decay Curve Calculation
        progress = step / total_target_iters
        lr = config["min_lr"] + 0.5 * (config["learning_rate"] - config["min_lr"]) * (1.0 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Fetch vectorized training slices
        xb, yb = get_batch(train_data, config["block_size"], config["batch_size"])
        
        _, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Norm clipping prevents exploding gradients from corrupting weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Evaluation cycles
        if step % config["eval_interval"] == 0 or step == total_target_iters:
            val_loss = estimate_loss(model, val_data, config["block_size"], config["batch_size"], eval_iters=5)
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{total_target_iters} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f} | Runtime: {elapsed:.1f}s")
            
        # Snapshot auto-saves
        if step % config["save_interval"] == 0 and step > start_step:
            print(f"--> Exporting comprehensive training session checkpoint matrices...")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_tar_path)
            
            # Save raw model weights file explicitly for the Flask runtime server
            torch.save(model.state_dict(), active_runtime_weights)
            save_config(config)
            print(f"--> Snapshot saved successfully to '{checkpoint_tar_path}' and '{active_runtime_weights}'.")
            
    # Final iteration milestone save
    torch.save({
        'step': total_target_iters,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, checkpoint_tar_path)
    torch.save(model.state_dict(), active_runtime_weights)
    save_config(config)
    print(f"\n[+] Milestone completed! Session data successfully frozen at Step {total_target_iters}.")
