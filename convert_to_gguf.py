import torch
import json
import os
import sys
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

def load_safetensors_manual(path):
    """
    Manually loads .safetensors files without requiring the 'safetensors' pip module.
    Adapted from model.py
    """
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Invalid safetensors file: Header length missing.")
        header_len = int.from_bytes(header_len_bytes, "little")
        header_json = f.read(header_len).decode("utf-8")
        header = json.loads(header_json)
        data_start_offset = 8 + header_len
        
        state_dict = {}
        for name, info in header.items():
            if name == "__metadata__": continue
            offsets = info.get("data_offsets")
            if not offsets: continue
            
            start, end = offsets
            dtype_str = info.get("dtype")
            shape = info.get("shape")
            
            torch_dtype = {
                "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
                "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
                "U8": torch.uint8, "BOOL": torch.bool,
            }.get(dtype_str)
            
            if torch_dtype is None: continue

            f.seek(data_start_offset + start)
            raw_bytes = f.read(end - start)
            
            try:
                if dtype_str == "BF16":
                    tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16).view(torch.bfloat16).reshape(shape)
                else:
                    tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype).reshape(shape)
            except Exception:
                # Fallback to numpy
                np_dtype = {
                    "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
                    "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
                }.get(dtype_str, np.float32)
                
                if dtype_str == "BF16":
                    tensor = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.int16).copy()).view(torch.bfloat16).reshape(shape)
                else:
                    tensor = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np_dtype).copy()).reshape(shape)
            
            state_dict[name] = tensor
            
        return state_dict

def convert_to_gguf(input_path, out_path, config_src=None):
    print(f"[*] Loading weights from {input_path}...")
    try:
        if input_path.endswith(".safetensors"):
            state_dict = load_safetensors_manual(input_path)
        else:
            state_dict = torch.load(input_path, map_location="cpu")
        
        # Extract model state if it is a checkpoint dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        print(f"[*] Loaded {len(state_dict)} tensors.")
        
        # Try to find config if not provided
        if config_src is None:
            config_src = os.path.join(os.path.dirname(input_path), "config.json")
            
        if os.path.exists(config_src):
            print(f"[*] Loading configuration from {config_src}...")
            with open(config_src, "r") as f:
                config = json.load(f)
        else:
            print("[!] Source config not found. Using defaults.")
            config = {
                "n_embd": 128, "n_head": 4, "n_layer": 4, "block_size": 128, "vocab_size": 259
            }

        # Determine architecture and map keys
        # We'll support both TermuxSLM and Qwen3Model by checking keys
        is_qwen3 = any("self_attn" in k for k in state_dict.keys())
        
        # Metadata mapping
        if is_qwen3:
            arch = "llama" # Qwen3 follows llama-like structure in model.py
            n_layers = config.get("num_hidden_layers", 4)
            n_embd = config.get("hidden_size", 256)
            n_head = config.get("num_attention_heads", 8)
            n_head_kv = config.get("num_key_value_heads", n_head)
            n_ff = config.get("intermediate_size", 1024)
            n_ctx = config.get("max_position_embeddings", 4096)
            rms_norm_eps = config.get("rms_norm_eps", 1e-6)
            rope_theta = config.get("rope_theta", 1000000.0)
        else:
            arch = "llama"
            n_layers = config.get("n_layer", 4)
            n_embd = config.get("n_embd", 128)
            n_head = config.get("n_head", 4)
            n_head_kv = n_head
            # TermuxSLM FeedForward hidden_dim calculation
            hidden_dim = config.get("hidden_dim")
            if hidden_dim is None:
                hidden_dim = int(2.7 * n_embd)
                hidden_dim = 32 * ((hidden_dim + 32 - 1) // 32)
            n_ff = hidden_dim
            n_ctx = config.get("block_size", 128)
            rms_norm_eps = 1e-6
            rope_theta = 10000.0

        writer = GGUFWriter(out_path, arch)
        
        # Add metadata
        writer.add_name("Native SLM")
        writer.add_description("Converted from PyTorch .pt model")
        writer.add_architecture()
        writer.add_block_count(n_layers)
        writer.add_embedding_length(n_embd)
        writer.add_feed_forward_length(n_ff)
        writer.add_context_length(n_ctx)
        writer.add_head_count(n_head)
        writer.add_head_count_kv(n_head_kv)
        writer.add_layer_norm_rms_eps(rms_norm_eps)
        writer.add_rope_dimension_count(n_embd // n_head)
        writer.add_rope_freq_base(rope_theta)
        
        # Add Tokenizer (ByteTokenizer or Qwen)
        vocab_size = config.get("vocab_size", 259)
        tokens = []
        scores = []
        toktypes = []
        
        if vocab_size == 259:
            # ByteTokenizer
            writer.add_tokenizer_model("llama") # Use llama-style byte tokens
            for i in range(256):
                tokens.append(bytes([i]))
                scores.append(0.0)
                toktypes.append(1) # Normal
            # Special tokens
            tokens.extend([b"<pad>", b"<bos>", b"<eos>"])
            scores.extend([0.0, 0.0, 0.0])
            toktypes.extend([3, 3, 3]) # Control
            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_types(toktypes)
            writer.add_bos_token_id(257)
            writer.add_eos_token_id(258)
            writer.add_padding_token_id(256)
        else:
            # Handle Qwen/HF vocab if possible, but for native SLM 259 is default
            print(f"[*] Custom vocab size {vocab_size} detected. Filling with placeholders.")
            for i in range(vocab_size):
                tokens.append(f"<token_{i}>".encode('utf-8'))
                scores.append(0.0)
                toktypes.append(1)
            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_types(toktypes)

        # Map and Add Tensors
        print("[*] Mapping tensors...")
        for name, tensor in state_dict.items():
            new_name = name
            
            # Common mappings
            if is_qwen3:
                new_name = new_name.replace("embed_tokens.weight", "token_embd.weight")
                new_name = new_name.replace("layers.", "blk.")
                new_name = new_name.replace(".self_attn.q_proj.weight", ".attn_q.weight")
                new_name = new_name.replace(".self_attn.k_proj.weight", ".attn_k.weight")
                new_name = new_name.replace(".self_attn.v_proj.weight", ".attn_v.weight")
                new_name = new_name.replace(".self_attn.o_proj.weight", ".attn_output.weight")
                new_name = new_name.replace(".input_layernorm.weight", ".attn_norm.weight")
                new_name = new_name.replace(".mlp.gate_proj.weight", ".ffn_gate.weight")
                new_name = new_name.replace(".mlp.up_proj.weight", ".ffn_up.weight")
                new_name = new_name.replace(".mlp.down_proj.weight", ".ffn_down.weight")
                new_name = new_name.replace(".post_attention_layernorm.weight", ".ffn_norm.weight")
                new_name = new_name.replace("norm.weight", "output_norm.weight")
                new_name = new_name.replace("lm_head.weight", "output.weight")
            else:
                new_name = new_name.replace("token_embedding_table.weight", "token_embd.weight")
                new_name = new_name.replace("layers.", "blk.")
                new_name = new_name.replace(".attention.wq.weight", ".attn_q.weight")
                new_name = new_name.replace(".attention.wk.weight", ".attn_k.weight")
                new_name = new_name.replace(".attention.wv.weight", ".attn_v.weight")
                new_name = new_name.replace(".attention.wo.weight", ".attn_output.weight")
                new_name = new_name.replace(".norm1.weight", ".attn_norm.weight")
                new_name = new_name.replace(".ffwd.w1.weight", ".ffn_gate.weight")
                new_name = new_name.replace(".ffwd.w2.weight", ".ffn_up.weight")
                new_name = new_name.replace(".ffwd.w3.weight", ".ffn_down.weight")
                new_name = new_name.replace(".norm2.weight", ".ffn_norm.weight")
                new_name = new_name.replace("final_norm.weight", "output_norm.weight")
                new_name = new_name.replace("lm_head.weight", "output.weight")

            # RoPE cached tensors are not needed in GGUF
            if "cos_cached" in new_name or "sin_cached" in new_name or "inv_freq" in new_name or "tril" in new_name:
                continue

            # Convert to numpy and add
            data = tensor.float().numpy()
            writer.add_tensor(new_name, data)
            print(f"    - {name} -> {new_name} {data.shape}")

        print(f"[*] Writing GGUF file to {out_path}...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        
        print(f"[+] Success! GGUF model saved as {out_path}")
        
    except Exception as e:
        print(f"[-] Conversion error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        out_path = sys.argv[2]
        config_src = sys.argv[3] if len(sys.argv) > 3 else None
        convert_to_gguf(input_path, out_path, config_src)
    else:
        # Default behavior: convert native SLM
        input_model = "models/native_slm/model.pt"
        if not os.path.exists(input_model):
            input_model = "models/native_slm/model.safetensors"
            
        output_model = "models/native_slm/model.gguf"
        config_file = "models/native_slm/config.json"
        
        if os.path.exists(input_model):
            convert_to_gguf(input_model, output_model, config_file)
        else:
            print(f"[-] Default input model '{input_model}' not found.")
            print("Usage: python convert_to_gguf.py <input.pt/.safetensors> <output.gguf> [config_src.json]")
