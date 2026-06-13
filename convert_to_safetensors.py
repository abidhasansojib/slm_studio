import torch
import json
import os
import sys

def save_safetensors(state_dict, path):
    """
    Manually saves a PyTorch state_dict as a .safetensors file without requiring the safetensors library.
    """
    header = {}
    offset = 0
    binary_data = []
    
    # Sort keys for consistency
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        
        # Ensure tensor is CPU, contiguous, and not a views
        tensor = tensor.cpu().contiguous()
        dtype = tensor.dtype
        shape = list(tensor.shape)
        
        # Map torch dtype to safetensors dtype string
        if dtype == torch.float32:
            dtype_str = "F32"
            t_bytes = tensor.numpy().tobytes()
        elif dtype == torch.float16:
            dtype_str = "F16"
            t_bytes = tensor.numpy().tobytes()
        elif dtype == torch.bfloat16:
            dtype_str = "BF16"
            # Special handling for bfloat16 (cast to int16 view to get raw bytes since numpy lacks bf16)
            t_bytes = tensor.view(torch.int16).numpy().tobytes()
        elif dtype == torch.int64:
            dtype_str = "I64"
            t_bytes = tensor.numpy().tobytes()
        elif dtype == torch.int32:
            dtype_str = "I32"
            t_bytes = tensor.numpy().tobytes()
        elif dtype == torch.int16:
            dtype_str = "I16"
            t_bytes = tensor.numpy().tobytes()
        elif dtype == torch.int8:
            dtype_str = "I8"
            t_bytes = tensor.numpy().tobytes()
        elif dtype == torch.uint8:
            dtype_str = "U8"
            t_bytes = tensor.numpy().tobytes()
        elif dtype == torch.bool:
            dtype_str = "BOOL"
            t_bytes = tensor.numpy().tobytes()
        else:
            print(f"[-] Warning: Unsupported dtype {dtype} for tensor {name}. Skipping.")
            continue
            
        length = len(t_bytes)
        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + length]
        }
        offset += length
        binary_data.append(t_bytes)
        
        # Align next tensor to 8 bytes
        padding = (8 - (offset % 8)) % 8
        if padding > 0:
            binary_data.append(b"\0" * padding)
            offset += padding
        
    header["__metadata__"] = {"format": "pt"}
    header_json = json.dumps(header).encode("utf-8")
    
    # Pad header to align binary data to 8 bytes
    # The total header size (8 bytes for length + len(header_json)) should be a multiple of 8.
    # Therefore, len(header_json) must be a multiple of 8.
    padding = (8 - (len(header_json) % 8)) % 8
    header_json += b" " * padding
    
    header_len = len(header_json)
    header_len_bytes = header_len.to_bytes(8, "little")
    
    # Write the safetensors file
    with open(path, "wb") as f:
        f.write(header_len_bytes)
        f.write(header_json)
        for chunk in binary_data:
            f.write(chunk)

def convert_to_safetensors(pt_path, out_path, config_src="models/native_slm/config.json"):
    print(f"[*] Loading weights from {pt_path}...")
    try:
        state_dict = torch.load(pt_path, map_location="cpu")
        
        # Extract model state if it is a checkpoint dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        print(f"[*] Loaded {len(state_dict)} tensors.")
        
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        print(f"[*] Writing safetensors to {out_path}...")
        save_safetensors(state_dict, out_path)
        print(f"[+] Success! Model saved as {out_path}")
        
        # Generate companion config file
        out_config_path = os.path.join(out_dir, "config.json")
        if os.path.exists(config_src):
            print(f"[*] Copying model configuration from {config_src}...")
            with open(config_src, "r") as f:
                config_data = json.load(f)
        else:
            print("[!] Source config not found. Generating default native config...")
            config_data = {
                "n_embd": 128,
                "n_head": 4,
                "n_layer": 2,
                "block_size": 128,
                "vocab_size": 259
            }
        
        # Ensure vocab_size is set (default to ByteTokenizer size if missing)
        if "vocab_size" not in config_data:
            config_data["vocab_size"] = 259
            
        with open(out_config_path, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"[+] Configuration file generated at {out_config_path}")
        
    except Exception as e:
        print(f"[-] Conversion error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        pt_path = sys.argv[1]
        out_path = sys.argv[2]
        config_src = sys.argv[3] if len(sys.argv) > 3 else "models/native_slm/config.json"
        convert_to_safetensors(pt_path, out_path, config_src)
    else:
        # Default behavior: convert native SLM
        input_model = "models/native_slm/model.pt"
        output_model = "models/native_slm/model.safetensors"
        config_file = "models/native_slm/config.json"
        
        if os.path.exists(input_model):
            convert_to_safetensors(input_model, output_model, config_file)
        else:
            print(f"[-] Default input model '{input_model}' not found.")
            print("Usage: python convert_to_safetensors.py <input.pt> <output.safetensors> [config_src.json]")
