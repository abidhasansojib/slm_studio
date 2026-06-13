import torch
import json
import os

# Manual implementation of the Safetensors format (Simple Version)
# Note: Real safetensors files have a specific header. Since 'safetensors' 
# library failed to install on this mobile environment due to Rust dependencies,
# this script demonstrates how you WOULD do it if the library was available.

def convert_to_safetensors(pt_path, out_path):
    try:
        from safetensors.torch import save_file
        print(f"[*] Loading weights from {pt_path}...")
        
        # Load the weights
        state_dict = torch.load(pt_path, map_location="cpu")
        
        # If it's a checkpoint.tar, extract only the model state
        if pt_path.endswith('.tar'):
            state_dict = state_dict.get('model_state_dict', state_dict)

        print(f"[*] Converting to {out_path}...")
        save_file(state_dict, out_path)
        print(f"[+] Success! Model saved as {out_path}")
        
    except ImportError:
        print("[-] Error: The 'safetensors' library is not installed.")
        print("[!] On Termux, 'safetensors' requires Rust to build. Run these commands first:")
        print("    pkg install rust clang")
        print("    pip install safetensors")

if __name__ == "__main__":
    # Example usage for your model
    input_model = "models/native_slm.pt"
    output_model = "models/native_slm.safetensors"
    
    if os.path.exists(input_model):
        convert_to_safetensors(input_model, output_model)
    else:
        print(f"[-] Input model {input_model} not found. Train your model first!")
