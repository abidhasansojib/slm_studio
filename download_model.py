import os
import urllib.request
import time
import sys
import json

def download_file(url, filename, log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log(f"[*] Downloading {os.path.basename(filename)}...")
    start_time = time.time()
    try:
        # User-agent header to prevent 403 errors
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            # Download in chunks to show progress for the large file
            total_size = int(response.getheader('Content-Length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024 # 1MB
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    # Show progress
                    progress_msg = f"\r    Progress: {percent:.2f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)"
                    if log_callback:
                        log_callback(progress_msg, end="")
                    else:
                        print(progress_msg, end="", flush=True)
            
            # Print a newline after progress is done
            if log_callback:
                log_callback("\n", end="")
            else:
                print()
        
        duration = time.time() - start_time
        size = os.path.getsize(filename) / (1024 * 1024)
        log(f"[+] Success! {os.path.basename(filename)} ({size:.2f} MB) downloaded in {duration:.1f}s")
        return True
    except Exception as e:
        log(f"\n[-] Error downloading {os.path.basename(filename)}: {e}")
        # Clean up partial download
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass
        return False

def parse_hf_url(input_str):
    input_str = input_str.strip()
    if not input_str:
        raise ValueError("Input string is empty.")
    
    # Remove trailing slashes
    if input_str.endswith('/'):
        input_str = input_str[:-1]
        
    # Check if it's a URL
    if input_str.startswith("http://") or input_str.startswith("https://"):
        if "huggingface.co/" not in input_str:
             raise ValueError("Not a valid Hugging Face URL.")
             
        # Handle blob/resolve links
        if "/blob/" in input_str:
            input_str = input_str.replace("/blob/", "/resolve/")
            
        parts = input_str.split("huggingface.co/")
        path_part = parts[1]
    else:
        path_part = input_str
        
    # Split path_part
    tokens = [t for t in path_part.split('/') if t]
    if len(tokens) < 2:
        raise ValueError("Could not parse repository owner and name from input.")
        
    owner = tokens[0]
    repo = tokens[1]
    
    # Determine branch and filename if they exist
    branch = "main"
    filename = None
    
    # Check if we have a direct file link
    if len(tokens) > 2:
        if tokens[2] in ["resolve", "blob"]:
            if len(tokens) > 3:
                branch = tokens[3]
            if len(tokens) > 4:
                filename = "/".join(tokens[4:])
        else:
            # Maybe it's owner/repo/filename (less common but possible)
            filename = "/".join(tokens[2:])
            
    base_url = f"https://huggingface.co/{owner}/{repo}/resolve/{branch}"
    model_dir_name = f"{owner}_{repo}".replace("/", "_").replace("\\", "_")
    
    return base_url, model_dir_name, filename, f"{owner}/{repo}"

def download_model(url_or_repo, models_base_dir="models", log_callback=None):
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    try:
        base_url, model_dir_name, primary_file, repo_id = parse_hf_url(url_or_repo)
    except Exception as e:
        log(f"[-] URL Parsing Error: {e}")
        return False

    model_dir = os.path.join(models_base_dir, model_dir_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log("====================================================")
    log(f"  Downloading Model to: {model_dir}")
    log("====================================================")
    
    files_to_download = []
    
    if primary_file:
        # User provided a direct link to a file
        if primary_file.endswith(".gguf"):
            files_to_download = [primary_file]
        else:
            # Likely a safetensors file, download config too
            files_to_download = [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                primary_file
            ]
    else:
        # User provided a repo ID or repo URL, download default set
        # We assume it's a safetensors repo if no file was specified
        files_to_download = [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "model.safetensors"
        ]
    
    # De-duplicate files
    files_to_download = list(dict.fromkeys(files_to_download))

    downloaded_any = False
    success_count = 0
    
    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        target_path = os.path.join(model_dir, filename)
        
        # Ensure subdirectories inside model_dir exist if filename contains slashes
        sub_dir = os.path.dirname(target_path)
        if sub_dir and not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        if os.path.exists(target_path):
            log(f"[!] {filename} already exists. Skipping.")
            downloaded_any = True
            success_count += 1
            continue
        
        success = download_file(url, target_path, log_callback)
        if success:
            downloaded_any = True
            success_count += 1
        elif filename == primary_file or (not primary_file and filename == "model.safetensors"):
            log(f"[-] Critical file {filename} failed to download.")

    if downloaded_any:
        log(f"\n[+] Model {model_dir_name} download attempt finished. ({success_count}/{len(files_to_download)} files)")
        return True
    else:
        log("\n[-] Failed to download any files.")
        if os.path.exists(model_dir) and not os.listdir(model_dir):
            try:
                os.rmdir(model_dir)
            except:
                pass
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_model.py <HuggingFace_URL_or_Repo_ID>")
        print("Example: python download_model.py Qwen/Qwen3-0.6B")
        print("Example: python download_model.py https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors")
        sys.exit(1)
        
    url_or_repo = sys.argv[1]
    download_model(url_or_repo)

if __name__ == "__main__":
    main()
