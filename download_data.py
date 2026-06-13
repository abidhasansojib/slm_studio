import os
import urllib.request
import time

def download_file(url, filename):
    print(f"[*] Downloading {filename}...")
    start_time = time.time()
    temp_filename = filename + ".tmp"
    try:
        # User-agent header to prevent 403 errors from some servers
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(temp_filename, 'wb') as out_file:
            chunk_size = 1024 * 1024  # 1MB
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        
        # Swap temporary file with real file name
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_filename, filename)
        
        duration = time.time() - start_time
        size = os.path.getsize(filename) / (1024 * 1024)
        print(f"[+] Success! {filename} ({size:.2f} MB) downloaded in {duration:.1f}s")
        return True
    except Exception as e:
        print(f"[-] Error downloading {filename}: {e}")
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass
        return False

def main():
    data_dir = "data_corpus"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Lightweight training dataset optimized for mobile training
    datasets = [
        # --- THE CORE: TINY STORIES (High Quality Reasoning) ---
        {
            "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
            "name": "tinystories_valid.txt"
        },
        
        # --- CLASSIC LITERATURE (Grammar & Style) ---
        {
            "url": "https://www.gutenberg.org/files/11/11-0.txt",
            "name": "alice_in_wonderland.txt"
        },
        {
            "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
            "name": "sherlock_holmes.txt"
        },
        {
            "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
            "name": "pride_and_prejudice.txt"
        },
        {
            "url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
            "name": "moby_dick.txt"
        },
        {
            "url": "https://www.gutenberg.org/files/84/84-0.txt",
            "name": "frankenstein.txt"
        },
        {
            "url": "https://www.gutenberg.org/files/98/98-0.txt",
            "name": "a_tale_of_two_cities.txt"
        },
        
        # --- KNOWLEDGE & ENCYCLOPEDIC (Facts) ---
        {
            "url": "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt",
            "name": "wikitext_valid.txt" # Using clean public validation set of wikitext
        },
        
        # --- TECHNICAL & PHILOSOPHY (Logic) ---
        {
            "url": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
            "name": "philosophy_basics_republic.txt" # Plato's Republic
        },
        {
            "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "name": "tinyshakespeare.txt" # High-quality literary patterns
        }
    ]

    print("====================================================")
    print("      SLM Data Acquisition Script (Lightweight)      ")
    print("====================================================")
    print(f"Targeting {len(datasets)} data sources for mobile training.\n")
    
    for ds in datasets:
        target_path = os.path.join(data_dir, ds["name"])
        if os.path.exists(target_path):
            print(f"[!] {ds['name']} already exists. Skipping.")
            continue
        
        download_file(ds["url"], target_path)

    print("\n[+] Data acquisition complete!")
    print("[i] Total corpus size is approx. 26MB (perfect for mobile training).")
    print("[i] Start training via the Web UI (Port 5000) or python model.py.")

if __name__ == "__main__":
    main()
