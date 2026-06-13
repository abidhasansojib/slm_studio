import os
import urllib.request
import time

def download_file(url, filename):
    print(f"[*] Downloading {filename}...")
    start_time = time.time()
    try:
        # User-agent header to prevent 403 errors from some servers
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
            out_file.write(response.read())
        
        duration = time.time() - start_time
        size = os.path.getsize(filename) / (1024 * 1024)
        print(f"[+] Success! {filename} ({size:.2f} MB) downloaded in {duration:.1f}s")
    except Exception as e:
        print(f"[-] Error downloading {filename}: {e}")

def main():
    data_dir = "data_corpus"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Comprehensive list of training data for a 50M parameter model
    datasets = [
        # --- THE CORE: TINY STORIES (High Quality Reasoning) ---
        {
            "url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
            "name": "tinystories_train.txt"
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
            "url": "https://huggingface.co/datasets/sedthh/wikipedia-24-09-simple-corpus/resolve/main/simple_wikipedia.txt",
            "name": "wikitext.txt" # Using high quality simple-wikipedia
        },
        
        # --- TECHNICAL & PHILOSOPHY (Logic) ---
        {
            "url": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
            "name": "philosophy_basics_republic.txt" # Plato's Republic
        },
        {
            "url": "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
            "name": "dictionary_full.txt" # English Word List
        }
    ]

    print("====================================================")
    print("      SLM Data Acquisition Script (Full Corpus)      ")
    print("====================================================")
    print(f"Targeting {len(datasets)} data sources for 50M model training.\n")
    
    for ds in datasets:
        target_path = os.path.join(data_dir, ds["name"])
        if os.path.exists(target_path):
            print(f"[!] {ds['name']} already exists. Skipping.")
            continue
        
        download_file(ds["url"], target_path)

    print("\n[+] Data acquisition complete!")
    print("[i] Total corpus size will be approx. 500MB - 800MB.")
    print("[i] Start training via the Web UI (Port 5000).")

if __name__ == "__main__":
    main()
