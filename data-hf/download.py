import os
import requests
import tarfile
import zstandard as zstd

BASE_URL = "https://huggingface.co/datasets/linrock/test80-2024/resolve/main"

FILES = [
    "test80-2024-01-jan-2tb7p.tar.zst",
    "test80-2024-02-feb-2tb7p.tar.zst",
    "test80-2024-03-mar-2tb7p.tar.zst",
    "test80-2024-04-apr-2tb7p.tar.zst",
    "test80-2024-05-may-2tb7p.tar.zst",
    "test80-2024-06-jun-2tb7p.tar.zst",
    "test80-2024-07-jul-2tb7p.tar.zst",
    "test80-2024-08-aug-2tb7p.tar.zst",
    "test80-2024-09-sep-2tb7p.tar.zst",
]

OUTPUT_DIR = "data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, path):
    print(f"Downloading {os.path.basename(path)}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        if total_size > 0:
            print(f"  Total size: {total_size / (1024**3):.2f} GB")
        
        downloaded = 0
        chunk_size = 8192
        
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress_pct = (downloaded / total_size) * 100
                        downloaded_gb = downloaded / (1024**3)
                        total_gb = total_size / (1024**3)
                        print(f"  Progress: {downloaded_gb:.2f}/{total_gb:.2f} GB ({progress_pct:.1f}%)", end='\r')
        
        print()  # New line after progress
        print(f"  Download complete: {downloaded / (1024**3):.2f} GB")

def decompress_zst(zst_path, tar_path):
    print(f"Decompressing {os.path.basename(zst_path)}...")
    file_size = os.path.getsize(zst_path)
    print(f"  File size: {file_size / (1024**3):.2f} GB")
    
    with open(zst_path, "rb") as compressed, open(tar_path, "wb") as out:
        dctx = zstd.ZstdDecompressor()
        dctx.copy_stream(compressed, out, write_size=262144)
    
    decompressed_size = os.path.getsize(tar_path)
    print(f"  Decompressed to: {decompressed_size / (1024**3):.2f} GB")

def extract_tar(tar_path, dest_dir):
    print(f"Extracting {os.path.basename(tar_path)}...")
    file_size = os.path.getsize(tar_path)
    print(f"  Archive size: {file_size / (1024**3):.2f} GB")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest_dir)
    print(f"  Extraction complete")

for filename in FILES:
    zst_path = os.path.join(OUTPUT_DIR, filename)
    tar_path = zst_path[:-4]
    month_dir = os.path.join(OUTPUT_DIR, filename.split("-")[2])
    os.makedirs(month_dir, exist_ok=True)

    url = f"{BASE_URL}/{filename}"
    if not os.path.exists(zst_path):
        download_file(url, zst_path)

    if not os.path.exists(tar_path):
        decompress_zst(zst_path, tar_path)

    extract_tar(tar_path, month_dir)

    print(f"Deleting {os.path.basename(tar_path)}...")
    os.remove(tar_path)
    
    print(f"Deleting {os.path.basename(zst_path)}...")
    os.remove(zst_path)

print("All files downloaded, extracted, and cleaned up.")
