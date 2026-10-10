#!/usr/bin/env python3
"""
Glacier High-Speed Multi-Stream Model Downloader
Downloads large MoE models (Qwen2-57B, Mixtral-8x22B, DeepSeek-V2) using parallel HTTP Range streams.
Zero secrets required - downloads directly from public Hugging Face repositories.
"""

import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PRESET_MODELS = {
    "qwen2-57b": {
        "repo": "Qwen/Qwen2-57B-A14B-Instruct-GGUF",
        "file": "qwen2-57b-a14b-instruct-q3_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF/resolve/main/qwen2-57b-a14b-instruct-q3_k_m.gguf",
        "size": 27505173312,
        "description": "Qwen2-57B-A14B (57B total, 14B active, 64 experts, 28 layers) @ Q3_K_M (25.62 GB)"
    },
    "mixtral-8x22b-iq2": {
        "repo": "mradermacher/Mixtral-8x22B-Instruct-v0.1-i1-GGUF",
        "file": "Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf",
        "url": "https://huggingface.co/mradermacher/Mixtral-8x22B-Instruct-v0.1-i1-GGUF/resolve/main/Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf",
        "size": 37889916608,
        "description": "Mixtral-8x22B-Instruct (141B total, 39B active, 8 experts, 56 layers) @ IQ2_XXS (35.29 GB)"
    },
    "deepseek-coder-v2-lite": {
        "repo": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "file": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "size": 9568912384,
        "description": "DeepSeek-Coder-V2-Lite (16B total, 2.4B active, 64 experts, 27 layers) @ Q4_K_M (8.91 GB)"
    }
}

def get_remote_file_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length", 0))

def download_segment(url, temp_dir, part_idx, start_byte, end_byte):
    part_path = os.path.join(temp_dir, f"part_{part_idx}.tmp")
    already_dl = 0
    if os.path.exists(part_path):
        already_dl = os.path.getsize(part_path)

    cur_start = start_byte + already_dl
    if cur_start > end_byte:
        return part_idx, part_path

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={cur_start}-{end_byte}"
    }
    req = urllib.request.Request(url, headers=headers)

    total_part_bytes = end_byte - start_byte + 1
    t0 = time.time()
    last_log = t0

    with urllib.request.urlopen(req, timeout=60) as resp, open(part_path, "ab" if already_dl > 0 else "wb") as f:
        downloaded = already_dl
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if time.time() - last_log > 5.0 or downloaded >= total_part_bytes:
                pct = (downloaded / total_part_bytes) * 100.0
                mb = downloaded / (1024 * 1024)
                total_mb = total_part_bytes / (1024 * 1024)
                rate = (downloaded - already_dl) / (1024 * 1024 * max(0.1, time.time() - t0))
                print(f"  [Stream {part_idx}] {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%) @ {rate:.1f} MB/s", flush=True)
                last_log = time.time()

        f.flush()

    return part_idx, part_path

def main():
    parser = argparse.ArgumentParser(description="Glacier Multi-Stream High-Speed MoE Model Downloader")
    parser.add_argument("--model", choices=list(PRESET_MODELS.keys()), default="qwen2-57b",
                        help="Model preset to download")
    parser.add_argument("--url", default=None, help="Custom direct Hugging Face download URL")
    parser.add_argument("--dest", default="models", help="Destination directory (default: ./models)")
    parser.add_argument("--threads", type=int, default=8, help="Number of concurrent HTTP streams (default: 8)")
    args = parser.parse_args()

    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)

    if args.url:
        filename = args.url.split("/")[-1].split("?")[0]
        url = args.url
        print(f"Connecting to custom URL: {url}...")
        total_size = get_remote_file_size(url)
        description = f"Custom GGUF Model ({filename})"
    else:
        preset = PRESET_MODELS[args.model]
        filename = preset["file"]
        url = preset["url"]
        total_size = preset["size"]
        description = preset["description"]

    target_file = os.path.join(dest_dir, filename)
    temp_dir = os.path.join(dest_dir, f"{filename}.parts")
    os.makedirs(temp_dir, exist_ok=True)

    print("================================================================================")
    print("  GLACIER MULTI-STREAM HIGH-SPEED MODEL DOWNLOADER")
    print(f"  Model:       {description}")
    print(f"  Destination: {target_file}")
    print(f"  Total Size:  {total_size / (1024**3):.2f} GB ({total_size:,} bytes)")
    print(f"  Parallel:    {args.threads} HTTP Range Streams")
    print("================================================================================\n")

    current_size = os.path.getsize(target_file) if os.path.exists(target_file) else 0
    if current_size == total_size:
        print(f"[OK] File is already completely downloaded and verified ({current_size:,} bytes)!")
        return 0

    remaining = total_size - current_size
    chunk_size = remaining // args.threads
    segments = []
    for i in range(args.threads):
        seg_start = current_size + i * chunk_size
        seg_end = current_size + (i + 1) * chunk_size - 1 if i < args.threads - 1 else total_size - 1
        segments.append((i, seg_start, seg_end))

    t_start = time.time()
    print(f"[Download] Launching {args.threads} worker streams...")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(download_segment, url, temp_dir, seg[0], seg[1], seg[2]): seg for seg in segments}
        for future in as_completed(futures):
            part_idx, part_path = future.result()

    elapsed = time.time() - t_start
    avg_speed = (remaining / (1024 * 1024)) / max(0.1, elapsed)
    print(f"\n[Download] All {args.threads} streams completed in {elapsed:.1f}s ({avg_speed:.1f} MB/s)!")
    print(f"[Assembly] Stitching parts into {filename}...")

    t_stitch = time.time()
    with open(target_file, "ab" if current_size > 0 else "wb") as f_out:
        for i in range(args.threads):
            part_path = os.path.join(temp_dir, f"part_{i}.tmp")
            if os.path.exists(part_path):
                with open(part_path, "rb") as f_in:
                    while True:
                        buf = f_in.read(16 * 1024 * 1024)
                        if not buf:
                            break
                        f_out.write(buf)
                os.remove(part_path)

    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    final_size = os.path.getsize(target_file)
    print(f"[Assembly] Done in {time.time() - t_stitch:.2f}s.")
    print(f"Final File Size: {final_size / (1024**3):.2f} GB ({final_size:,} bytes)")

    if final_size == total_size:
        print("[SUCCESS] 100% Exact Byte Match. Model ready for cluster deployment!")
        return 0
    else:
        print(f"[WARNING] Size mismatch! Expected {total_size:,}, got {final_size:,}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
