#!/usr/bin/env python3
"""
Nougat OCR Worker for vast.ai using nougat-ocr package.
Processes arXiv papers and stores results in Supabase.

Requirements (install in order):
    pip install albumentations==1.3.1 pypdfium2==4.16.0 nougat-ocr supabase python-dotenv

Usage:
    python nougat_worker_vast.py --workers 4
    python nougat_worker_vast.py --workers 4 --model base
"""

import os
import subprocess
import tempfile
import time
import argparse
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
import queue

import requests
import fitz  # PyMuPDF for page counting
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()

# Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# GCS bucket for arXiv PDFs (accessible via HTTPS, no rate limits)
ARXIV_GCS_BASE = "https://storage.googleapis.com/arxiv-dataset/arxiv"


def get_gcs_pdf_url(arxiv_id: str, version: str = "v1") -> str:
    """Get GCS URL for a paper's PDF.

    GCS format:
    - New style (YYMM.NNNNN): .../arxiv/pdf/YYMM/YYMM.NNNNNvV.pdf
    - Old style (cat/YYMMNNN): .../cat/pdf/YYMM/YYMMNNNvV.pdf
    """
    if '/' in arxiv_id:  # Old format: math.GT/0401234
        cat, number = arxiv_id.split('/', 1)
        cat_prefix = cat.split('.')[0]  # "math" from "math.GT"
        yymm = number[:4]
        return f"{ARXIV_GCS_BASE}/{cat_prefix}/pdf/{yymm}/{number}{version}.pdf"
    else:  # New format: 1234.56789
        yymm = arxiv_id.split('.')[0][:4]
        return f"{ARXIV_GCS_BASE}/arxiv/pdf/{yymm}/{arxiv_id}{version}.pdf"


def get_supabase():
    """Get Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def download_pdf(arxiv_id: str, output_path: str) -> bool:
    """Download PDF from GCS bucket (with fallback to arxiv.org)."""
    # Try GCS first (no rate limits)
    for version in ["v1", "v2", "v3"]:
        url = get_gcs_pdf_url(arxiv_id, version)
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 200 and resp.content[:4] == b'%PDF':
                with open(output_path, 'wb') as f:
                    f.write(resp.content)
                return True
        except Exception:
            continue

    # Fallback to arxiv.org (rate limited, but works for edge cases)
    try:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200 and resp.content[:4] == b'%PDF':
            with open(output_path, 'wb') as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"Download error for {arxiv_id}: {e}")

    return False


def count_pdf_pages(pdf_path: str) -> int:
    """Count pages in a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def run_nougat(pdf_path: str, output_dir: str, model: str = "small", gpu_id: int = 0) -> dict:
    """Run nougat CLI on a PDF file."""
    result = {
        "success": False,
        "pages": [],
        "error": None,
        "processing_time": 0
    }

    start_time = time.time()

    try:
        # Run nougat CLI - it may segfault after writing output, so we check for output
        cmd = [
            "nougat", pdf_path,
            "-o", output_dir,
            "--no-skipping",
            "-m", f"0.1.0-{model}"
        ]

        # Explicitly set CUDA_VISIBLE_DEVICES in subprocess environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Run with timeout, ignore segfault since output is written before crash
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min timeout per paper
            env=env
        )

        # Check for output file regardless of exit code (segfault writes output first)
        pdf_name = Path(pdf_path).stem
        output_file = Path(output_dir) / f"{pdf_name}.mmd"

        if output_file.exists():
            with open(output_file, 'r') as f:
                full_text = f.read()

            if full_text.strip():
                # Split by page markers if present, otherwise treat as single page
                # Nougat doesn't always add page markers, so we store as one block
                result["pages"] = [{
                    "page_number": 0,
                    "text": full_text,
                    "char_count": len(full_text)
                }]
                result["success"] = True
            else:
                result["error"] = "Empty output"
        else:
            result["error"] = process.stderr or "No output file generated"

    except subprocess.TimeoutExpired:
        result["error"] = "Processing timeout"
    except Exception as e:
        result["error"] = str(e)

    result["processing_time"] = time.time() - start_time
    return result


def gpu_worker(gpu_id: int, task_queue: Queue, result_queue: Queue, model: str):
    """Worker process for a single GPU."""
    print(f"[GPU {gpu_id}] Starting...")

    while True:
        try:
            task = task_queue.get(timeout=60)
            if task is None:  # Shutdown signal
                break

            arxiv_id = task

            # Create temp directory for this paper
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = os.path.join(tmpdir, f"{arxiv_id.replace('/', '_')}.pdf")
                output_dir = os.path.join(tmpdir, "output")
                os.makedirs(output_dir, exist_ok=True)

                # Download PDF
                if not download_pdf(arxiv_id, pdf_path):
                    result_queue.put({
                        "arxiv_id": arxiv_id,
                        "success": False,
                        "error": "PDF download failed",
                        "pages": [],
                        "page_count": 0,
                        "processing_time": 0
                    })
                    continue

                # Count pages before processing
                page_count = count_pdf_pages(pdf_path)

                # Run nougat on specific GPU
                result = run_nougat(pdf_path, output_dir, model, gpu_id)
                result["arxiv_id"] = arxiv_id
                result["page_count"] = page_count
                result_queue.put(result)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {e}")
            continue

    print(f"[GPU {gpu_id}] Shutting down")


def save_results_to_supabase(supabase, result: dict, worker_id: str):
    """Save OCR results to Supabase."""
    arxiv_id = result["arxiv_id"]
    page_count = result.get("page_count", 0)

    if result["success"]:
        # Save page results
        for page in result["pages"]:
            supabase.table("ocr_results").upsert({
                "arxiv_id": arxiv_id,
                "page_number": page["page_number"],
                "markdown_text": page["text"],
                "char_count": page["char_count"]
            }, on_conflict="arxiv_id,page_number").execute()

        # Update queue status with page_count and worker_id for analysis
        supabase.table("ocr_queue").update({
            "status": "completed",
            "completed_at": "now()",
            "page_count": page_count,
            "processing_time_seconds": result["processing_time"],
            "worker_id": worker_id
        }).eq("arxiv_id", arxiv_id).execute()
    else:
        # Mark as failed
        supabase.table("ocr_queue").update({
            "status": "failed",
            "error_message": result["error"],
            "worker_id": worker_id
        }).eq("arxiv_id", arxiv_id).execute()


def main():
    parser = argparse.ArgumentParser(description="Nougat OCR Worker (vast.ai version)")
    parser.add_argument("--workers", type=int, default=4, help="Number of GPU workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Papers to claim per batch")
    parser.add_argument("--worker-id", type=str, default=None, help="Unique worker ID")
    parser.add_argument("--gpu-type", type=str, default="unknown",
                        help="GPU type for cost analysis (e.g., '4xRTX5080', '4xRTX5090')")
    parser.add_argument("--model", type=str, default="base", choices=["small", "base"],
                        help="Nougat model size (default: base)")
    args = parser.parse_args()

    print(f"Starting with {args.workers} workers using nougat-{args.model} on {args.gpu_type}")

    # Worker ID includes GPU type for cost analysis
    worker_id = args.worker_id or f"{args.gpu_type}-{os.getpid()}"
    supabase = get_supabase()

    # Create queues
    task_queue = Queue()
    result_queue = Queue()

    # Start GPU workers
    workers = []
    for gpu_id in range(args.workers):
        p = Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue, args.model))
        p.start()
        workers.append(p)

    # Give workers time to initialize
    time.sleep(5)

    # Main loop
    total_processed = 0
    total_success = 0
    total_failed = 0

    try:
        while True:
            # Claim papers from Supabase
            response = supabase.rpc("claim_papers", {
                "p_worker_id": worker_id,
                "p_batch_size": args.batch_size
            }).execute()

            papers = response.data
            if not papers:
                print("No more papers to process. Waiting 60s...")
                time.sleep(60)
                continue

            print(f"Claimed {len(papers)} papers")

            # Queue papers for processing
            for paper in papers:
                task_queue.put(paper["arxiv_id"])

            # Collect results
            collected = 0
            total_pages = 0
            batch_start = time.time()
            while collected < len(papers):
                try:
                    result = result_queue.get(timeout=1260)  # 21 min timeout (slightly > processing timeout)
                    save_results_to_supabase(supabase, result, worker_id)
                    collected += 1
                    total_processed += 1

                    if result["success"]:
                        total_success += 1
                        page_count = result.get("page_count", 0)
                        total_pages += page_count
                        pages_per_sec = page_count / result["processing_time"] if result["processing_time"] > 0 else 0
                        status = f"OK {page_count}pp in {result['processing_time']:.1f}s ({pages_per_sec:.2f} pp/s)"
                    else:
                        total_failed += 1
                        status = f"FAILED: {result['error']}"

                    print(f"[{total_processed}] {result['arxiv_id']}: {status}")

                except queue.Empty:
                    print("Timeout waiting for results")
                    break

            batch_elapsed = time.time() - batch_start
            batch_pps = total_pages / batch_elapsed if batch_elapsed > 0 else 0
            print(f"Progress: {total_success} success, {total_failed} failed | Batch: {total_pages} pages in {batch_elapsed:.1f}s ({batch_pps:.1f} pp/s)")

    except KeyboardInterrupt:
        print("\nShutting down...")

    # Shutdown workers
    for _ in workers:
        task_queue.put(None)
    for w in workers:
        w.join(timeout=10)

    print(f"\nTotal: {total_processed} processed, {total_success} success, {total_failed} failed")


if __name__ == "__main__":
    main()
