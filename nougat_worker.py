#!/usr/bin/env python3
"""
Nougat OCR Worker for vast.ai
Processes arXiv papers and stores results in Supabase.

Usage:
    python nougat_worker.py --workers 4
    python nougat_worker.py --workers 4 --batch-size 20
"""

import os
import io
import time
import argparse
import warnings
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
import queue

warnings.filterwarnings("ignore")

import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel
from PIL import Image
import fitz  # PyMuPDF
from supabase import create_client
from tqdm import tqdm


# Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
MODEL_NAME = "facebook/nougat-small"  # or facebook/nougat-base for better quality


def get_supabase():
    """Get Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def download_pdf(arxiv_id: str) -> bytes | None:
    """Download PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.content
        return None
    except Exception:
        return None


def process_paper_on_gpu(gpu_id: int, arxiv_id: str, pdf_bytes: bytes,
                          processor, model) -> dict:
    """Process a single paper on a specific GPU."""
    result = {
        "arxiv_id": arxiv_id,
        "success": False,
        "pages": [],
        "error": None,
        "processing_time": 0
    }

    start_time = time.time()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        result["page_count"] = len(doc)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            pixel_values = processor(img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(f"cuda:{gpu_id}")

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    max_new_tokens=4096,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                )

            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            result["pages"].append({
                "page_number": page_num,
                "text": text,
                "char_count": len(text)
            })

        doc.close()
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    result["processing_time"] = time.time() - start_time
    return result


def gpu_worker(gpu_id: int, task_queue: Queue, result_queue: Queue):
    """Worker process for a single GPU."""
    print(f"[GPU {gpu_id}] Initializing...")

    # Load model on this GPU
    processor = NougatProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model = model.to(f"cuda:{gpu_id}")
    model.eval()

    print(f"[GPU {gpu_id}] Ready")

    while True:
        try:
            task = task_queue.get(timeout=30)
            if task is None:  # Shutdown signal
                break

            arxiv_id, pdf_bytes = task
            result = process_paper_on_gpu(gpu_id, arxiv_id, pdf_bytes, processor, model)
            result_queue.put(result)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Error: {e}")
            continue

    print(f"[GPU {gpu_id}] Shutting down")


def save_results_to_supabase(supabase, result: dict):
    """Save OCR results to Supabase."""
    arxiv_id = result["arxiv_id"]

    if result["success"]:
        # Save page results
        for page in result["pages"]:
            supabase.table("ocr_results").upsert({
                "arxiv_id": arxiv_id,
                "page_number": page["page_number"],
                "markdown_text": page["text"],
                "char_count": page["char_count"]
            }, on_conflict="arxiv_id,page_number").execute()

        # Update queue status
        supabase.table("ocr_queue").update({
            "status": "completed",
            "completed_at": "now()",
            "page_count": result.get("page_count"),
            "processing_time_seconds": result["processing_time"]
        }).eq("arxiv_id", arxiv_id).execute()
    else:
        # Mark as failed
        supabase.table("ocr_queue").update({
            "status": "failed",
            "error_message": result["error"]
        }).eq("arxiv_id", arxiv_id).execute()


def main():
    parser = argparse.ArgumentParser(description="Nougat OCR Worker")
    parser.add_argument("--workers", type=int, default=4, help="Number of GPU workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Papers to claim per batch")
    parser.add_argument("--worker-id", type=str, default=None, help="Unique worker ID")
    args = parser.parse_args()

    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available!")
        return

    args.workers = min(args.workers, num_gpus)
    print(f"Using {args.workers} GPUs out of {num_gpus} available")

    worker_id = args.worker_id or f"worker-{os.getpid()}"
    supabase = get_supabase()

    # Create queues
    task_queue = Queue()
    result_queue = Queue()

    # Start GPU workers
    workers = []
    for gpu_id in range(args.workers):
        p = Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue))
        p.start()
        workers.append(p)

    # Give workers time to initialize
    time.sleep(10)

    # Main loop
    total_processed = 0
    try:
        while True:
            # Claim papers from Supabase
            response = supabase.rpc("claim_papers", {
                "p_worker_id": worker_id,
                "p_batch_size": args.batch_size
            }).execute()

            papers = response.data
            if not papers:
                print("No more papers to process. Waiting...")
                time.sleep(60)
                continue

            print(f"Claimed {len(papers)} papers")

            # Download PDFs and queue for processing
            for paper in papers:
                arxiv_id = paper["arxiv_id"]
                pdf_bytes = download_pdf(arxiv_id)

                if pdf_bytes:
                    task_queue.put((arxiv_id, pdf_bytes))
                else:
                    # Mark as failed
                    supabase.table("ocr_queue").update({
                        "status": "failed",
                        "error_message": "PDF download failed"
                    }).eq("arxiv_id", arxiv_id).execute()

            # Collect results
            collected = 0
            while collected < len(papers):
                try:
                    result = result_queue.get(timeout=300)  # 5 min timeout per paper
                    save_results_to_supabase(supabase, result)
                    collected += 1
                    total_processed += 1

                    status = "OK" if result["success"] else f"FAILED: {result['error']}"
                    print(f"[{total_processed}] {result['arxiv_id']}: {status} "
                          f"({result['processing_time']:.1f}s)")

                except queue.Empty:
                    print("Timeout waiting for results")
                    break

    except KeyboardInterrupt:
        print("\nShutting down...")

    # Shutdown workers
    for _ in workers:
        task_queue.put(None)
    for w in workers:
        w.join(timeout=10)

    print(f"Total processed: {total_processed}")


if __name__ == "__main__":
    main()
