#!/usr/bin/env python3
"""
Sync completed OCR results from Supabase back to local PostgreSQL.

Usage:
    export SUPABASE_URL="https://xxx.supabase.co"
    export SUPABASE_KEY="your-service-key"
    python sync_from_supabase.py
"""

import os
import argparse
import psycopg2
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()


# Local database config
LOCAL_DB = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", 5432)),
    "database": os.environ.get("POSTGRES_DB", "arxiv"),
    "user": os.environ.get("POSTGRES_USER", "arxiv"),
    "password": os.environ.get("POSTGRES_PASSWORD", "arxiv123"),
}

# Supabase config
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


def main():
    parser = argparse.ArgumentParser(description="Sync OCR results from Supabase")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    args = parser.parse_args()

    # Connect to Supabase
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set")
        return

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Get completed papers
    response = supabase.table("ocr_queue").select("arxiv_id").eq("status", "completed").execute()
    completed_arxiv_ids = [r["arxiv_id"] for r in response.data]

    print(f"Found {len(completed_arxiv_ids)} completed papers in Supabase")

    if args.dry_run:
        print(f"Would sync {len(completed_arxiv_ids)} papers")
        return

    # Connect to local database
    conn = psycopg2.connect(**LOCAL_DB)
    cur = conn.cursor()

    synced = 0
    for arxiv_id in tqdm(completed_arxiv_ids, desc="Syncing"):
        # Get OCR results from Supabase
        response = supabase.table("ocr_results").select("*").eq("arxiv_id", arxiv_id).order("page_number").execute()
        pages = response.data

        if not pages:
            continue

        # Combine all pages into one document
        full_text = "\n\n---\n\n".join(p["markdown_text"] for p in pages if p["markdown_text"])

        # Get paper_id from local database
        cur.execute("SELECT id FROM papers WHERE arxiv_id = %s", (arxiv_id,))
        row = cur.fetchone()
        if not row:
            continue
        paper_id = row[0]

        # Update local database - store as latex_content (nougat output is markdown/latex)
        cur.execute("""
            UPDATE papers
            SET nougat_content = %s,
                nougat_status = 'completed',
                needs_reextraction = FALSE
            WHERE id = %s
        """, (full_text, paper_id))

        synced += 1

        if synced % 100 == 0:
            conn.commit()

    conn.commit()
    cur.close()
    conn.close()

    print(f"Synced {synced} papers to local database")


if __name__ == "__main__":
    main()
