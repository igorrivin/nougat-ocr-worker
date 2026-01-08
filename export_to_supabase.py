#!/usr/bin/env python3
"""
Export papers needing OCR from local PostgreSQL to Supabase queue.

Usage:
    export SUPABASE_URL="https://xxx.supabase.co"
    export SUPABASE_KEY="your-service-key"
    python export_to_supabase.py
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
    parser = argparse.ArgumentParser(description="Export papers to Supabase OCR queue")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be exported")
    args = parser.parse_args()

    # Connect to local database
    conn = psycopg2.connect(**LOCAL_DB)
    cur = conn.cursor()

    # Get papers needing OCR
    query = """
        SELECT arxiv_id, title
        FROM papers
        WHERE needs_reextraction = TRUE
        ORDER BY submitted_date DESC
    """
    if args.limit:
        query += f" LIMIT {args.limit}"

    cur.execute(query)
    papers = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Found {len(papers)} papers needing OCR")

    if args.dry_run:
        print("\nWould export:")
        for arxiv_id, title in papers[:10]:
            print(f"  {arxiv_id}: {title[:60]}...")
        if len(papers) > 10:
            print(f"  ... and {len(papers) - 10} more")
        return

    # Connect to Supabase
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set")
        return

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Export in batches
    batch_size = 100
    for i in tqdm(range(0, len(papers), batch_size), desc="Exporting"):
        batch = papers[i:i + batch_size]
        records = [
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "status": "pending"
            }
            for arxiv_id, title in batch
        ]

        # Upsert to avoid duplicates
        supabase.table("ocr_queue").upsert(
            records,
            on_conflict="arxiv_id"
        ).execute()

    print(f"Exported {len(papers)} papers to Supabase ocr_queue")


if __name__ == "__main__":
    main()
