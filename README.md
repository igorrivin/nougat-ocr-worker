# Nougat OCR Worker

Distributed OCR processing for arXiv papers using Meta's Nougat model.
Designed to run on vast.ai with results stored in Supabase.

## Performance

| Setup | Speed | 5,784 papers | Cost |
|-------|-------|--------------|------|
| 1x GH200 | 1,240 pp/hr | 47 hours | - |
| 4x RTX 5090 | ~5,000 pp/hr | ~10 hours | ~$12 |

## Quick Start

### 1. Setup Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to SQL Editor and run `supabase_schema.sql`
3. Get your credentials from Settings > API:
   - Project URL (e.g., `https://xxx.supabase.co`)
   - Service role key (use this, not anon key)

### 2. Export Papers to Queue

On your local machine:

```bash
cd /home/igor/devel/nougat-ocr-worker

export SUPABASE_URL="https://xxx.supabase.co"
export SUPABASE_KEY="your-service-role-key"

# Test first
python export_to_supabase.py --dry-run

# Export all papers needing OCR
python export_to_supabase.py
```

### 3. Run on vast.ai

1. Rent a machine:
   - Search for 4x RTX 5090 (or 4090, A100, etc.)
   - Use Docker image: `nvcr.io/nvidia/pytorch:24.05-py3`
   - ~$1.15/hr for 4x RTX 5090

2. SSH into the machine and run:

```bash
# Clone this repo (or upload files)
git clone https://github.com/YOUR_USERNAME/nougat-ocr-worker.git
cd nougat-ocr-worker

# Install dependencies
pip install -r requirements.txt

# Set credentials
export SUPABASE_URL="https://xxx.supabase.co"
export SUPABASE_KEY="your-service-role-key"

# Run with all GPUs
python nougat_worker.py --workers 4 --batch-size 20

# Or run in background
nohup python nougat_worker.py --workers 4 --batch-size 20 > worker.log 2>&1 &
tail -f worker.log
```

### 4. Monitor Progress

Check Supabase dashboard or run:

```sql
SELECT status, COUNT(*)
FROM ocr_queue
GROUP BY status;
```

### 5. Sync Results Back

On your local machine:

```bash
export SUPABASE_URL="https://xxx.supabase.co"
export SUPABASE_KEY="your-service-role-key"

python sync_from_supabase.py
```

## Files

- `nougat_worker.py` - Main worker script (runs on vast.ai)
- `export_to_supabase.py` - Export papers from local DB to Supabase queue
- `sync_from_supabase.py` - Sync completed results back to local DB
- `supabase_schema.sql` - Database schema for Supabase
- `requirements.txt` - Python dependencies

## Architecture

```
Local PostgreSQL          Supabase              vast.ai
┌─────────────┐      ┌─────────────────┐    ┌──────────────┐
│ papers      │      │ ocr_queue       │    │ GPU Workers  │
│ (5,784 need │──────│ (pending jobs)  │────│ (4x RTX 5090)│
│  OCR)       │export│                 │claim│              │
└─────────────┘      │ ocr_results     │────│ Nougat Model │
       ▲             │ (page text)     │save│              │
       │             └─────────────────┘    └──────────────┘
       │ sync               │
       └────────────────────┘
```

## Notes

- Model: `facebook/nougat-small` (~350MB, fast)
- For better quality: `facebook/nougat-base` (~1GB, slower)
- The worker is fault-tolerant and resumable
- Papers are claimed with row-level locking to support multiple workers
