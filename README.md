# Nougat OCR Worker

Distributed OCR processing for arXiv papers using Meta's Nougat model.
Designed to run on vast.ai with results stored in Supabase.

## Performance

Measured with `nougat-base` model:

| Setup | Speed | Pages/hr | 5,784 papers (~100K pages) | Cost |
|-------|-------|----------|---------------------------|------|
| 4x RTX 5080 | ~0.13 pp/s/GPU | ~1,800 | ~60 hours | ~$31 |
| 4x RTX 5090 | ~0.2 pp/s/GPU (est) | ~2,800 | ~40 hours | ~$50 |

- Average: **~7-8 seconds per page**
- Throughput is memory-bandwidth bound, not compute bound
- GPU utilization typically 20-35%

### Full Math arXiv Corpus (720K papers)

| Setup | Time | Cost (est) |
|-------|------|------------|
| 1x 4xRTX5080 box | ~7,400 hours | ~$3,800 |
| 10x parallel boxes | ~740 hours | ~$3,800 |

## Quick Start

### 1. Setup Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to SQL Editor and run `supabase_schema.sql`
3. Get your credentials from Settings > API:
   - Project URL (e.g., `https://xxx.supabase.co`)
   - Service role key (use this, not anon key)

### 2. Export Papers to Queue

On your local machine, create a `.env` file:

```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-service-role-key
```

Then export:

```bash
# Test first
python export_to_supabase.py --dry-run

# Export all papers needing OCR
python export_to_supabase.py
```

### 3. Run on vast.ai

1. Rent a machine:
   - Search for 4x RTX 5080/5090 (or 4090, A100, etc.)
   - Use Docker image: `nvcr.io/nvidia/pytorch:24.05-py3`
   - 4x RTX 5080: ~$0.52/hr
   - 4x RTX 5090: ~$1.20/hr

2. SSH into the machine and run:

```bash
# Clone repo
git clone https://github.com/igorrivin/nougat-ocr-worker.git
cd nougat-ocr-worker

# Install dependencies (order matters!)
pip install albumentations==1.3.1 pypdfium2==4.16.0 nougat-ocr
pip install supabase python-dotenv tqdm PyMuPDF requests

# Create .env file
cat > .env << 'EOF'
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-service-role-key
EOF

# Run with all GPUs (use --gpu-type for cost analysis)
python nougat_worker_vast.py --workers 4 --batch-size 12 --model base --gpu-type 4xRTX5080
```

PDFs are downloaded from the GCS mirror (`gs://arxiv-dataset`) to avoid rate limiting.

### 4. Monitor Progress

Check Supabase dashboard or run:

```sql
-- Overall progress
SELECT status, COUNT(*), SUM(page_count) as pages
FROM ocr_queue
GROUP BY status;

-- Performance by GPU type
SELECT
  split_part(worker_id, '-', 1) as gpu_type,
  COUNT(*) as papers,
  SUM(page_count) as pages,
  ROUND(SUM(processing_time_seconds)::numeric, 0) as total_seconds,
  ROUND((SUM(page_count) / NULLIF(SUM(processing_time_seconds), 0))::numeric, 3) as pages_per_sec
FROM ocr_queue
WHERE status = 'completed' AND page_count > 0
GROUP BY split_part(worker_id, '-', 1);
```

### 5. Sync Results Back

On your local machine:

```bash
python sync_from_supabase.py
```

## Files

- `nougat_worker_vast.py` - Worker script for vast.ai (uses nougat-ocr package)
- `nougat_worker.py` - Alternative worker using transformers directly (for GH200/other setups)
- `export_to_supabase.py` - Export papers from local DB to Supabase queue
- `sync_from_supabase.py` - Sync completed results back to local DB
- `supabase_schema.sql` - Database schema for Supabase
- `Dockerfile.vast` - Docker image for easy deployment
- `requirements.txt` - Python dependencies

## Docker Deployment

Build and push (from an x86 machine or vast.ai):

```bash
docker build -f Dockerfile.vast -t yourusername/nougat-worker:latest .
docker push yourusername/nougat-worker:latest
```

Then on vast.ai, use `yourusername/nougat-worker:latest` as the Docker image.

## Architecture

```
Local PostgreSQL          Supabase              vast.ai
┌─────────────┐      ┌─────────────────┐    ┌──────────────┐
│ papers      │      │ ocr_queue       │    │ GPU Workers  │
│ (papers     │──────│ (pending jobs)  │────│ (4x RTX 5080)│
│  need OCR)  │export│                 │claim│              │
└─────────────┘      │ ocr_results     │────│ Nougat Model │
       ▲             │ (markdown text) │save│              │
       │             └─────────────────┘    └──────────────┘
       │ sync               │
       └────────────────────┘
```

## Notes

- Model: `nougat-base` recommended for math papers (~1GB, better LaTeX quality)
- Alternative: `nougat-small` (~350MB, faster but lower quality)
- Worker is fault-tolerant and resumable
- Papers are claimed with row-level locking to support multiple parallel workers
- The `--gpu-type` flag tags results for cost/performance analysis across hardware
