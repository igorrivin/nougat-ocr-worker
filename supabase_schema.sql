-- Supabase schema for Nougat OCR results
-- Run this in Supabase SQL Editor

-- Table to track papers needing OCR
CREATE TABLE IF NOT EXISTS ocr_queue (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT,
    pdf_url TEXT,
    status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
    worker_id TEXT,                 -- which worker is processing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    page_count INTEGER,
    processing_time_seconds REAL
);

-- Table to store OCR results
CREATE TABLE IF NOT EXISTS ocr_results (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    markdown_text TEXT,
    char_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(arxiv_id, page_number)
);

-- Table to store extracted elements (optional, for structured extraction)
CREATE TABLE IF NOT EXISTS ocr_elements (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT NOT NULL,
    element_type TEXT,  -- definition, theorem, lemma, proposition, corollary, proof
    label TEXT,
    statement TEXT,
    proof TEXT,
    page_number INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_ocr_queue_status ON ocr_queue(status);
CREATE INDEX IF NOT EXISTS idx_ocr_queue_arxiv_id ON ocr_queue(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_ocr_results_arxiv_id ON ocr_results(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_ocr_elements_arxiv_id ON ocr_elements(arxiv_id);

-- Function to claim a batch of papers for processing
CREATE OR REPLACE FUNCTION claim_papers(p_worker_id TEXT, p_batch_size INTEGER DEFAULT 10)
RETURNS TABLE(arxiv_id TEXT, pdf_url TEXT) AS $$
BEGIN
    RETURN QUERY
    WITH claimed AS (
        UPDATE ocr_queue
        SET status = 'processing',
            worker_id = p_worker_id,
            started_at = NOW()
        WHERE id IN (
            SELECT id FROM ocr_queue
            WHERE status = 'pending'
            ORDER BY id
            LIMIT p_batch_size
            FOR UPDATE SKIP LOCKED
        )
        RETURNING ocr_queue.arxiv_id, ocr_queue.pdf_url
    )
    SELECT * FROM claimed;
END;
$$ LANGUAGE plpgsql;
