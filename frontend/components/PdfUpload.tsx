"use client";

import { useState, useEffect } from "react";
import { uploadPdf, ensureSession } from "@/lib/api";

/** PDF upload form — sends file to backend for parsing and embedding. */
export default function PdfUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    ensureSession();
  }, []);

  async function handleUpload() {
    if (!file) return;

    setLoading(true);
    setStatus(null);

    try {
      const result = await uploadPdf(file);
      setStatus(
        `${result.filename} uploaded — ${result.chunks_stored} chunks stored.`
      );
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Upload PDF</h2>
      <p className="subtitle">
        Parse with LlamaParse, embed with BGE-M3, store in Qdrant.
      </p>

      <input
        type="file"
        accept=".pdf"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />

      <button onClick={handleUpload} disabled={!file || loading}>
        {loading ? "Processing…" : "Upload & Ingest"}
      </button>

      {status && <p className="status">{status}</p>}
    </section>
  );
}
