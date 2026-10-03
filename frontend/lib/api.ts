/**
 * Backend API client for the RAG pipeline.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface UploadResult {
  message: string;
  filename: string;
  chunks_stored: number;
}

export interface QueryResult {
  answer: string;
  sources: string[];
}

let sessionInitialized = false;

/** Fetch a new session ID from the backend, which sets an HttpOnly cookie. */
export async function ensureSession(): Promise<void> {
  if (sessionInitialized) return;

  try {
    const response = await fetch(`${API_URL}/api/session`, {
      method: "GET",
      credentials: "include",
    });
    if (response.ok) {
      sessionInitialized = true;
    }
  } catch (err) {
    console.error("Failed to initialize session:", err);
  }
}

/** Upload a PDF to the backend for parsing, embedding, and storage. */
export async function uploadPdf(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/api/upload`, {
    method: "POST",
    credentials: "include",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail ?? "Upload failed.");
  }

  return response.json();
}

/** Send a question to the RAG pipeline and get a Groq-generated answer. */
export async function askQuestion(question: string): Promise<QueryResult> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/query`, {
      method: "POST",
      credentials: "include",
      headers: { 
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error(
      "Could not reach the backend. Make sure it is running on port 8000.",
    );
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = error.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg).join(", ")
          : "Query failed.",
    );
  }

  return response.json();
}
