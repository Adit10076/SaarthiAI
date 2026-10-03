"use client";

import { FormEvent, useState, useEffect } from "react";
import { askQuestion, ensureSession } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

/** Query interface — retrieves context and returns a Groq-generated answer. */
export default function QueryChat() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    ensureSession();
  }, []);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;

    const userMessage: Message = { role: "user", content: question.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const result = await askQuestion(userMessage.content);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "Query failed.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>Ask a Question</h2>
      <p className="subtitle">
        Retrieve relevant chunks and generate an answer via Groq.
      </p>

      <div className="messages">
        {messages.length === 0 && (
          <p className="empty">Upload a PDF first, then ask a question.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <strong>{msg.role === "user" ? "You" : "Saarthi"}</strong>
            <p>{msg.content}</p>
            {msg.sources && msg.sources.length > 0 && (
              <p className="sources">Sources: {msg.sources.join(", ")}</p>
            )}
          </div>
        ))}
        {loading && <p className="loading">Retrieving and thinking…</p>}
      </div>

      <form onSubmit={handleSubmit} className="query-form">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask something about your documents…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          Ask
        </button>
      </form>
    </section>
  );
}
