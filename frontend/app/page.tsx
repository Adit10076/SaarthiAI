import PdfUpload from "@/components/PdfUpload";
import QueryChat from "@/components/QueryChat";

export default function Home() {
  return (
    <main>
      <header>
        <h1>Saarthi RAG</h1>
        <p>Upload PDFs and ask questions powered by your documents.</p>
        <div className="pipeline">
          <span>PDF Upload</span>
          <span className="arrow">↓</span>
          <span>Parsing</span>
          <span className="arrow">↓</span>
          <span>Embedding</span>
          <span className="arrow">↓</span>
          <span>Vector DB</span>
          <span className="arrow">↓</span>
          <span>Retrieval</span>
          <span className="arrow">↓</span>
          <span>Groq Response</span>
        </div>
      </header>

      <div className="grid">
        <PdfUpload />
        <QueryChat />
      </div>
    </main>
  );
}
