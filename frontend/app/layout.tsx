import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Saarthi RAG",
  description: "PDF upload, retrieval, and Groq-powered Q&A",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
