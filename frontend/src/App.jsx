import React, { useEffect, useMemo, useRef, useState } from "react";

const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/+$/, "") || "http://localhost:8000";

function confidenceFromScore(score) {
  if (score >= 0.6) return { label: "High", tone: "text-green-700" };
  if (score >= 0.45) return { label: "Medium", tone: "text-yellow-700" };
  return { label: "Low", tone: "text-red-700" };
}

function prettyFile(name) {
  return name || "";
}

export default function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [docs, setDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [uploading, setUploading] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [sending, setSending] = useState(false);

  const [fileToUpload, setFileToUpload] = useState(null);

  const [messages, setMessages] = useState([
    {
      id: crypto.randomUUID(),
      role: "user",
      text: "What is this document about?",
      meta: { doc: "" },
    },
  ]);

  const [input, setInput] = useState("");

  const chatEndRef = useRef(null);

  const hasSelectedDoc = useMemo(() => !!selectedDoc, [selectedDoc]);

  // Auto scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();
      setBackendStatus(data?.status || "OK");
    } catch (e) {
      setBackendStatus("Backend not reachable");
    }
  }

  async function refreshDocs(selectFirstIfEmpty = false) {
    try {
      const res = await fetch(`${API_BASE}/list_docs`);
      const data = await res.json();
      const list = Array.isArray(data?.documents) ? data.documents : [];
      setDocs(list);

      if (selectFirstIfEmpty && !selectedDoc && list.length > 0) {
        setSelectedDoc(list[0]);
      }
    } catch (e) {
      // ignore
    }
  }

  useEffect(() => {
    checkHealth();
    refreshDocs(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleUpload() {
    if (!fileToUpload) return;
    setUploading(true);

    try {
      const form = new FormData();
      form.append("file", fileToUpload);

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || "Upload failed");
      }

      const data = await res.json();

      await refreshDocs(false);
      setSelectedDoc(data?.filename || selectedDoc);

      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "system",
          text: `Uploaded: ${data?.filename} • chunks: ${data?.total_chunks} • dim: ${data?.embedding_dim}`,
          meta: { type: "upload", doc: data?.filename || "" },
        },
      ]);

      setFileToUpload(null);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "system",
          text: `Upload error: ${e.message}`,
          meta: { type: "error" },
        },
      ]);
    } finally {
      setUploading(false);
    }
  }

  async function handleSend() {
    const q = input.trim();
    if (!q) return;

    if (!selectedDoc) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "system",
          text: "Select a document first.",
          meta: { type: "error" },
        },
      ]);
      return;
    }

    const userMsg = {
      id: crypto.randomUUID(),
      role: "user",
      text: q,
      meta: { doc: selectedDoc },
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);

    try {
      const res = await fetch(`${API_BASE}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: selectedDoc,
          query: q,
          top_k: 5,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || "Answer failed");
      }

      const data = await res.json();

      const topScore =
        Array.isArray(data?.citations) && data.citations.length > 0
          ? Number(data.citations[0].score ?? 0)
          : 0;

      const conf = confidenceFromScore(topScore);

      const assistantMsg = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: data?.answer || "",
        meta: {
          type: "answer",
          doc: selectedDoc,
          confidence: conf,
          topScore,
          citations: data?.citations || [],
          expanded: false,
        },
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "system",
          text: `Answer error: ${e.message}`,
          meta: { type: "error" },
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  async function handleSummarize() {
    if (!selectedDoc) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "system",
          text: "Select a document first.",
          meta: { type: "error" },
        },
      ]);
      return;
    }
  
    setSummarizing(true);
  
    // Add user-style message for natural chat flow
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "user",
        text: "Summarize this document.",
        meta: { doc: selectedDoc },
      },
    ]);
  
    try {
      const res = await fetch(`${API_BASE}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: selectedDoc,
          intro_chunks: 3,
          top_k: 5,
          max_output_tokens: 350,
        }),
      });
  
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || "Summarize failed");
      }
  
      const data = await res.json();
  
      const assistantMsg = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: data?.summary || "",
        meta: {
          type: "summary",
          doc: selectedDoc,
          citations: data?.citations || [],
          expanded: false,
        },
      };
  
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "system",
          text: `Summarize error: ${e.message}`,
          meta: { type: "error" },
        },
      ]);
    } finally {
      setSummarizing(false);
    }
  }  

  function toggleSources(id) {
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id !== id) return m;
        return {
          ...m,
          meta: { ...m.meta, expanded: !m.meta?.expanded },
        };
      })
    );
  }

  function clearChat() {
    setMessages([]);
  }

  function copyText(text) {
    if (!text) return;
    navigator.clipboard?.writeText(text).catch(() => {});
  }

  return (
    <div className="h-screen w-screen flex bg-gray-50 text-gray-900">
      {/* Sidebar */}
      <aside className="w-[320px] border-r bg-white p-4 flex flex-col gap-4">
        <div>
          <div className="text-xl font-semibold">SecRAG</div>
          <div className="text-sm text-gray-500">Day 8 UI polish + Summarize</div>
        </div>

        <div className="text-sm text-gray-600">
          Backend: <span className="font-mono">{API_BASE}</span>
        </div>

        <div className="text-xs text-gray-500">
          Status: <span className="font-medium">{backendStatus}</span>
        </div>

        {/* Upload */}
        <div className="border rounded-lg p-3 bg-gray-50">
          <div className="font-medium mb-2">Upload PDF</div>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFileToUpload(e.target.files?.[0] || null)}
          />
          <button
            className="mt-2 w-full rounded-md bg-black text-white py-2 disabled:opacity-50"
            onClick={handleUpload}
            disabled={!fileToUpload || uploading}
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>

          <div className="mt-2 text-xs text-gray-500">
            Uploads to backend <span className="font-mono">/upload</span>
          </div>
        </div>

        {/* Docs */}
        <div className="border rounded-lg p-3 bg-white">
          <div className="flex items-center justify-between mb-2">
            <div className="font-medium">Documents</div>
            <button
              className="text-sm px-2 py-1 rounded border bg-gray-50"
              onClick={() => refreshDocs(false)}
            >
              Refresh
            </button>
          </div>

          <div className="max-h-[340px] overflow-auto border rounded">
            {docs.length === 0 ? (
              <div className="p-3 text-sm text-gray-500">No documents yet.</div>
            ) : (
              docs.map((d) => (
                <button
                  key={d}
                  onClick={() => setSelectedDoc(d)}
                  className={`w-full text-left px-3 py-2 border-b last:border-b-0 hover:bg-gray-50 ${
                    selectedDoc === d ? "bg-gray-100 font-medium" : ""
                  }`}
                >
                  {prettyFile(d)}
                </button>
              ))
            )}
          </div>

          <div className="mt-2 text-xs text-gray-500">
            Selected:{" "}
            <span className="font-medium">{selectedDoc || "None"}</span>
          </div>

          {/* ✅ NEW: Summarize button */}
          <button
            className="mt-3 w-full rounded-md border bg-white py-2 disabled:opacity-50"
            onClick={handleSummarize}
            disabled={!hasSelectedDoc || summarizing}
            title="Summarize the selected document using first 5 chunks"
          >
            {summarizing ? "Summarizing..." : "Summarize document"}
          </button>

          {/* Existing controls */}
          <div className="mt-3 flex gap-2">
            <button
              className="flex-1 rounded-md border bg-white py-2 text-sm"
              onClick={clearChat}
            >
              Clear chat
            </button>
            <button
              className="flex-1 rounded-md border bg-white py-2 text-sm"
              onClick={() =>
                setInput("What is this document about? Give a short answer.")
              }
            >
              Sample
            </button>
          </div>
        </div>
      </aside>

      {/* Main chat */}
      <main className="flex-1 flex flex-col">
        <header className="h-14 border-b bg-white px-6 flex items-center justify-between">
          <div className="font-medium">Chat</div>
          <div className="text-sm text-gray-500">
            {sending ? "Thinking..." : "Ready"}
          </div>
        </header>

        <section className="flex-1 overflow-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="text-sm text-gray-500">
              Ask a question about the selected PDF, or click “Summarize
              document”.
            </div>
          ) : (
            messages.map((m) => (
              <div
                key={m.id}
                className={`max-w-3xl ${
                  m.role === "user" ? "ml-auto" : "mr-auto"
                }`}
              >
                <div className="text-xs text-gray-500 mb-1">
                  {m.role === "user" ? "You" : m.role === "assistant" ? "SecRAG" : "System"}
                  {m.meta?.doc ? ` • ${m.meta.doc}` : ""}
                  {m.meta?.type === "answer" && (
                    <>
                      {" "}
                      • Confidence:{" "}
                      <span className={m.meta.confidence?.tone || ""}>
                        {m.meta.confidence?.label || "Low"}{" "}
                        {typeof m.meta.topScore === "number"
                          ? `(top score ${m.meta.topScore.toFixed(3)})`
                          : ""}
                      </span>
                    </>
                  )}
                  {m.meta?.type === "summary" && (
                    <> • Summary</>
                  )}
                </div>

                <div className="rounded-lg border bg-white p-4 shadow-sm">
                  <div className="whitespace-pre-wrap leading-relaxed">
                    {m.text}
                  </div>

                  {m.role === "assistant" && (
                    <div className="mt-3 flex gap-2">
                      <button
                        className="text-sm px-3 py-1 rounded border bg-gray-50"
                        onClick={() => copyText(m.text)}
                      >
                        Copy
                      </button>

                      {(m.meta?.citations?.length || 0) > 0 && (
                        <button
                          className="text-sm px-3 py-1 rounded border bg-gray-50"
                          onClick={() => toggleSources(m.id)}
                        >
                          {m.meta?.expanded ? "Hide Sources" : `Sources (${m.meta.citations.length})`}
                        </button>
                      )}
                    </div>
                  )}

                  {m.role === "assistant" &&
                    m.meta?.expanded &&
                    (m.meta?.citations?.length || 0) > 0 && (
                      <div className="mt-3 border-t pt-3 space-y-2">
                        {m.meta.citations.map((c, idx) => (
                          <div
                            key={`${m.id}-${idx}`}
                            className="rounded-md border bg-gray-50 p-3 text-sm"
                          >
                            <div className="font-medium">
                              Chunk {c.chunk_id ?? "?"}
                              {typeof c.score === "number"
                                ? ` • score ${c.score.toFixed(3)}`
                                : ""}
                            </div>
                            {Array.isArray(c.char_range) && c.char_range.length === 2 && (
                              <div className="text-xs text-gray-600">
                                char range: {c.char_range[0]}–{c.char_range[1]}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </section>

        <footer className="border-t bg-white p-4">
          <div className="max-w-4xl mx-auto flex gap-3">
            <textarea
              className="flex-1 border rounded-lg p-3 resize-none h-[52px] focus:outline-none focus:ring-2 focus:ring-black/20"
              placeholder="Ask a question about the selected PDF..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!sending) handleSend();
                }
              }}
            />
            <button
              className="w-[110px] rounded-lg bg-black text-white disabled:opacity-50"
              onClick={handleSend}
              disabled={sending || !selectedDoc}
            >
              {sending ? "..." : "Send"}
            </button>
          </div>
          <div className="max-w-4xl mx-auto text-xs text-gray-500 mt-2">
            Tip: Press Enter to send • Shift+Enter for newline
          </div>
        </footer>
      </main>
    </div>
  );
}