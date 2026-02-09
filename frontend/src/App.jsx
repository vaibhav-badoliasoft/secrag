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

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  // Drawer
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerTitle, setDrawerTitle] = useState("Sources");
  const [drawerItems, setDrawerItems] = useState([]);

  // Toasts
  const [toasts, setToasts] = useState([]);

  // ✅ Dynamic sample questions
  const [sampleQs, setSampleQs] = useState([]);
  const [loadingSamples, setLoadingSamples] = useState(false);

  const chatEndRef = useRef(null);

  const hasSelectedDoc = useMemo(() => !!selectedDoc, [selectedDoc]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function pushToast(text) {
    const id = crypto.randomUUID();
    setToasts((prev) => [...prev, { id, text }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3500);
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();
      setBackendStatus(data?.status || "OK");
    } catch (e) {
      setBackendStatus("Backend not reachable");
      pushToast("Backend not reachable. Is uvicorn running?");
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
      pushToast("Failed to load documents list.");
    }
  }

  useEffect(() => {
    checkHealth();
    refreshDocs(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ✅ Load sample questions whenever selectedDoc changes
  useEffect(() => {
    async function loadSamples() {
      if (!selectedDoc) {
        setSampleQs([]);
        return;
      }

      setLoadingSamples(true);
      try {
        const res = await fetch(`${API_BASE}/sample_questions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename: selectedDoc,
            intro_chunks: 2,
            top_k: 4,
            max_output_tokens: 220,
          }),
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data?.detail || "Failed to load sample questions");

        const qs = Array.isArray(data?.questions) ? data.questions : [];
        setSampleQs(qs);
      } catch (e) {
        setSampleQs([]);
        pushToast(e.message || "Failed to load sample questions");
      } finally {
        setLoadingSamples(false);
      }
    }

    loadSamples();
  }, [selectedDoc]);

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
      pushToast(e.message || "Upload failed");
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

  function openDrawer(title, items) {
    setDrawerTitle(title || "Sources");
    setDrawerItems(Array.isArray(items) ? items : []);
    setDrawerOpen(true);
  }

  async function handleSend() {
    const q = input.trim();
    if (!q) return;

    if (!selectedDoc) {
      pushToast("Select a document first.");
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

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || "Answer failed");

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
        },
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e) {
      pushToast(e.message || "Answer failed");
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
      pushToast("Select a document first.");
      return;
    }

    setSummarizing(true);

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

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data?.detail || "Summarize failed");

      const assistantMsg = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: data?.summary || "",
        meta: {
          type: "summary",
          doc: selectedDoc,
          citations: data?.citations || [],
        },
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (e) {
      pushToast(e.message || "Summarize failed");
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

  function clearChat() {
    setMessages([]);
    setDrawerOpen(false);
    setDrawerItems([]);
  }

  function copyText(text) {
    if (!text) return;
    navigator.clipboard?.writeText(text).catch(() => {});
    pushToast("Copied.");
  }

  return (
    <div className="h-screen w-screen flex bg-gray-50 text-gray-900">
      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="max-w-[340px] rounded-lg bg-black text-white px-4 py-3 text-sm shadow-lg"
          >
            {t.text}
          </div>
        ))}
      </div>

      <aside className="w-[320px] border-r bg-white p-4 flex flex-col gap-4">
        <div>
          <div className="text-xl font-semibold">SecRAG</div>
          <div className="text-sm text-gray-500">Day 9 — Dynamic Samples</div>
        </div>

        <div className="text-sm text-gray-600">
          Backend: <span className="font-mono">{API_BASE}</span>
        </div>

        <div className="text-xs text-gray-500">
          Status: <span className="font-medium">{backendStatus}</span>
        </div>

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
        </div>

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

          <div className="max-h-[260px] overflow-auto border rounded">
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
            Selected: <span className="font-medium">{selectedDoc || "None"}</span>
          </div>

          <button
            className="mt-3 w-full rounded-md border bg-white py-2 disabled:opacity-50"
            onClick={handleSummarize}
            disabled={!hasSelectedDoc || summarizing}
          >
            {summarizing ? "Summarizing..." : "Summarize document"}
          </button>

          <div className="mt-3">
            <div className="text-xs text-gray-500 mb-2">Sample questions</div>

            {loadingSamples ? (
              <div className="text-sm text-gray-500">Loading sample questions…</div>
            ) : sampleQs.length === 0 ? (
              <div className="text-sm text-gray-500">
                No samples yet. (Select a doc or check API key.)
              </div>
            ) : (
              <div className="space-y-2">
                {sampleQs.slice(0, 4).map((q) => (
                  <button
                    key={q}
                    className="w-full text-left text-sm rounded border bg-gray-50 px-3 py-2 hover:bg-gray-100"
                    onClick={() => setInput(q)}
                    disabled={!selectedDoc}
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="mt-3 flex gap-2">
            <button
              className="flex-1 rounded-md border bg-white py-2 text-sm"
              onClick={clearChat}
            >
              Clear chat
            </button>
            <button
              className="flex-1 rounded-md border bg-white py-2 text-sm"
              onClick={() => setInput("What is this document about? Give a short answer.")}
              disabled={!selectedDoc}
            >
              Sample
            </button>
          </div>
        </div>
      </aside>

      <main className="flex-1 flex flex-col relative">
        <header className="h-14 border-b bg-white px-6 flex items-center justify-between">
          <div className="font-medium">Chat</div>
          <div className="text-sm text-gray-500">
            {sending ? "Thinking..." : summarizing ? "Summarizing..." : "Ready"}
          </div>
        </header>

        <section className="flex-1 overflow-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="text-sm text-gray-500">
              Ask a question or click “Summarize document”.
            </div>
          ) : (
            messages.map((m) => (
              <div
                key={m.id}
                className={`max-w-3xl ${m.role === "user" ? "ml-auto" : "mr-auto"}`}
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
                  {m.meta?.type === "summary" && <> • Summary</>}
                </div>

                <div className="rounded-lg border bg-white p-4 shadow-sm">
                  <div className="whitespace-pre-wrap leading-relaxed">{m.text}</div>

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
                          onClick={() =>
                            openDrawer(`Sources • ${m.meta?.doc || ""}`, m.meta?.citations || [])
                          }
                        >
                          Sources ({m.meta.citations.length})
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {(sending || summarizing) && (
            <div className="max-w-3xl mr-auto">
              <div className="rounded-lg border bg-white p-4 shadow-sm">
                <div className="h-3 bg-gray-200 rounded w-3/4 mb-2" />
                <div className="h-3 bg-gray-200 rounded w-2/3 mb-2" />
                <div className="h-3 bg-gray-200 rounded w-1/2" />
              </div>
            </div>
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
            Tip: Enter to send • Shift+Enter for newline
          </div>
        </footer>

        {drawerOpen && (
          <div className="absolute inset-y-0 right-0 w-[420px] bg-white border-l shadow-xl z-40 flex flex-col">
            <div className="h-14 px-4 border-b flex items-center justify-between">
              <div className="font-medium">{drawerTitle}</div>
              <button
                className="text-sm px-3 py-1 rounded border bg-gray-50"
                onClick={() => setDrawerOpen(false)}
              >
                Close
              </button>
            </div>

            <div className="flex-1 overflow-auto p-4 space-y-3">
              {drawerItems.length === 0 ? (
                <div className="text-sm text-gray-500">No sources.</div>
              ) : (
                drawerItems.map((c, idx) => (
                  <div key={idx} className="rounded-lg border bg-gray-50 p-3">
                    <div className="font-medium">
                      Chunk {c.chunk_id ?? "?"}
                      {typeof c.score === "number" ? ` • score ${c.score.toFixed(3)}` : ""}
                      {c.source ? ` • ${c.source}` : ""}
                    </div>

                    {Array.isArray(c.char_range) && c.char_range.length === 2 && (
                      <div className="text-xs text-gray-600 mt-1">
                        char range: {c.char_range[0]}–{c.char_range[1]}
                      </div>
                    )}

                    {c.preview && (
                      <div className="text-sm text-gray-800 mt-2 whitespace-pre-wrap">
                        {c.preview}
                        {c.preview.length >= 240 ? "…" : ""}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>

            <div className="p-4 border-t">
              <button
                className="w-full rounded-md border bg-white py-2 text-sm"
                onClick={() => {
                  const txt = drawerItems
                    .map((c) => `Chunk ${c.chunk_id} | score ${c.score}\n${c.preview || ""}`)
                    .join("\n\n---\n\n");
                  navigator.clipboard?.writeText(txt).catch(() => {});
                  pushToast("Sources copied.");
                }}
                disabled={drawerItems.length === 0}
              >
                Copy sources
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}