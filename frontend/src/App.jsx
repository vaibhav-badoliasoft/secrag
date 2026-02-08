import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPostJson(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function apiPostForm(path, formData) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function confidenceLabel(topScore) {
  if (topScore >= 0.6) return "High";
  if (topScore >= 0.45) return "Medium";
  return "Low";
}

function formatTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

export default function App() {
  const [docs, setDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [uploading, setUploading] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(false);

  const [query, setQuery] = useState("");
  const [asking, setAsking] = useState(false);

  const [messages, setMessages] = useState([]); // {role, text, ts, sources?, meta?}
  const [error, setError] = useState("");

  const bottomRef = useRef(null);

  const canAsk = useMemo(
    () => selectedDoc && query.trim().length > 0 && !asking,
    [selectedDoc, query, asking]
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function refreshDocs() {
    setError("");
    setLoadingDocs(true);
    try {
      const data = await apiGet("/list_docs");
      const list = data.documents || [];
      setDocs(list);
      setSelectedDoc((prev) => (list.includes(prev) ? prev : ""));
    } catch (e) {
      setError(`Docs load failed: ${e.message}`);
    } finally {
      setLoadingDocs(false);
    }
  }

  useEffect(() => {
    refreshDocs();
  }, []);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setError("");
    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      await apiPostForm("/upload", fd);

      await refreshDocs();
      setSelectedDoc(file.name.toLowerCase().endsWith(".pdf") ? file.name : `${file.name}.pdf`);
    } catch (e) {
      setError(`Upload failed: ${e.message}`);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // ignore
    }
  }

  async function ask() {
    if (!canAsk) return;
    setError("");
    setAsking(true);

    const userText = query.trim();
    setQuery("");

    setMessages((m) => [...m, { role: "user", text: userText, ts: Date.now() }]);

    try {
      const data = await apiPostJson("/answer", {
        filename: selectedDoc,
        query: userText,
        top_k: 5,
      });

      const citations = data.citations || [];
      const topScore = Number(citations?.[0]?.score ?? 0);

      const sources = citations.map((c) => ({
        chunk_id: c.chunk_id,
        score: c.score,
        char_range: c.char_range,
      }));

      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: data.answer || "No answer returned.",
          ts: Date.now(),
          meta: {
            confidence: confidenceLabel(topScore),
            topScore,
          },
          sources,
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: "Something went wrong while answering.", ts: Date.now() },
      ]);
      setError(`Answer failed: ${e.message}`);
    } finally {
      setAsking(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  }

  const sampleQuestions = [
    "Explain working memory",
    "What is connectionist theory?",
    "Explain cognitive load in simple words",
  ];

  return (
    <div className="h-screen w-screen bg-gray-100 text-gray-900">
      <div className="h-full grid grid-cols-12">
        {/* Sidebar */}
        <aside className="col-span-3 border-r bg-white p-4 flex flex-col gap-4 shadow-sm">
          <div>
            <div className="text-xl font-semibold">SecRAG</div>
            <div className="text-xs text-gray-500">Day 7 — UI MVP</div>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">Upload PDF</label>
            <input
              type="file"
              accept="application/pdf"
              onChange={handleUpload}
              disabled={uploading}
              className="text-sm"
            />
            <div className="text-xs text-gray-500">
              {uploading ? "Uploading..." : "Uploads to backend /upload"}
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="text-sm font-medium">Documents</div>
            <button
              onClick={refreshDocs}
              disabled={loadingDocs}
              className="text-xs px-2 py-1 rounded border hover:bg-gray-50 disabled:opacity-60"
            >
              {loadingDocs ? "..." : "Refresh"}
            </button>
          </div>

          <div className="flex-1 overflow-auto border rounded bg-white">
            {docs.length === 0 ? (
              <div className="p-3 text-sm text-gray-500">No PDFs found yet.</div>
            ) : (
              <ul className="divide-y">
                {docs.map((d) => (
                  <li key={d}>
                    <button
                      onClick={() => setSelectedDoc(d)}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-50 ${
                        selectedDoc === d ? "bg-gray-100 font-medium" : ""
                      }`}
                    >
                      {d}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="text-xs">
            <div className="text-gray-500">Selected</div>
            <div className="mt-1 inline-flex max-w-full items-center rounded bg-gray-100 px-2 py-1 text-gray-800">
              <span className="truncate">{selectedDoc || "None"}</span>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="col-span-9 flex flex-col">
          {/* Header */}
          <div className="border-b bg-white p-4 flex items-center justify-between">
            <div>
              <div className="font-semibold">Chat</div>
              <div className="text-xs text-gray-500">Backend: {API_BASE}</div>
            </div>

            <button
              onClick={() => setMessages([])}
              className="text-xs px-3 py-2 rounded border hover:bg-gray-50"
            >
              Clear chat
            </button>
          </div>

          {/* Error */}
          {error ? (
            <div className="mx-4 mt-3 p-3 rounded border border-red-200 bg-red-50 text-sm text-red-700">
              {error}
            </div>
          ) : null}

          {/* Messages */}
          <div className="flex-1 overflow-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="text-sm text-gray-500">
                Upload a PDF, select it, then ask a question.
              </div>
            ) : null}

            {messages.map((m, idx) => {
              const isUser = m.role === "user";
              return (
                <div key={idx} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                  <div className="max-w-3xl rounded-2xl border bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <div className="text-xs text-gray-500">
                        {isUser ? "You" : "SecRAG"} • {formatTime(m.ts)}
                        {!isUser && m.meta ? (
                          <span className="ml-2">
                            • Confidence: {m.meta.confidence} (top score{" "}
                            {Number(m.meta.topScore).toFixed(3)})
                          </span>
                        ) : null}
                      </div>

                      {!isUser ? (
                        <button
                          onClick={() => copyToClipboard(m.text)}
                          className="text-xs px-2 py-1 rounded border hover:bg-gray-50"
                        >
                          Copy
                        </button>
                      ) : null}
                    </div>

                    <div className="whitespace-pre-wrap text-sm leading-relaxed">
                      {m.text}
                    </div>

                    {!isUser && m.sources?.length ? (
                      <details className="mt-3">
                        <summary className="cursor-pointer text-sm font-medium">
                          Sources ({m.sources.length})
                        </summary>
                        <div className="mt-2 space-y-2">
                          {m.sources.map((s) => (
                            <div
                              key={`${s.chunk_id}-${s.score}`}
                              className="text-xs p-3 rounded border bg-gray-50"
                            >
                              <div className="flex items-center justify-between">
                                <div>
                                  Chunk <span className="font-medium">{s.chunk_id}</span> •
                                  score {Number(s.score).toFixed(3)}
                                </div>
                                <div className="text-gray-500">
                                  {s.char_range?.[0]}–{s.char_range?.[1]}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </details>
                    ) : null}
                  </div>
                </div>
              );
            })}

            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="border-t bg-white p-4">
            <div className="mb-3 flex gap-2 flex-wrap">
              {sampleQuestions.map((q) => (
                <button
                  key={q}
                  onClick={() => setQuery(q)}
                  className="text-xs px-3 py-1 rounded-full border hover:bg-gray-50"
                  disabled={!selectedDoc || asking}
                >
                  {q}
                </button>
              ))}
            </div>

            <div className="flex gap-3">
              <textarea
                className="flex-1 resize-none rounded-xl border p-3 text-sm focus:outline-none focus:ring-2 focus:ring-gray-200 disabled:bg-gray-100"
                rows={2}
                placeholder={
                  selectedDoc
                    ? "Ask a question about the selected PDF…"
                    : "Select a document first…"
                }
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                disabled={!selectedDoc || asking}
              />

              <button
                onClick={ask}
                disabled={!canAsk}
                className="px-5 py-2 rounded-xl bg-gray-900 text-white text-sm disabled:opacity-50"
              >
                {asking ? "Asking..." : "Send"}
              </button>
            </div>

            <div className="mt-2 text-xs text-gray-500">
              Enter to send • Shift+Enter for newline
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
