"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnswerText } from "../components/AnswerText";
import { SourceCard, type SourceView } from "../components/SourceCard";
import { type Health, type Mode, type Timings, chat, getHealth, search } from "../lib/api";

type Turn = {
  question: string;
  answer: string;
  sources: SourceView[];
  status: "retrieving" | "answering" | "done" | "error";
  error?: string;
  timings?: Timings;
  latency?: { retrieval_ms: number; first_token_ms: number; total_ms: number };
  searchOnly?: boolean;
};

const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "hybrid", label: "Hybrid", hint: "Keyword and vector search, combined" },
  { value: "bm25", label: "Keyword", hint: "BM25 only" },
  { value: "dense", label: "Vector", hint: "Embeddings only" },
];

const EXAMPLES = [
  "My React app on port 3000 can't call my API on port 8000",
  "Close the database session after every request",
  "HTTPException with custom headers",
];

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("hybrid");
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [activeSource, setActiveSource] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const busy = turns.some((t) => t.status === "retrieving" || t.status === "answering");

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealthError("Can't reach the API. Start the backend, then reload this page."));
  }, []);

  const update = useCallback((i: number, patch: Partial<Turn> | ((t: Turn) => Partial<Turn>)) => {
    setTurns((ts) => ts.map((t, j) => (j === i ? { ...t, ...(typeof patch === "function" ? patch(t) : patch) } : t)));
  }, []);

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    setInput("");
    setActiveSource(null);
    const i = turns.length;
    const history = turns
      .filter((t) => t.status === "done" && !t.searchOnly)
      .flatMap((t) => [
        { role: "user" as const, content: t.question },
        { role: "assistant" as const, content: t.answer },
      ]);
    setTurns((ts) => [...ts, { question: q, answer: "", sources: [], status: "retrieving" }]);

    if (!health?.chat_enabled) {
      try {
        const r = await search(q, mode, 6);
        update(i, {
          status: "done",
          searchOnly: true,
          timings: r.timings_ms,
          sources: r.hits.map((h, n) => ({
            key: h.id,
            n: n + 1,
            title: [h.title, ...h.heading_path].filter(Boolean).join(" › "),
            docId: h.doc_id,
            url: h.url,
            text: h.text,
            ranks: { bm25: h.bm25_rank, dense: h.dense_rank, fused: h.fused_rank },
          })),
        });
      } catch (e) {
        update(i, { status: "error", error: (e as Error).message });
      }
      return;
    }

    abortRef.current = new AbortController();
    try {
      await chat(
        q,
        history,
        mode,
        {
          onSources: (d) =>
            update(i, {
              status: "answering",
              timings: d.timings_ms,
              sources: d.sources.map((s) => ({
                key: s.chunk_ids.join("-"),
                n: s.n,
                title: s.title,
                docId: s.doc_id,
                url: s.url,
                text: s.text,
                ranks: s.ranks,
              })),
            }),
          onToken: (text) => update(i, (t) => ({ answer: t.answer + text })),
          onCitations: (d) =>
            update(i, (t) => ({ sources: t.sources.map((s) => ({ ...s, cited: d.cited.includes(s.n) })) })),
          onDone: (d) => update(i, { status: "done", latency: d }),
          onError: (message) => update(i, { status: "error", error: message }),
        },
        abortRef.current.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") update(i, { status: "error", error: "The connection to the API dropped." });
    } finally {
      update(i, (t) => (t.status === "answering" || t.status === "retrieving" ? { status: "done" } : {}));
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function showSource(n: number) {
    setActiveSource(n);
    document.getElementById(`source-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  const current = turns[turns.length - 1];

  return (
    <div className="shell">
      <header className="masthead">
        <div className="brand">
          <span className="brand-mark" aria-hidden />
          <h1>NexusRAG</h1>
        </div>
        <p className="corpus">
          {health
            ? `${health.index.documents} pages, ${health.index.chunks.toLocaleString()} passages, embedded with ${health.index.embedder}${health.reranker !== "none" ? ` and re-ranked by ${health.reranker}` : ""}.`
            : (healthError ?? "Connecting…")}
        </p>
      </header>

      <main className="layout">
        <section className="conversation" aria-live="polite">
          {turns.length === 0 && (
            <div className="intro">
              <h2>Ask the documentation.</h2>
              <p>
                Answers quote only what retrieval found, with numbered citations. Each source shows where it ranked in
                keyword search, vector search, and the combined list.
              </p>
              {health && !health.chat_enabled && (
                <p className="notice">
                  Answer generation is off because the server has no Anthropic API key. You'll get the ranked passages
                  instead.
                </p>
              )}
              <ul className="examples">
                {EXAMPLES.map((e) => (
                  <li key={e}>
                    <button type="button" onClick={() => ask(e)} disabled={!health}>
                      {e}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {turns.map((t, i) => (
            <div className="turn" key={i}>
              <p className="question">{t.question}</p>
              {t.status === "retrieving" && <p className="status">Searching {health?.index.chunks.toLocaleString()} passages…</p>}
              {t.searchOnly && t.status === "done" && (
                <p className="status">
                  {t.sources.length ? `Showing the top ${t.sources.length} passages under Sources.` : "Nothing matched. Try other words."}
                </p>
              )}
              {t.answer && (
                <AnswerText text={t.answer} sourceCount={t.sources.length} onCite={(n) => i === turns.length - 1 && showSource(n)} />
              )}
              {t.status === "answering" && !t.answer && <p className="status">Reading {t.sources.length} sources…</p>}
              {t.status === "error" && <p className="error">{t.error}</p>}
              {t.status === "done" && (t.latency || t.timings) && (
                <p className="timing">
                  {t.latency
                    ? `Found sources in ${Math.round(t.latency.retrieval_ms)} ms, first words after ${(t.latency.first_token_ms / 1000).toFixed(1)} s, finished in ${(t.latency.total_ms / 1000).toFixed(1)} s.`
                    : `Found in ${t.timings!.total.toFixed(1)} ms.`}
                </p>
              )}
            </div>
          ))}

          <form
            className="ask"
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
          >
            <label htmlFor="q" className="visually-hidden">
              Your question
            </label>
            <textarea
              id="q"
              rows={2}
              value={input}
              placeholder="Ask about the docs"
              maxLength={2000}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input);
                }
              }}
            />
            <div className="ask-row">
              <fieldset className="modes">
                <legend className="visually-hidden">Retrieval mode</legend>
                {MODES.map((m) => (
                  <label key={m.value} title={m.hint} className={mode === m.value ? "on" : ""}>
                    <input type="radio" name="mode" value={m.value} checked={mode === m.value} onChange={() => setMode(m.value)} />
                    {m.label}
                  </label>
                ))}
              </fieldset>
              {busy ? (
                <button type="button" className="primary" onClick={stop}>
                  Stop
                </button>
              ) : (
                <button type="submit" className="primary" disabled={!input.trim() || !health}>
                  Ask
                </button>
              )}
            </div>
          </form>
        </section>

        <aside className="sources" aria-label="Sources">
          <h2 className="rail-title">Sources</h2>
          {!current || current.sources.length === 0 ? (
            <p className="rail-empty">Sources for your question appear here, in the order retrieval ranked them.</p>
          ) : (
            current.sources.map((s) => <SourceCard key={s.key} s={s} active={activeSource === s.n} />)
          )}
          {current && current.sources.length > 0 && (
            <p className="legend">
              <span className="swatch kw" /> keyword rank <span className="swatch vec" /> vector rank{" "}
              <span className="swatch fused" /> combined rank. Further left is better.
            </p>
          )}
        </aside>
      </main>
    </div>
  );
}
