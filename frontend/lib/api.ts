import { readSSE } from "./sse";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type Mode = "hybrid" | "bm25" | "dense";

export type Health = {
  status: string;
  index: { chunks: number; documents: number; embedder: string; vector_store: string };
  reranker: string;
  chat_enabled: boolean;
  cache: string;
};

export type SearchHit = {
  id: string;
  doc_id: string;
  title: string;
  heading_path: string[];
  url: string | null;
  text: string;
  score: number;
  bm25_rank: number | null;
  dense_rank: number | null;
  fused_rank: number | null;
  rerank_score: number | null;
};

export type ChatSource = {
  n: number;
  doc_id: string;
  title: string;
  url: string | null;
  chunk_ids: string[];
  text: string;
  ranks: { bm25: number | null; dense: number | null; fused: number | null };
};

export type Timings = Record<string, number>;

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${API_URL}/health`);
  if (!r.ok) throw new Error(`The API answered ${r.status}`);
  return r.json();
}

export async function search(query: string, mode: Mode, k = 8) {
  const r = await fetch(`${API_URL}/search`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, mode, k }),
  });
  if (r.status === 429) throw new Error("Too many requests. Wait a minute and try again.");
  if (!r.ok) throw new Error(`Search failed with status ${r.status}`);
  return (await r.json()) as { hits: SearchHit[]; timings_ms: Timings; cached: boolean; reranker: string };
}

export type ChatHandlers = {
  onSources: (d: { sources: ChatSource[]; context_tokens: number; dropped: number; timings_ms: Timings }) => void;
  onToken: (text: string) => void;
  onCitations: (d: { cited: number[]; invalid: number[] }) => void;
  onDone: (d: { retrieval_ms: number; first_token_ms: number; total_ms: number }) => void;
  onError: (message: string) => void;
};

export async function chat(
  question: string,
  history: { role: "user" | "assistant"; content: string }[],
  mode: Mode,
  h: ChatHandlers,
  signal?: AbortSignal,
) {
  const r = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ question, history, mode }),
    signal,
  });
  if (r.status === 429) return h.onError("Too many requests. Wait a minute and try again.");
  if (!r.ok) return h.onError(`The API answered ${r.status}.`);
  for await (const ev of readSSE(r)) {
    const data = JSON.parse(ev.data);
    if (ev.event === "sources") h.onSources(data);
    else if (ev.event === "token") h.onToken(data.text);
    else if (ev.event === "citations") h.onCitations(data);
    else if (ev.event === "done") h.onDone(data);
    else if (ev.event === "error") h.onError(`Generation stopped: ${data.message}.`);
  }
}
