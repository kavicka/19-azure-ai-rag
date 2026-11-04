// Typed client for the FastAPI backend. /api is proxied by Vite in dev (see vite.config.ts).

export type Mode = "no_rag" | "bm25" | "dense" | "hybrid" | "rerank";

export interface Hit {
  chunk_id: string;
  nct_id: string;
  title: string;
  section: string;
  text: string;
  score: number;
  scores: Partial<Record<"bm25" | "dense" | "rrf" | "rerank", number>>;
}

export interface Done {
  citations: string[];
  abstained: boolean;
  unsupported_citations: string[];
  first_token_ms: number;
  total_ms: number;
}

export interface Health {
  llm: string;
  search: string;
  chunking: string;
  n_chunks: number;
  n_studies: number;
  modes: Mode[];
}

export interface Study {
  nct_id: string;
  title: string;
  url: string;
  sections: { name: string; text: string }[];
}

export interface AskHandlers {
  onSources: (hits: Hit[], retrievalMs: number) => void;
  onToken: (text: string) => void;
  onDone: (d: Done) => void;
  onError: (message: string) => void;
}

export async function getHealth(): Promise<Health> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`health: HTTP ${r.status}`);
  return r.json();
}

export async function getStudy(nctId: string): Promise<Study> {
  const r = await fetch(`/api/studies/${encodeURIComponent(nctId)}`);
  if (!r.ok) throw new Error(`study ${nctId}: HTTP ${r.status}`);
  return r.json();
}

/**
 * POST /api/ask and parse the Server-Sent Events stream by hand.
 * EventSource only supports GET, and the question belongs in a POST body, so we read fetch's body stream:
 * events are separated by a blank line, each has an "event:" and a "data:" line.
 */
export async function ask(question: string, mode: Mode, k: number, h: AskHandlers, signal: AbortSignal) {
  const r = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode, k }),
    signal,
  });
  if (!r.ok || !r.body) {
    h.onError(`HTTP ${r.status}: ${await r.text()}`);
    return;
  }
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let sep: number;
    while ((sep = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const event = raw.match(/^event: (.*)$/m)?.[1];
      const data = JSON.parse(raw.match(/^data: (.*)$/m)?.[1] ?? "null");
      if (event === "sources") h.onSources(data.hits, data.retrieval_ms);
      else if (event === "token") h.onToken(data.text);
      else if (event === "done") h.onDone(data);
      else if (event === "error") h.onError(data.message);
    }
  }
}
