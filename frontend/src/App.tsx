import { useEffect, useRef, useState } from "react";
import { ask, getHealth, type Done, type Health, type Hit, type Mode } from "./api";
import { AnswerView } from "./components/AnswerView";
import { SourcesPanel } from "./components/SourcesPanel";
import { StudyDrawer } from "./components/StudyDrawer";

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "no_rag", label: "No RAG", hint: "LLM alone: the baseline" },
  { id: "bm25", label: "BM25", hint: "keyword match" },
  { id: "dense", label: "Dense", hint: "embedding similarity" },
  { id: "hybrid", label: "Hybrid", hint: "BM25 + dense, fused with RRF" },
  { id: "rerank", label: "Rerank", hint: "hybrid → cross-encoder" },
];

const EXAMPLES = [
  'How many participants were enrolled in the trial "Diabetes Prevention Program Outcomes Study"?',
  "Which psoriasis trials tested bimekizumab?",
  'What was the total budget of the trial "Diabetes Prevention Program Outcomes Study"?',
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [mode, setMode] = useState<Mode>("rerank");
  const [k, setK] = useState(5);
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [retrievalMs, setRetrievalMs] = useState<number | null>(null);
  const [done, setDone] = useState<Done | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [openStudy, setOpenStudy] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getHealth().then(setHealth, (e: Error) => setHealthError(e.message));
  }, []);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    abortRef.current?.abort(); // a new question cancels the stream of the previous one
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setAnswer("");
    setHits([]);
    setDone(null);
    setError(null);
    setRetrievalMs(null);
    setBusy(true);
    try {
      await ask(
        question,
        mode,
        k,
        {
          onSources: (h, ms) => {
            setHits(h);
            setRetrievalMs(ms);
          },
          onToken: (t) => setAnswer((a) => a + t),
          onDone: setDone,
          onError: setError,
        },
        ctrl.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Clinical Trials RAG</h1>
        <p className="muted">
          {health
            ? `${health.n_studies} trials · ${health.n_chunks} chunks (${health.chunking}) · search ${health.search} · LLM ${health.llm}`
            : healthError
              ? `Backend not reachable: ${healthError}`
              : "Connecting…"}
        </p>
      </header>

      <form className="ask" onSubmit={submit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
          rows={2}
          maxLength={500}
          aria-label="Question"
        />
        <div className="controls">
          <div className="modes" role="radiogroup" aria-label="Retrieval mode">
            {MODES.map((m) => (
              <button
                type="button"
                key={m.id}
                role="radio"
                aria-checked={mode === m.id}
                className={mode === m.id ? "on" : ""}
                title={m.hint}
                onClick={() => setMode(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
          <label className="k">
            top-k <input type="number" min={1} max={10} value={k} onChange={(e) => setK(Number(e.target.value))} />
          </label>
          <button type="submit" className="primary" disabled={busy || question.trim().length < 3}>
            {busy ? "Answering…" : "Ask"}
          </button>
        </div>
        <div className="examples">
          {EXAMPLES.map((q) => (
            <button type="button" key={q} onClick={() => setQuestion(q)}>
              {q}
            </button>
          ))}
        </div>
      </form>

      <main>
        <AnswerView text={answer} done={done} error={error} busy={busy} onCite={setOpenStudy} />
        <SourcesPanel hits={hits} mode={mode} retrievalMs={retrievalMs} cited={done?.citations ?? []} onOpen={setOpenStudy} />
      </main>

      {openStudy && <StudyDrawer nctId={openStudy} onClose={() => setOpenStudy(null)} />}
    </div>
  );
}
