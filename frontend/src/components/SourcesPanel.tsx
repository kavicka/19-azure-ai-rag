import type { Hit, Mode } from "../api";

interface Props {
  hits: Hit[];
  mode: Mode;
  retrievalMs: number | null;
  cited: string[];
  onOpen: (nctId: string) => void;
}

const STAGES = ["bm25", "dense", "rrf", "rerank"] as const;

/**
 * Shows *why* the model answered what it answered: every retrieved chunk with the score of each stage.
 * A chunk with a high BM25 score but no dense score was found only by keywords, and vice versa.
 */
export function SourcesPanel({ hits, mode, retrievalMs, cited, onOpen }: Props) {
  if (mode === "no_rag") {
    return (
      <section className="panel sources">
        <h2>Sources</h2>
        <p className="muted">No retrieval in this mode: the model answers from its own weights only.</p>
      </section>
    );
  }
  // scale each stage's bar to the best score in this result list (scores of different stages are not comparable)
  const max = Object.fromEntries(
    STAGES.map((s) => [s, Math.max(...hits.map((h) => Math.abs(h.scores[s] ?? 0)), 1e-9)]),
  ) as Record<(typeof STAGES)[number], number>;

  return (
    <section className="panel sources">
      <h2>
        Sources {retrievalMs !== null && <span className="muted small">retrieval {retrievalMs} ms</span>}
      </h2>
      {hits.length === 0 && <p className="muted">Retrieved chunks appear here.</p>}
      <ol>
        {hits.map((h) => (
          <li key={h.chunk_id} className={cited.includes(h.nct_id) ? "cited" : ""}>
            <div className="hit-head">
              <button className="cite" onClick={() => onOpen(h.nct_id)}>
                {h.nct_id}
              </button>
              <span className="section">{h.section}</span>
              {cited.includes(h.nct_id) && <span className="badge ok">cited</span>}
            </div>
            <div className="hit-title">{h.title}</div>
            <div className="scores">
              {STAGES.filter((s) => h.scores[s] !== undefined).map((s) => (
                <div key={s} className="score" title={`${s} = ${h.scores[s]}`}>
                  <span>{s}</span>
                  <div className="bar">
                    <div style={{ width: `${(100 * Math.abs(h.scores[s]!)) / max[s]}%` }} />
                  </div>
                  <code>{h.scores[s]!.toFixed(3)}</code>
                </div>
              ))}
            </div>
            <details>
              <summary>chunk text</summary>
              <pre>{h.text}</pre>
            </details>
          </li>
        ))}
      </ol>
    </section>
  );
}
