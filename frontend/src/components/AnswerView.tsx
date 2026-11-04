import type { Done } from "../api";

interface Props {
  text: string;
  done: Done | null;
  error: string | null;
  busy: boolean;
  onCite: (nctId: string) => void;
}

/** Turns "[NCT01234567]" in the streamed answer into buttons that open the source record. */
function withCitations(text: string, onCite: (id: string) => void) {
  return text.split(/(\[NCT\d{8}\])/g).map((part, i) => {
    const m = part.match(/^\[(NCT\d{8})\]$/);
    return m ? (
      <button key={i} className="cite" onClick={() => onCite(m[1])} title="Open the source record">
        {m[1]}
      </button>
    ) : (
      <span key={i}>{part}</span>
    );
  });
}

export function AnswerView({ text, done, error, busy, onCite }: Props) {
  return (
    <section className="panel answer" aria-live="polite">
      <h2>Answer</h2>
      {error && <p className="error">{error}</p>}
      {!text && !error && <p className="muted">{busy ? "Retrieving and generating…" : "Ask a question about the trials."}</p>}
      {text && (
        <p className="text">
          {withCitations(text, onCite)}
          {busy && <span className="caret" />}
        </p>
      )}
      {done && (
        <div className="badges">
          {done.abstained && <span className="badge warn">abstained</span>}
          {done.unsupported_citations.length > 0 && (
            <span className="badge bad">cites a trial not in the sources: {done.unsupported_citations.join(", ")}</span>
          )}
          {!done.abstained && done.citations.length === 0 && <span className="badge warn">no citation</span>}
          <span className="badge">first token {done.first_token_ms} ms</span>
          <span className="badge">total {done.total_ms} ms</span>
        </div>
      )}
    </section>
  );
}
