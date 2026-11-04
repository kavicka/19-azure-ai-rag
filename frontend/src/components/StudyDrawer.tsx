import { useEffect, useState } from "react";
import { getStudy, type Study } from "../api";

/** The full source record, so every citation can be checked by a human in one click. */
export function StudyDrawer({ nctId, onClose }: { nctId: string; onClose: () => void }) {
  const [study, setStudy] = useState<Study | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setStudy(null);
    setError(null);
    getStudy(nctId).then(setStudy, (e: Error) => setError(e.message));
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nctId, onClose]);

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label={`Record ${nctId}`}>
        <button className="close" onClick={onClose} aria-label="Close">
          ×
        </button>
        {error && <p className="error">{error}</p>}
        {!study && !error && <p className="muted">Loading {nctId}…</p>}
        {study && (
          <>
            <p className="muted">
              <a href={study.url} target="_blank" rel="noreferrer">
                {study.nct_id} on ClinicalTrials.gov ↗
              </a>
            </p>
            <h2>{study.title}</h2>
            {study.sections.map((s) => (
              <section key={s.name}>
                <h3>{s.name}</h3>
                <p className="pre">{s.text}</p>
              </section>
            ))}
          </>
        )}
      </aside>
    </div>
  );
}
